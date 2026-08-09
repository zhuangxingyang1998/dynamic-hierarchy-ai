"""Vectorized shared binary composer for the Stage 2 R5 diagnostic ladder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from .stage2_ladder_config import LadderModelSpec
from .data import ADD, SUB
from .stage2_config import Stage2ModelSpec
from .stage2_ladder_data import (
    ADD_FIRST,
    ADD_OP,
    LadderGeneratedSplit,
    LadderModelInput,
    SUB_OP,
    SUB_FIRST,
    to_stage2_oracle_batch,
)
from .stage2_model import Stage2MergeClassifier


@dataclass(frozen=True)
class LadderModelOutput:
    root_logits: torch.Tensor
    intermediate_logits: tuple[torch.Tensor, ...]


class ArithmeticComposerModel(nn.Module):
    """Apply one shared composer once or twice under a fixed oracle order."""

    def __init__(self, spec: LadderModelSpec) -> None:
        super().__init__()
        spec.validate()
        hidden = spec.hidden_dim
        self.spec = spec
        self.literal_embedding = nn.Embedding(7, hidden)
        self.operator_embedding = nn.Embedding(2, hidden)
        self.query_embedding = nn.Embedding(2, hidden)
        self.composer = nn.Sequential(
            nn.Linear(hidden * 3, spec.feedforward_dim),
            nn.GELU(),
            nn.Linear(spec.feedforward_dim, hidden),
            nn.LayerNorm(hidden),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden * 2, spec.feedforward_dim),
            nn.GELU(),
            nn.Linear(spec.feedforward_dim, 7),
        )

    def _compose(
        self, left: torch.Tensor, right: torch.Tensor, operator: torch.Tensor
    ) -> torch.Tensor:
        return self.composer(torch.cat((left, right, operator), dim=-1))

    def _logits(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.readout(torch.cat((state, query), dim=-1))

    def forward(
        self,
        model_input: LadderModelInput,
        *,
        merge_query_ids: torch.Tensor | None = None,
        teacher_intermediate_labels: torch.Tensor | None = None,
    ) -> LadderModelOutput:
        if not isinstance(model_input, LadderModelInput):
            raise TypeError("R5 model accepts LadderModelInput only")
        values = model_input.values
        operators = model_input.operators
        query_ids = model_input.query_ids
        if values.ndim != 2 or operators.ndim != 2 or query_ids.ndim != 1:
            raise ValueError("R5 model input dimensions are invalid")
        if values.shape[0] != operators.shape[0] or values.shape[0] != query_ids.shape[0]:
            raise ValueError("R5 model input row counts differ")
        if values.shape[1] not in {2, 3} or operators.shape[1] != values.shape[1] - 1:
            raise ValueError("R5 model supports two or three literals only")
        literal_states = self.literal_embedding(values)
        operator_states = self.operator_embedding(operators)
        query_states = self.query_embedding(query_ids)
        if values.shape[1] == 2:
            if bool(torch.any(query_ids != ADD_FIRST).detach().cpu().item()):
                raise ValueError("R5 binary rung requires a constant query")
            root = self._compose(
                literal_states[:, 0], literal_states[:, 1], operator_states[:, 0]
            )
            return LadderModelOutput(self._logits(root, query_states), ())
        structure_queries = query_ids if merge_query_ids is None else merge_query_ids
        if structure_queries.shape != query_ids.shape:
            raise ValueError("R5.1 merge-query intervention shape mismatch")
        expected_left = torch.full_like(operators[:, 0], SUB_OP)
        expected_right = torch.full_like(operators[:, 1], ADD_OP)
        if bool(
            torch.any(operators[:, 0] != expected_left).detach().cpu().item()
            or torch.any(operators[:, 1] != expected_right).detach().cpu().item()
        ):
            raise ValueError("R5 three-literal model requires the '-+' operator pattern")
        if teacher_intermediate_labels is not None and teacher_intermediate_labels.shape != query_ids.shape:
            raise ValueError("R5.1 teacher intermediate shape mismatch")
        first_parts: list[torch.Tensor] = []
        root_parts: list[torch.Tensor] = []
        structure_queries_cpu = structure_queries.detach().cpu().tolist()
        spans: list[tuple[int, int, int]] = []
        start = 0
        while start < len(structure_queries_cpu):
            structure_query = int(structure_queries_cpu[start])
            stop = start + 1
            while stop < len(structure_queries_cpu) and structure_queries_cpu[stop] == structure_query:
                stop += 1
            spans.append((start, stop, structure_query))
            start = stop
        if any(query not in {ADD_FIRST, SUB_FIRST} for _, _, query in spans):
            raise ValueError("R5.1 merge-query intervention contains an invalid query")
        for start, stop, structure_query in spans:
            selected_literals = literal_states[start:stop]
            selected_operators = operator_states[start:stop]
            if structure_query == ADD_FIRST:
                first = self._compose(
                    selected_literals[:, 1],
                    selected_literals[:, 2],
                    selected_operators[:, 1],
                )
                first_for_root = (
                    self.literal_embedding(
                        teacher_intermediate_labels[start:stop]
                    )
                    if teacher_intermediate_labels is not None
                    else first
                )
                root = self._compose(
                    selected_literals[:, 0], first_for_root, selected_operators[:, 0]
                )
            else:
                first = self._compose(
                    selected_literals[:, 0],
                    selected_literals[:, 1],
                    selected_operators[:, 0],
                )
                first_for_root = (
                    self.literal_embedding(
                        teacher_intermediate_labels[start:stop]
                    )
                    if teacher_intermediate_labels is not None
                    else first
                )
                root = self._compose(
                    first_for_root, selected_literals[:, 2], selected_operators[:, 1]
                )
            first_parts.append(first)
            root_parts.append(root)
        first = torch.cat(first_parts)
        root = torch.cat(root_parts)
        return LadderModelOutput(
            root_logits=self._logits(root, query_states),
            intermediate_logits=(self._logits(first, query_states),),
        )


def model_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def build_stage2_bridge(model: ArithmeticComposerModel) -> Stage2MergeClassifier:
    """Map a pure arithmetic state into the existing B compose/readout path."""

    spec = Stage2ModelSpec(
        vocab_size=64,
        hidden_dim=model.spec.hidden_dim,
        heads=4,
        layers=1,
        feedforward_dim=model.spec.feedforward_dim,
        dropout=0.0,
        temperature=1.0,
    )
    bridge = Stage2MergeClassifier(spec, allow_stop=False)
    with torch.no_grad():
        bridge.token_embedding.weight.zero_()
        bridge.token_embedding.weight[8:15].copy_(model.literal_embedding.weight.detach().cpu())
        bridge.token_embedding.weight[ADD].copy_(model.operator_embedding.weight[ADD_OP].detach().cpu())
        bridge.token_embedding.weight[SUB].copy_(model.operator_embedding.weight[SUB_OP].detach().cpu())
        for parameter in bridge.position_projection.parameters():
            parameter.zero_()
    bridge.query_embedding.load_state_dict(_cpu_state(model.query_embedding))
    bridge.composer.load_state_dict(_cpu_state(model.composer))
    bridge.classifier.load_state_dict(_cpu_state(model.readout))
    return bridge


def bridge_root_logits(
    model: ArithmeticComposerModel,
    batch: LadderGeneratedSplit,
    *,
    merge_query_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    ordinary, structure = to_stage2_oracle_batch(
        batch,
        merge_query_ids=merge_query_ids,
    )
    bridge = build_stage2_bridge(model).to(device)
    output = bridge(
        ordinary.to(device),
        policy="oracle",
        oracle_structure=structure,
        forced_compute_mode="selected_only",
    )
    return output.logits
