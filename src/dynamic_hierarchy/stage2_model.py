"""Hard-path learned and fixed merge models for Stage 2 R2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data import ADD, SUB
from .stage2_config import Stage2ModelSpec
from .stage2_data import Stage2OrdinaryBatch


@dataclass(frozen=True)
class Stage2TraceStep:
    step: int
    action: str
    merge_index: int | None
    left_node_id: int | None
    right_node_id: int | None
    parent_node_id: int | None
    source_start: int | None
    source_end: int | None
    operator_source_index: int | None
    legal_merge_count: int


@dataclass(frozen=True)
class Stage2Trace:
    steps: tuple[Stage2TraceStep, ...]
    stopped_early: bool
    reached_root: bool
    final_node_count: int


@dataclass(frozen=True)
class Stage2ComputeAccounting:
    recurrent_steps: int
    candidate_scores: int
    candidate_compositions: int
    selected_compositions: int
    stop_scores: int


@dataclass(frozen=True)
class Stage2MergeOutput:
    logits: torch.Tensor
    traces: tuple[Stage2Trace, ...]
    compute: Stage2ComputeAccounting


@dataclass
class _RuntimeNode:
    node_id: int
    state: torch.Tensor
    source_start: int
    source_end: int


def straight_through_hard_mask(probabilities: torch.Tensor, hard_index: int) -> torch.Tensor:
    """Hard forward mask with soft gradients, without DirectML scatter/one_hot."""

    if probabilities.ndim != 1:
        raise ValueError("straight-through probabilities must be one-dimensional")
    if not 0 <= hard_index < probabilities.shape[0]:
        raise ValueError("hard_index is outside the action range")
    positions = torch.arange(probabilities.shape[0], device=probabilities.device)
    hard = (positions == hard_index).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach()


def straight_through_select(
    candidates: torch.Tensor,
    probabilities: torch.Tensor,
    hard_index: int,
) -> torch.Tensor:
    if candidates.ndim < 2 or candidates.shape[0] != probabilities.shape[0]:
        raise ValueError("candidates must start with an action dimension matching probabilities")
    mask = straight_through_hard_mask(probabilities, hard_index)
    mask_shape = (mask.shape[0],) + (1,) * (candidates.ndim - 1)
    return (mask.reshape(mask_shape) * candidates).sum(dim=0)


class Stage2MergeClassifier(nn.Module):
    """Adjacent merge classifier whose answer path contains hard-selected states only."""

    LEGAL_POLICIES = {"learned", "stop", "left", "right", "add", "sub"}
    LEGAL_QUERY_MODES = {"query", "blind", "sham"}

    def __init__(self, spec: Stage2ModelSpec) -> None:
        super().__init__()
        spec.validate()
        hidden = spec.hidden_dim
        self.spec = spec
        self.token_embedding = nn.Embedding(spec.vocab_size, hidden, padding_idx=0)
        self.position_projection = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden)
        )
        self.query_embedding = nn.Embedding(2, hidden)
        self.blind_router_query = nn.Parameter(torch.zeros(hidden))
        self.composer = nn.Sequential(
            nn.Linear(hidden * 3, spec.feedforward_dim),
            nn.GELU(),
            nn.Linear(spec.feedforward_dim, hidden),
            nn.LayerNorm(hidden),
        )
        self.router = nn.Sequential(
            nn.Linear(hidden * 4 + 1, spec.feedforward_dim),
            nn.GELU(),
            nn.Linear(spec.feedforward_dim, 1),
        )
        self.stop_router = nn.Sequential(
            nn.Linear(hidden * 2 + 1, spec.feedforward_dim),
            nn.GELU(),
            nn.Linear(spec.feedforward_dim, 1),
        )
        self.terminal_attention = nn.Linear(hidden * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, spec.feedforward_dim),
            nn.GELU(),
            nn.Linear(spec.feedforward_dim, 7),
        )

    def _router_query(self, query_id: torch.Tensor, mode: str) -> torch.Tensor:
        if mode not in self.LEGAL_QUERY_MODES:
            raise ValueError(f"unsupported router query mode: {mode}")
        if mode == "blind":
            return self.blind_router_query
        routed_id = 1 - query_id if mode == "sham" else query_id
        return self.query_embedding(routed_id)

    def _terminal_logits(
        self,
        active: list[_RuntimeNode],
        answer_query: torch.Tensor,
    ) -> torch.Tensor:
        states = torch.stack([node.state for node in active])
        repeated_query = answer_query.expand(states.shape[0], -1)
        attention = torch.softmax(
            self.terminal_attention(torch.cat((states, repeated_query), dim=-1)).squeeze(-1),
            dim=0,
        )
        summary = (attention[:, None] * states).sum(dim=0)
        return self.classifier(torch.cat((summary, answer_query), dim=-1))

    @staticmethod
    def _fixed_action(policy: str, operator_tokens: list[int]) -> int:
        if policy == "stop":
            return len(operator_tokens)
        if policy == "left":
            return 0
        if policy == "right":
            return len(operator_tokens) - 1
        preferred = ADD if policy == "add" else SUB
        return next(
            (index for index, token in enumerate(operator_tokens) if token == preferred),
            0,
        )

    def forward(
        self,
        batch: Stage2OrdinaryBatch,
        *,
        policy: str = "learned",
        router_query_mode: str = "query",
        merge_budget: int | None = None,
    ) -> Stage2MergeOutput:
        if not isinstance(batch, Stage2OrdinaryBatch):
            raise TypeError("Stage 2 merge models require Stage2OrdinaryBatch")
        if policy not in self.LEGAL_POLICIES:
            raise ValueError(f"unsupported Stage 2 merge policy: {policy}")
        if router_query_mode not in self.LEGAL_QUERY_MODES:
            raise ValueError(f"unsupported router query mode: {router_query_mode}")
        if batch.token_ids.shape[0] != batch.query_ids.shape[0]:
            raise ValueError("query count must match Stage 2 batch rows")
        token_states = self.token_embedding(batch.token_ids) + self.position_projection(
            batch.position_features
        )
        row_logits: list[torch.Tensor] = []
        traces: list[Stage2Trace] = []
        recurrent_steps = 0
        candidate_scores = 0
        candidate_compositions = 0
        selected_compositions = 0
        stop_scores = 0
        for row_index in range(batch.token_ids.shape[0]):
            query_id = batch.query_ids[row_index]
            answer_query = self.query_embedding(query_id)
            router_query = self._router_query(query_id, router_query_mode)
            literal_positions = batch.literal_source_indices[row_index].tolist()
            operator_positions = batch.operator_source_indices[row_index].tolist()
            active = [
                _RuntimeNode(
                    node_id=index,
                    state=token_states[row_index, source_index],
                    source_start=source_index,
                    source_end=source_index,
                )
                for index, source_index in enumerate(literal_positions)
            ]
            active_operator_positions = list(operator_positions)
            active_operator_tokens = [
                int(batch.token_ids[row_index, source_index].detach().cpu().item())
                for source_index in active_operator_positions
            ]
            next_node_id = len(active)
            row_trace: list[Stage2TraceStep] = []
            stopped_early = False
            max_merges = len(active) - 1
            row_budget = max_merges if merge_budget is None else merge_budget
            if not 0 <= row_budget <= max_merges:
                raise ValueError("merge_budget must be between zero and leaf_count - 1")
            for step in range(row_budget):
                if len(active) == 1:
                    break
                step_fraction = token_states.new_tensor([step / max(1, max_merges)])
                pair_inputs: list[torch.Tensor] = []
                composition_inputs: list[torch.Tensor] = []
                for pair_index, operator_source_index in enumerate(active_operator_positions):
                    left = active[pair_index].state
                    right = active[pair_index + 1].state
                    operator = token_states[row_index, operator_source_index]
                    composition_inputs.append(torch.cat((left, right, operator), dim=-1))
                    pair_inputs.append(
                        torch.cat((left, right, operator, router_query, step_fraction), dim=-1)
                    )
                compositions = self.composer(torch.stack(composition_inputs))
                pair_logits = self.router(torch.stack(pair_inputs)).squeeze(-1)
                active_summary = torch.stack([node.state for node in active]).mean(dim=0)
                stop_logit = self.stop_router(
                    torch.cat((active_summary, router_query, step_fraction), dim=-1)
                ).reshape(1)
                action_logits = torch.cat((pair_logits, stop_logit), dim=0)
                probabilities = torch.softmax(
                    action_logits / self.spec.temperature,
                    dim=0,
                )
                candidate_scores += action_logits.shape[0]
                candidate_compositions += compositions.shape[0]
                stop_scores += 1
                recurrent_steps += 1
                hard_index = (
                    int(action_logits.detach().argmax().cpu().item())
                    if policy == "learned"
                    else self._fixed_action(policy, active_operator_tokens)
                )
                if hard_index == len(active_operator_positions):
                    row_trace.append(
                        Stage2TraceStep(
                            step=step,
                            action="STOP",
                            merge_index=None,
                            left_node_id=None,
                            right_node_id=None,
                            parent_node_id=None,
                            source_start=None,
                            source_end=None,
                            operator_source_index=None,
                            legal_merge_count=len(active_operator_positions),
                        )
                    )
                    stopped_early = len(active) > 1
                    break
                selected = straight_through_select(compositions, probabilities[:-1], hard_index)
                left_node = active[hard_index]
                right_node = active[hard_index + 1]
                operator_source_index = active_operator_positions.pop(hard_index)
                active_operator_tokens.pop(hard_index)
                parent = _RuntimeNode(
                    node_id=next_node_id,
                    state=selected,
                    source_start=left_node.source_start,
                    source_end=right_node.source_end,
                )
                active[hard_index : hard_index + 2] = [parent]
                row_trace.append(
                    Stage2TraceStep(
                        step=step,
                        action="MERGE",
                        merge_index=hard_index,
                        left_node_id=left_node.node_id,
                        right_node_id=right_node.node_id,
                        parent_node_id=next_node_id,
                        source_start=parent.source_start,
                        source_end=parent.source_end,
                        operator_source_index=operator_source_index,
                        legal_merge_count=len(active_operator_positions) + 1,
                    )
                )
                next_node_id += 1
                selected_compositions += 1
            if len(active) > 1 and not stopped_early and len(row_trace) >= row_budget:
                raise RuntimeError("Stage 2 merge budget exhausted with multiple active nodes")
            row_logits.append(self._terminal_logits(active, answer_query))
            traces.append(
                Stage2Trace(
                    steps=tuple(row_trace),
                    stopped_early=stopped_early,
                    reached_root=len(active) == 1,
                    final_node_count=len(active),
                )
            )
        return Stage2MergeOutput(
            logits=torch.stack(row_logits),
            traces=tuple(traces),
            compute=Stage2ComputeAccounting(
                recurrent_steps=recurrent_steps,
                candidate_scores=candidate_scores,
                candidate_compositions=candidate_compositions,
                selected_compositions=selected_compositions,
                stop_scores=stop_scores,
            ),
        )


@dataclass(frozen=True)
class Stage2RecurrentOutput:
    logits: torch.Tensor
    recurrent_steps: int
    stop_scores: int
    early_stops: int


class Stage2RecurrentFlatBaseline(nn.Module):
    """Query-aware shared recurrence with hard halting and no merge structure."""

    def __init__(self, spec: Stage2ModelSpec) -> None:
        super().__init__()
        spec.validate()
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.hidden_dim, padding_idx=0)
        self.position_projection = nn.Sequential(
            nn.Linear(3, spec.hidden_dim), nn.Tanh(), nn.Linear(spec.hidden_dim, spec.hidden_dim)
        )
        self.shared_block = nn.TransformerEncoderLayer(
            d_model=spec.hidden_dim,
            nhead=spec.heads,
            dim_feedforward=spec.feedforward_dim,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.halting_router = nn.Sequential(
            nn.Linear(spec.hidden_dim * 2 + 1, spec.feedforward_dim),
            nn.GELU(),
            nn.Linear(spec.feedforward_dim, 2),
        )
        self.classifier = nn.Linear(spec.hidden_dim, 7)
        self.temperature = spec.temperature

    def forward(self, batch: Stage2OrdinaryBatch) -> Stage2RecurrentOutput:
        if not isinstance(batch, Stage2OrdinaryBatch):
            raise TypeError("A-recur requires Stage2OrdinaryBatch")
        encoded = self.token_embedding(batch.token_ids) + self.position_projection(
            batch.position_features
        )
        row_logits: list[torch.Tensor] = []
        recurrent_steps = 0
        stop_scores = 0
        early_stops = 0
        maximum_steps = max(1, batch.literal_source_indices.shape[1] - 1)
        for row_index in range(encoded.shape[0]):
            hidden = encoded[row_index : row_index + 1]
            mask = batch.attention_mask[row_index : row_index + 1]
            last_index = int(mask.long().sum().sub(1).detach().cpu().item())
            stopped = False
            for step in range(maximum_steps):
                candidate = self.shared_block(hidden, src_key_padding_mask=~mask.bool())
                step_fraction = hidden.new_tensor([step / maximum_steps])
                router_input = torch.cat(
                    (
                        candidate[0, last_index],
                        hidden[0, last_index],
                        step_fraction,
                    ),
                    dim=0,
                )
                action_logits = self.halting_router(router_input)
                probabilities = torch.softmax(action_logits / self.temperature, dim=0)
                hard_index = int(action_logits.detach().argmax().cpu().item())
                hidden = straight_through_select(
                    torch.stack((candidate[0], hidden[0])),
                    probabilities,
                    hard_index,
                ).unsqueeze(0)
                recurrent_steps += 1
                stop_scores += 1
                if hard_index == 1:
                    early_stops += int(step + 1 < maximum_steps)
                    stopped = True
                    break
            if not stopped and maximum_steps <= 0:
                raise RuntimeError("A-recur exhausted an invalid zero recurrence budget")
            row_logits.append(self.classifier(hidden[0, last_index]))
        return Stage2RecurrentOutput(
            logits=torch.stack(row_logits),
            recurrent_steps=recurrent_steps,
            stop_scores=stop_scores,
            early_stops=early_stops,
        )
