"""R6 fixed-interface model with explicit generated-state substitution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .stage2_ladder_config import LadderModelSpec
from .stage2_ladder_data import (
    ADD_FIRST,
    ADD_OP,
    SUB_FIRST,
    SUB_OP,
    LadderModelInput,
)
from .stage2_ladder_model import ArithmeticComposerModel


@dataclass(frozen=True)
class CongruenceForward:
    first_state: torch.Tensor
    ordinary_logits: torch.Tensor
    intervention_logits: torch.Tensor | None
    operation_counts: dict[str, int]


class StateCongruenceModel(ArithmeticComposerModel):
    """Keep R5 parameters while executing only the two R6 `+-` interfaces."""

    def __init__(self, spec: LadderModelSpec) -> None:
        super().__init__(spec)
        self._active_operation_trace: dict[str, int] | None = None
        self._active_compose_stage: str | None = None

    def _compose(
        self, left: torch.Tensor, right: torch.Tensor, operator: torch.Tensor
    ) -> torch.Tensor:
        if self._active_operation_trace is not None:
            if self._active_compose_stage not in {"first", "outer"}:
                raise RuntimeError("R6 compose call lacks a traced stage")
            self._active_operation_trace[
                f"{self._active_compose_stage}_compositions"
            ] += 1
        return super()._compose(left, right, operator)

    def _logits(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if self._active_operation_trace is not None:
            self._active_operation_trace["readouts"] += 1
        return super()._logits(state, query)

    @staticmethod
    def _validate_input(model_input: LadderModelInput) -> None:
        if not isinstance(model_input, LadderModelInput):
            raise TypeError("R6 model accepts LadderModelInput only")
        values = model_input.values
        operators = model_input.operators
        queries = model_input.query_ids
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("R6 model requires exactly three literals")
        if operators.shape != (values.shape[0], 2) or queries.shape != (values.shape[0],):
            raise ValueError("R6 model input dimensions are inconsistent")
        if bool(torch.any(operators[:, 0] != ADD_OP).detach().cpu().item()) or bool(
            torch.any(operators[:, 1] != SUB_OP).detach().cpu().item()
        ):
            raise ValueError("R6 model requires the '+-' operator pattern")
        if bool(
            torch.any((queries != ADD_FIRST) & (queries != SUB_FIRST))
            .detach()
            .cpu()
            .item()
        ):
            raise ValueError("R6 model query_id is invalid")

    def first_states(self, model_input: LadderModelInput) -> torch.Tensor:
        self._validate_input(model_input)
        literals = self.literal_embedding(model_input.values)
        operators = self.operator_embedding(model_input.operators)
        queries = model_input.query_ids
        query = int(queries[0].detach().cpu().item())
        if bool(torch.any(queries != query).detach().cpu().item()):
            raise ValueError("R6 model batches must contain one fixed query")
        prior_stage = self._active_compose_stage
        self._active_compose_stage = "first"
        try:
            if query == ADD_FIRST:
                return self._compose(
                    literals[:, 0], literals[:, 1], operators[:, 0]
                )
            return self._compose(literals[:, 1], literals[:, 2], operators[:, 1])
        finally:
            self._active_compose_stage = prior_stage

    def outer_logits(
        self, model_input: LadderModelInput, intermediate_state: torch.Tensor
    ) -> torch.Tensor:
        self._validate_input(model_input)
        if intermediate_state.shape != (
            model_input.values.shape[0],
            self.spec.hidden_dim,
        ):
            raise ValueError("R6 intermediate-state override shape is invalid")
        literals = self.literal_embedding(model_input.values)
        operators = self.operator_embedding(model_input.operators)
        queries = model_input.query_ids
        query_states = self.query_embedding(queries)
        query = int(queries[0].detach().cpu().item())
        if bool(torch.any(queries != query).detach().cpu().item()):
            raise ValueError("R6 model batches must contain one fixed query")
        prior_stage = self._active_compose_stage
        self._active_compose_stage = "outer"
        try:
            if query == ADD_FIRST:
                root = self._compose(
                    intermediate_state, literals[:, 2], operators[:, 1]
                )
            else:
                root = self._compose(
                    literals[:, 0], intermediate_state, operators[:, 0]
                )
        finally:
            self._active_compose_stage = prior_stage
        return self._logits(root, query_states)

    def forward(
        self,
        model_input: LadderModelInput,
        *,
        source_indices: torch.Tensor | None = None,
        teacher_intermediate_labels: torch.Tensor | None = None,
    ) -> CongruenceForward:
        if self._active_operation_trace is not None:
            raise RuntimeError("R6 operation trace is already active")
        self._active_operation_trace = {
            "first_compositions": 0,
            "outer_compositions": 0,
            "readouts": 0,
        }
        try:
            if teacher_intermediate_labels is not None:
                self._validate_input(model_input)
                if teacher_intermediate_labels.shape != model_input.query_ids.shape:
                    raise ValueError("R6 teacher labels have the wrong shape")
                first = self.literal_embedding(teacher_intermediate_labels)
                ordinary = self.outer_logits(model_input, first)
                intervention = None
            else:
                first = self.first_states(model_input)
                ordinary = self.outer_logits(model_input, first)
                intervention = None
                if source_indices is not None:
                    if source_indices.shape != model_input.query_ids.shape:
                        raise ValueError("R6 source indices have the wrong shape")
                    intervention = self.outer_logits(
                        model_input, first[source_indices]
                    )
            measured = dict(self._active_operation_trace)
        finally:
            self._active_operation_trace = None
            self._active_compose_stage = None
        return CongruenceForward(
            first,
            ordinary,
            intervention,
            measured,
        )
