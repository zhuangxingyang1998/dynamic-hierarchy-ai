"""Candidate-only hierarchy interface; it does not perform merges in Stage 0."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HierarchyProposal:
    phase: torch.Tensor
    operation_scores: torch.Tensor
    operations: tuple[str, str] = ("MERGE", "STOP")


class CandidateHierarchyController(nn.Module):
    """Produces bounded proposal signals while preserving every original token."""

    def __init__(self, hidden_dim: int, position_dim: int = 3) -> None:
        super().__init__()
        self.phase = nn.Sequential(nn.Linear(hidden_dim + position_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.operation = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2))

    def forward(self, hidden: torch.Tensor, position_features: torch.Tensor) -> HierarchyProposal:
        if hidden.shape[:2] != position_features.shape[:2]:
            raise ValueError("hidden states and position features must share batch and sequence dimensions")
        phase = torch.sigmoid(self.phase(torch.cat((hidden, position_features), dim=-1))).squeeze(-1)
        operation_scores = self.operation(torch.cat((hidden, phase.unsqueeze(-1)), dim=-1))
        return HierarchyProposal(phase=phase, operation_scores=operation_scores)
