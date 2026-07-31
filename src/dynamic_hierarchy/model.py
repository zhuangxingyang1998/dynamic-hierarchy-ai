"""A conventional small Transformer baseline with continuous position inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig
from .data import (
    LeafSourceReference,
    MergeSourceReference,
    StructureOnlyBatch,
)


class SmallTransformerBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        config: ModelConfig,
        output_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.position_projection = nn.Sequential(nn.Linear(3, config.embedding_dim), nn.Tanh(), nn.Linear(config.embedding_dim, config.embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(config.embedding_dim, output_classes or vocab_size)

    def forward(self, token_ids: torch.Tensor, position_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or position_features.shape != (*token_ids.shape, 3):
            raise ValueError("expected token_ids [batch, sequence] and position_features [batch, sequence, 3]")
        hidden = self.token_embedding(token_ids) + self.position_projection(position_features)
        encoded = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        last_index = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
        final_state = encoded[torch.arange(token_ids.shape[0], device=token_ids.device), last_index]
        return self.classifier(final_state)


@dataclass(frozen=True)
class StructureDiagnostics:
    logits: torch.Tensor
    node_counts: tuple[int, ...]
    maximum_tree_depths: tuple[int, ...]
    combined_nodes: tuple[int, ...]
    compose_module_calls: int
    original_node_references: tuple[tuple[int, ...], ...]


class TrueStructureDiagnosticD(nn.Module):
    """Privileged-structure diagnostic that composes generator-provided trees.

    The tree supplies only structure and source-token references. Labels and
    intermediate arithmetic values never enter this model.
    """

    def __init__(
        self,
        vocab_size: int,
        config: ModelConfig,
        output_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.position_projection = nn.Sequential(
            nn.Linear(3, config.embedding_dim),
            nn.Tanh(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.context_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )
        self.compose = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )
        self.classifier = nn.Linear(config.embedding_dim, output_classes or vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        position_features: torch.Tensor,
        attention_mask: torch.Tensor,
        structure: StructureOnlyBatch,
    ) -> StructureDiagnostics:
        if not isinstance(structure, StructureOnlyBatch):
            raise TypeError("D requires StructureOnlyBatch; complete truth objects are forbidden")
        if len(structure.samples) != token_ids.shape[0]:
            raise ValueError("structure sample count must match batch size")
        hidden = self.token_embedding(token_ids) + self.position_projection(position_features)
        encoded = self.context_encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        node_states: list[dict[int, torch.Tensor]] = [dict() for _ in structure.samples]
        merge_nodes: list[list[MergeSourceReference]] = []
        references: list[tuple[int, ...]] = []
        node_counts: list[int] = []
        maximum_depths: list[int] = []
        for row_index, sample_structure in enumerate(structure.samples):
            row_merges: list[MergeSourceReference] = []
            leaf_references: list[int] = []
            for node in sample_structure.nodes:
                if isinstance(node, LeafSourceReference):
                    node_states[row_index][node.node_id] = encoded[row_index, node.source_index]
                    leaf_references.append(node.source_index)
                elif isinstance(node, MergeSourceReference):
                    row_merges.append(node)
                else:
                    raise TypeError(f"unsupported structure node type: {type(node).__name__}")
            merge_nodes.append(row_merges)
            references.append(tuple(leaf_references))
            node_counts.append(len(sample_structure.nodes))
            depth_by_node: dict[int, int] = {}
            for node in sample_structure.nodes:
                if isinstance(node, LeafSourceReference):
                    depth_by_node[node.node_id] = 0
                else:
                    depth_by_node[node.node_id] = 1 + max(
                        depth_by_node[node.left],
                        depth_by_node[node.right],
                    )
            maximum_depths.append(depth_by_node[sample_structure.root_id])
        merge_counts = {len(nodes) for nodes in merge_nodes}
        if len(merge_counts) != 1:
            raise ValueError("all samples in a structured batch must have the same recursive depth")
        compose_calls = 0
        for merge_index in range(next(iter(merge_counts))):
            inputs = []
            for row_index, row_merges in enumerate(merge_nodes):
                node = row_merges[merge_index]
                left = node_states[row_index][node.left]
                right = node_states[row_index][node.right]
                operator = encoded[row_index, node.operator_source_index]
                inputs.append(torch.cat((left, right, operator), dim=-1))
            outputs = self.compose(torch.stack(inputs))
            compose_calls += 1
            for row_index, row_merges in enumerate(merge_nodes):
                node_states[row_index][row_merges[merge_index].node_id] = outputs[row_index]
        roots = [
            node_states[row_index][sample_structure.root_id]
            for row_index, sample_structure in enumerate(structure.samples)
        ]
        combined_nodes = tuple(len(nodes) for nodes in merge_nodes)
        return StructureDiagnostics(
            logits=self.classifier(torch.stack(roots)),
            node_counts=tuple(node_counts),
            maximum_tree_depths=tuple(maximum_depths),
            combined_nodes=combined_nodes,
            compose_module_calls=compose_calls,
            original_node_references=tuple(references),
        )
