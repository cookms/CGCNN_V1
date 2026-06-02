"""Reusable neural network layers for crystal graph models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    activation: type[nn.Module] = nn.SiLU,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Construct a small fully connected network."""

    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
        layers.append(nn.Linear(in_dim, out_dim))
        if out_dim != output_dim:
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GatedGraphConv(nn.Module):
    """Edge-gated message passing layer that updates nodes and edges.

    For an atom graph, node features represent atoms and edge features represent bonds.
    For a line graph, node features represent directed bonds and edge features represent
    bond angles. The same layer can therefore be reused in CGCNN-style and ALIGNN-like
    blocks.

    Update sketch for each directed edge u -> v:
        1. Update the edge state from source node, destination node, and old edge state.
        2. Compute a message from source node and updated edge state.
        3. Gate the message with a learned sigmoid gate from the edge state.
        4. Aggregate gated messages into destination nodes.
        5. Update each node from its old state and aggregated neighborhood message.

    The implementation uses ``index_add_`` instead of PyG/DGL scatter operations so the
    first prototype remains plain PyTorch.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        *,
        hidden_dim: int | None = None,
        residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or max(node_dim, edge_dim)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.residual = residual
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = build_mlp(
            input_dim=2 * node_dim + edge_dim,
            hidden_dims=[hidden],
            output_dim=edge_dim,
            dropout=dropout,
        )
        self.message_mlp = build_mlp(
            input_dim=node_dim + edge_dim,
            hidden_dims=[hidden],
            output_dim=node_dim,
            dropout=dropout,
        )
        self.gate_mlp = nn.Sequential(nn.Linear(edge_dim, node_dim), nn.Sigmoid())
        self.node_mlp = build_mlp(
            input_dim=2 * node_dim,
            hidden_dims=[hidden],
            output_dim=node_dim,
            dropout=dropout,
        )

        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one message-passing step.

        Args:
            x: Node features ``[num_nodes, node_dim]``.
            edge_index: Directed edges ``[2, num_edges]``.
            edge_attr: Edge features ``[num_edges, edge_dim]``.

        Returns:
            Updated ``(x, edge_attr)``.
        """

        if x.ndim != 2:
            raise ValueError("x must have shape [num_nodes, node_dim]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_attr.ndim != 2:
            raise ValueError("edge_attr must have shape [num_edges, edge_dim]")
        if edge_index.shape[1] != edge_attr.shape[0]:
            raise ValueError("edge_index and edge_attr disagree on num_edges")

        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            # A graph can be edge-less with a tiny cutoff or malformed input. We still pass
            # nodes through a learnable update using a zero aggregate to keep shapes valid.
            aggregate = torch.zeros_like(x)
            node_delta = self.node_mlp(torch.cat([x, aggregate], dim=-1))
            x_new = x + self.dropout(node_delta) if self.residual else node_delta
            return self.node_norm(x_new), edge_attr

        src, dst = edge_index[0].long(), edge_index[1].long()

        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_delta = self.edge_mlp(edge_input)
        edge_new = edge_attr + self.dropout(edge_delta) if self.residual else edge_delta
        edge_new = self.edge_norm(edge_new)

        message_input = torch.cat([x[src], edge_new], dim=-1)
        messages = self.message_mlp(message_input)
        gates = self.gate_mlp(edge_new)

        weighted_messages = gates * messages

        # Accumulate in float32 for numerical stability under AMP.
        # This is especially important for ALIGNN-like line graphs, where many
        # angle messages may accumulate into the same bond node.
        aggregate_fp32 = torch.zeros(
            (num_nodes, self.node_dim),
            device=weighted_messages.device,
            dtype=torch.float32,
        )
        gate_sum_fp32 = torch.zeros(
            (num_nodes, self.node_dim),
            device=gates.device,
            dtype=torch.float32,
        )

        aggregate_fp32.index_add_(0, dst, weighted_messages.float())
        gate_sum_fp32.index_add_(0, dst, gates.float())

        aggregate = aggregate_fp32 / gate_sum_fp32.clamp_min(1e-4)
        aggregate = aggregate.to(dtype=x.dtype)

        node_delta = self.node_mlp(torch.cat([x, aggregate], dim=-1))
        x_new = x + self.dropout(node_delta) if self.residual else node_delta
        x_new = self.node_norm(x_new)
        return x_new, edge_new
