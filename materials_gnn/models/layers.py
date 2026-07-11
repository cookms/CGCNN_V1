"""Reusable neural network layers for crystal graph models."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from materials_gnn.models.implicit_bias import ImplicitBiasActivation


_CONV_IB_TARGETS = frozenset({"edge", "message", "node"})


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


def build_mlp_with_activation_factory(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    activation_factory: Callable[[int], nn.Module],
    dropout: float = 0.0,
) -> nn.Sequential:
    """Construct an MLP whose hidden activation can depend on hidden width."""

    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    last_hidden_idx = len(dims) - 2
    for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(nn.Linear(in_dim, out_dim))
        if idx < last_hidden_idx:
            layers.append(activation_factory(out_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _normalize_conv_ib_targets(conv_ib_targets: Sequence[str] | str) -> frozenset[str]:
    if isinstance(conv_ib_targets, str):
        raw_targets = tuple(
            part.strip().lower() for part in conv_ib_targets.split(",") if part.strip()
        )
    else:
        raw_targets = tuple(str(part).strip().lower() for part in conv_ib_targets if str(part).strip())

    if not raw_targets or raw_targets == ("none",):
        return frozenset()

    allowed = _CONV_IB_TARGETS | {"all", "none"}
    invalid = sorted(set(raw_targets) - allowed)
    if invalid:
        raise ValueError(
            "conv_ib_targets must contain only 'edge', 'message', 'node', 'all', or 'none'; "
            f"got {invalid}"
        )
    if "none" in raw_targets and len(raw_targets) > 1:
        raise ValueError("conv_ib_targets='none' cannot be combined with other targets")
    if "all" in raw_targets:
        return _CONV_IB_TARGETS
    return frozenset(raw_targets)


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
        conv_activation_type: str = "silu",
        conv_ib_lambda: float = 0.01,
        conv_ib_sigma_slope: float = 1.0,
        conv_ib_fixed_point_iters: int = 8,
        conv_ib_coupling: str = "ring",
        conv_ib_trainable_lambda: bool = False,
        conv_ib_targets: Sequence[str] | str = (),
    ) -> None:
        super().__init__()
        if conv_activation_type not in {"silu", "implicit_bias"}:
            raise ValueError(
                "conv_activation_type must be either 'silu' or 'implicit_bias'; "
                f"got {conv_activation_type!r}"
            )
        targets = _normalize_conv_ib_targets(conv_ib_targets)

        hidden = hidden_dim or max(node_dim, edge_dim)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.residual = residual
        self.conv_activation_type = conv_activation_type
        self.conv_ib_targets = tuple(sorted(targets))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def _uses_implicit_bias(target_name: str) -> bool:
            return conv_activation_type == "implicit_bias" and target_name in targets

        def _make_activation_factory(target_name: str) -> Callable[[int], nn.Module]:
            if _uses_implicit_bias(target_name):
                return lambda dim: ImplicitBiasActivation(
                    dim,
                    ib_lambda=conv_ib_lambda,
                    sigma_slope=conv_ib_sigma_slope,
                    fixed_point_iters=conv_ib_fixed_point_iters,
                    coupling=conv_ib_coupling,
                    activation="silu",
                    trainable_lambda=conv_ib_trainable_lambda,
                )
            return lambda _dim: nn.SiLU()

        def _build_conv_mlp(
            target_name: str,
            *,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
        ) -> nn.Sequential:
            if _uses_implicit_bias(target_name):
                return build_mlp_with_activation_factory(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=output_dim,
                    activation_factory=_make_activation_factory(target_name),
                    dropout=dropout,
                )
            return build_mlp(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout=dropout,
            )

        self.edge_mlp = _build_conv_mlp(
            "edge",
            input_dim=2 * node_dim + edge_dim,
            hidden_dims=[hidden],
            output_dim=edge_dim,
        )
        self.message_mlp = _build_conv_mlp(
            "message",
            input_dim=node_dim + edge_dim,
            hidden_dims=[hidden],
            output_dim=node_dim,
        )
        self.gate_mlp = nn.Sequential(nn.Linear(edge_dim, node_dim), nn.Sigmoid())
        self.node_mlp = _build_conv_mlp(
            "node",
            input_dim=2 * node_dim,
            hidden_dims=[hidden],
            output_dim=node_dim,
        )

        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply one message-passing step.

        Args:
            x: Node features ``[num_nodes, node_dim]``.
            edge_index: Directed edges ``[2, num_edges]``.
            edge_attr: Edge features ``[num_edges, edge_dim]``.
            edge_weight: Optional scalar weights ``[num_edges]`` or ``[num_edges, 1]``
                applied to aggregation gates. Graph builders are responsible for
                producing meaningful weights.

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
        use_edge_weight = edge_weight is not None
        if use_edge_weight:
            if not isinstance(edge_weight, Tensor):
                raise TypeError("edge_weight must be a tensor when provided")
            if edge_weight.ndim == 1:
                if edge_weight.shape[0] != num_edges:
                    raise ValueError(
                        "edge_weight length must match num_edges; "
                        f"got {edge_weight.shape[0]} and {num_edges}"
                    )
            elif edge_weight.ndim == 2:
                if edge_weight.shape != (num_edges, 1):
                    raise ValueError(
                        "edge_weight must have shape [num_edges] or [num_edges, 1]; "
                        f"got {tuple(edge_weight.shape)} for num_edges={num_edges}"
                    )
            else:
                raise ValueError(
                    "edge_weight must have shape [num_edges] or [num_edges, 1]; "
                    f"got {tuple(edge_weight.shape)}"
                )

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

        if edge_weight is None:
            effective_gates = gates
        else:
            edge_weight = edge_weight.to(device=gates.device, dtype=gates.dtype)
            if edge_weight.ndim == 1:
                edge_weight = edge_weight.unsqueeze(-1)
            effective_gates = gates * edge_weight

        weighted_messages = effective_gates * messages

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
        gate_sum_fp32.index_add_(0, dst, effective_gates.float())

        if use_edge_weight:
            aggregate = torch.zeros_like(aggregate_fp32)
            nonzero_gate_sum = gate_sum_fp32 != 0
            aggregate[nonzero_gate_sum] = (
                aggregate_fp32[nonzero_gate_sum] / gate_sum_fp32[nonzero_gate_sum]
            )
        else:
            aggregate = aggregate_fp32 / gate_sum_fp32.clamp_min(1e-4)
        aggregate = aggregate.to(dtype=x.dtype)

        node_delta = self.node_mlp(torch.cat([x, aggregate], dim=-1))
        x_new = x + self.dropout(node_delta) if self.residual else node_delta
        x_new = self.node_norm(x_new)
        return x_new, edge_new
