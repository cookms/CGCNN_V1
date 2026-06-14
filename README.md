# materials-gnn

`materials-gnn` is a small research-oriented Python package for materials property prediction with crystal graph neural networks. It starts with a clear CGCNN-style atom/bond graph model and an ALIGNN-inspired atom/bond/angle model, while keeping the code modular enough to swap graph builders, featurizers, message-passing layers, pooling methods, losses, and metrics.

This is a first prototype, not a performance-optimized benchmark implementation.

## What the prototype supports

- Convert `pymatgen.Structure` objects or CIF files into directed periodic neighbor graphs.
- Swap graph-construction strategies: cutoff, k-nearest-neighbor, Voronoi, adaptive local shells, or strain-consensus robust cutoff graphs.
- Encode bond distances with configurable Gaussian, Bessel/sine, or Fourier basis functions.
- Optionally add normalized elemental descriptors such as electronegativity, group, period, covalent radius, valence electrons, electron affinity, polarizability, magnetic moment, and ionization energy when pymatgen provides the data.
- Build an ALIGNN-style line graph where directed bonds become nodes and line-graph edges encode bond angles with configurable bases and optional memory caps.
- Train raw PyTorch CGCNN-style and ALIGNN-like models for scalar regression.
- Optionally use atom-graph `edge_weight` scalars from weighted graph builders during message aggregation.
- Use `device=auto`, CUDA batch transfer, optional CUDA mixed precision, and DataLoader pinned memory.
- Reuse expensive CIF-to-graph preprocessing with RAM, persistent disk graph caching, or an explicit cache-precompute CLI.
- Use CSV datasets with columns such as `material_id,cif_path,target`.
- Normalize targets using train-set statistics.
- Report MAE, RMSE, and R2.
- Export predictions and create parity or residual plots.

## Installation

```bash
pip install -e .
```

Minimal dependencies:

```bash
pip install torch pymatgen numpy pandas scikit-learn
```

For plots and tests:

```bash
pip install matplotlib pytest
```

## Expected CSV format

```text
material_id,cif_path,target
mp-149,data/cifs/Si.cif,-5.42
mp-13,data/cifs/Fe.cif,-8.31
```

Relative `cif_path` values are resolved relative to the CSV file location unless a different root is supplied in the dataset class.

## Graph-construction strategies

The graph topology is now controlled by `materials_gnn.featurization.neighbor_strategies`.

```python
from materials_gnn.featurization import structure_to_bond_graph

graph = structure_to_bond_graph(
    structure,
    cutoff=5.0,
    neighbor_strategy="cutoff",
    num_rbf=64,
)
```

Supported strategies:

| Strategy | Use when | Main options |
| --- | --- | --- |
| `cutoff` | You want the standard CGCNN-style graph with every periodic neighbor inside a radius. | `cutoff` |
| `knn` | You want controlled graph size with a fixed number of outgoing neighbors per atom. | `neighbor_kwargs={"k": 12, "max_radius": 8.0}` |
| `voronoi` | You want coordination based on periodic Voronoi faces rather than a global radius. | `neighbor_kwargs={"cutoff": 10.0, "tol": 0.0}` |
| `adaptive_shell` | You want an experimental local-shell graph that adapts to each atom's distance gaps. | `neighbor_kwargs={"max_radius": 8.0, "min_neighbors": 4, "max_neighbors": 24}` |
| `strain_consensus` | You want a speculative robust graph that keeps cutoff edges stable under small virtual lattice strains. | `neighbor_kwargs={"cutoff": 5.0, "strain_epsilon": 0.02, "min_survival_fraction": 0.5}` |

Examples:

```python
# Fixed radius baseline.
graph = structure_to_bond_graph(structure, cutoff=5.0, neighbor_strategy="cutoff")

# KNN graph. Set rbf_cutoff to a fixed dataset-level scale if max_radius exceeds cutoff.
graph = structure_to_bond_graph(
    structure,
    cutoff=5.0,
    rbf_cutoff=8.0,
    neighbor_strategy="knn",
    neighbor_kwargs={"k": 12, "max_radius": 8.0},
)

# Voronoi coordination graph with optional edge weights from pymatgen.
graph = structure_to_bond_graph(
    structure,
    cutoff=5.0,
    rbf_cutoff=10.0,
    neighbor_strategy="voronoi",
    neighbor_kwargs={"cutoff": 10.0},
)

# Experimental adaptive local shell graph.
graph = structure_to_bond_graph(
    structure,
    cutoff=5.0,
    rbf_cutoff=8.0,
    neighbor_strategy="adaptive_shell",
    neighbor_kwargs={"max_radius": 8.0, "min_neighbors": 4, "max_neighbors": 24},
)

# Speculative robust graph: keep edges that survive small virtual lattice strains.
graph = structure_to_bond_graph(
    structure,
    cutoff=5.0,
    neighbor_strategy="strain_consensus",
    neighbor_kwargs={"strain_epsilon": 0.02, "min_survival_fraction": 0.5},
)
```

Custom strategies can implement `build(structure) -> NeighborList` and be passed directly as `neighbor_strategy=my_strategy`. This is the intended path for new graph construction research.

### Optional atom-graph edge weights

Some graph builders, including Voronoi and strain-consensus strategies, may attach an
optional `edge_weight` tensor to the atom/bond graph. By default the models ignore this
field, preserving the original unweighted aggregation behavior. To use those weights in
atom-graph message aggregation, opt in with `use_edge_weight=True`:

```python
from materials_gnn.models import CGCNNModel

model = CGCNNModel(
    edge_input_dim=64,
    hidden_dim=128,
    use_edge_weight=True,
)
```

The same flag is available on `ALIGNNLikeModel`, where it affects only atom/bond graph
updates. Line-graph edge weighting is not implemented. If `use_edge_weight=True` but a
graph has no `edge_weight`, the model falls back to the original unweighted aggregation.
Graph builders are responsible for producing meaningful weights; the model does not
normalize or clamp user-provided values.

## Atom, bond, and angle featurization

The package now separates graph topology from feature design.

### Elemental descriptors

Atomic number embeddings are still the default. To add periodic-table descriptors, pass a comma-separated feature list or `default`:

```python
from materials_gnn.models import CGCNNModel

model = CGCNNModel(
    edge_input_dim=64,
    hidden_dim=128,
    atom_feature_names="electronegativity,group,period,covalent_radius,valence_electrons",
)
```

The default descriptor set is:

```text
electronegativity, group, period, covalent_radius, valence_electrons,
electron_affinity, polarizability, magnetic_moment, ionization_energy
```

Descriptors are looked up with `ElementalDescriptorFeaturizer`, normalized over elements 1--118, and missing values are filled with zero after normalization. Missing-value indicator columns are appended by default, because not every property is available or well-defined for every element. Magnetic moment is especially environment-dependent, so the featurizer only uses a tabulated value when available and otherwise marks it missing.

You can also precompute descriptors into the graph:

```python
graph = structure_to_bond_graph(
    structure,
    cutoff=5.0,
    atom_feature_names="default",
)
```

The resulting graph includes:

```python
{
    "atom_attr": atom_descriptor_features,  # [num_atoms, num_atom_features]
}
```

### Configurable scalar bases

Distance and angle bases can be changed during preprocessing:

```python
graph = structure_to_bond_graph(
    structure,
    cutoff=5.0,
    num_rbf=64,
    distance_basis_type="bessel",  # gaussian | bessel | fourier
)

graph = add_line_graph(
    graph,
    num_angle_rbf=32,
    angle_basis_type="fourier",    # gaussian | bessel | fourier
)
```

For learnable scalar bases, let the model expand raw `distance` and `angle` tensors:

```python
model = ALIGNNLikeModel(
    edge_input_dim=64,
    angle_input_dim=32,
    distance_basis_type="learnable_gaussian",
    distance_basis_cutoff=5.0,
    angle_basis_type="learnable_gaussian",
)
```

This keeps trainable basis centers and widths inside the model checkpoint instead of freezing them into graph preprocessing.

### Equivariant-feature hooks

The graph now carries raw geometric tensors needed by future equivariant layers:

```python
{
    "pos": cartesian_positions,
    "edge_vec": periodic_displacement_vectors,
    "edge_unit_vec": normalized_edge_directions,
    "distance": bond_lengths,
    "angle": line_graph_bond_angles,
    "cosine": line_graph_angle_cosines,
}
```

The current CGCNN/ALIGNN-like layers remain invariant scalar message-passing layers, but these fields make it straightforward to add vector/tensor features later.


## CUDA and graph caching

The right stage to add CUDA is after the graph schema, collation, and model forward passes are stable. That is now true for this prototype: graph construction remains CPU-bound because it depends on pymatgen neighbor searches, while tensor batches and model parameters can be moved to CUDA during training.

Use automatic device selection:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --device auto
```

Use CUDA automatic mixed precision when training on a supported NVIDIA GPU:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --device cuda \
  --amp \
  --num-workers 4
```

The examples use pinned memory automatically when the resolved device is CUDA. The trainer keeps `device="auto"` as the default, so CPU-only machines still work.

Graph construction can be cached in two ways:

```bash
# RAM cache: helpful within one training process.
python examples/train_cgcnn.py \
  --csv data/id_prop.csv \
  --target target \
  --cache-graphs

# Persistent disk cache: helpful across runs and hyperparameter sweeps.
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --graph-cache-dir .cache/materials_gnn_graphs
```

The persistent cache key includes only graph-defining inputs: the CIF file fingerprint, graph strategy, cutoff, basis settings, atom features, and line-graph settings. Labels, target column names, material IDs, train/validation splits, model hidden dimensions, learning rates, and other training-only settings are intentionally excluded. Changing a graph-defining option creates a new cache entry instead of silently reusing stale graphs. Cached graphs are stored as CPU tensors and moved to CUDA only after DataLoader collation.

You can warm the persistent cache before training so the first epoch does not pay the CIF-to-graph construction cost:

```bash
# ALIGNN-like cache: includes atom/bond graphs and line graphs.
python examples/precompute_graph_cache.py \
  --csv data/id_prop.csv \
  --target target \
  --model alignn_like \
  --graph-cache-dir .cache/materials_gnn_graphs \
  --max-line-neighbors 8 \
  --num-workers 4

# Then train with the same graph settings and cache directory.
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --graph-cache-dir .cache/materials_gnn_graphs \
  --max-line-neighbors 8 \
  --num-workers 4
```

Use `--model cgcnn` when warming a distance-only CGCNN cache, because CGCNN graphs do not include line-graph tensors. `--num-workers` uses CPU worker processes to build independent CIF graphs in parallel while saving the same persistent CPU-tensor cache that training later reads.

To force regeneration:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --graph-cache-dir .cache/materials_gnn_graphs \
  --overwrite-graph-cache
```


## ALIGNN-like memory controls

The ALIGNN-like model can use much more memory than the CGCNN-style model because every
angle path `i -> j -> k` becomes a line-graph edge. If a center atom has many incoming
and outgoing bonds, angle edges scale roughly with the square of the local coordination.
For large structures, high cutoffs, or dense KNN/Voronoi graphs, use the line-graph caps:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --batch-size 1 \
  --hidden-dim 64 \
  --num-layers 2 \
  --num-rbf 32 \
  --num-angle-rbf 16 \
  --max-line-neighbors 8 \
  --amp
```

For a hard safety cap per crystal, add for example:

```bash
--max-line-edges 20000
```

`--max-line-neighbors` is usually preferable to `--max-line-edges` because it limits each
local angular environment rather than truncating the graph globally. Changing either flag
changes the graph cache key, so cached graphs will be rebuilt automatically.

Before training, you can scan a CSV and rank structures by graph size and likely OOM risk:

```bash
python examples/analyze_graph_dataset.py \
  --csv data/id_prop.csv \
  --target target \
  --cutoff 5.0 \
  --hidden-dim 128 \
  --num-layers 3 \
  --batch-size 1 \
  --graph-cache-dir .cache/materials_gnn_graphs \
  --num-workers 4 \
  --output-csv runs/graph_stats.csv \
  --output-json runs/graph_stats_summary.json
```

The report includes atom counts, directed bond-edge counts, line-edge counts, distance
ranges, rough activation-memory estimates, and risk flags such as `line_edges>250000` or
`estimated_batch_mb>2000`. The script accepts the same graph strategy, basis, atom-feature,
and line-graph cap flags used by `train_alignn_like.py`, so it can preview the exact graph
settings planned for training. When `--graph-cache-dir` is provided, analysis also warms
or reuses the persistent cache; `--num-workers` parallelizes the CPU-bound graph build.

## Train an ALIGNN-like model

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --cutoff 5.0 \
  --epochs 50 \
  --batch-size 16
```

Use an alternate graph strategy:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --neighbor-strategy knn \
  --neighbor-k 12 \
  --neighbor-max-radius 8.0 \
  --rbf-cutoff 8.0 \
  --distance-basis bessel \
  --atom-features default
```

Use optional atom-graph edge weights from strategies that emit them:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --neighbor-strategy voronoi \
  --neighbor-max-radius 10.0 \
  --use-edge-weight
```

## Train a CGCNN-style model

```bash
python examples/train_cgcnn.py \
  --csv data/id_prop.csv \
  --target target \
  --cutoff 5.0
```

For CGCNN-style runs, the same `--use-edge-weight` flag enables weighted atom-graph
aggregation when the graph contains `edge_weight`.

## Implicit-bias readout experiment

The first implicit-bias experiment leaves graph construction and message passing unchanged
and only swaps the hidden activation in the final post-pooling readout head.

Baseline CGCNN:

```bash
python examples/train_cgcnn.py --csv data/id_prop.csv --target target --output-dir runs/cgcnn_baseline
```

Implicit-bias readout CGCNN:

```bash
python examples/train_cgcnn.py --csv data/id_prop.csv --target target \
  --readout-type implicit_bias \
  --ib-lambda 0.01 \
  --ib-sigma-slope 1.0 \
  --ib-fixed-point-iters 8 \
  --ib-coupling ring \
  --output-dir runs/cgcnn_ib_readout_lam001
```

Baseline ALIGNN-like:

```bash
python examples/train_alignn_like.py --csv data/id_prop.csv --target target --output-dir runs/alignn_baseline
```

Implicit-bias readout ALIGNN-like:

```bash
python examples/train_alignn_like.py --csv data/id_prop.csv --target target \
  --readout-type implicit_bias \
  --ib-lambda 0.01 \
  --ib-sigma-slope 1.0 \
  --ib-fixed-point-iters 8 \
  --ib-coupling ring \
  --output-dir runs/alignn_ib_readout_lam001
```

## Predict from a CIF

Training writes `experiment_config.json`, `training_history.json`, `best_model.pt`, and
`final_model.pt` into the run directory. The JSON config and checkpoint metadata include
the package version, target column, split seed and indices, model architecture, graph
strategy, featurization settings, target normalizer, and training CLI arguments.
Each training-history epoch also records timing fields such as `train_seconds`,
`val_seconds`, `epoch_seconds`, `elapsed_seconds`, and train throughput estimates, so
architectural changes can be compared on speed as well as metrics.

The checkpoints also retain top-level inference metadata, so prediction can reconstruct
the model architecture plus CIF-to-graph settings such as cutoff, neighbor strategy,
distance basis, atom features, and ALIGNN line-graph memory caps. For new checkpoints,
prediction therefore only needs the CIF path and checkpoint path:

```bash
python examples/predict_from_cif.py \
  --checkpoint runs/alignn_like/best_model.pt \
  --cif data/cifs/Si.cif
```

For older checkpoints that do not contain config metadata, keep passing the model and
graph flags explicitly, or add `--ignore-checkpoint-config` to force CLI settings. This
fallback path also supports optional edge weighting and implicit-bias readout flags, so
legacy implicit-bias checkpoints can be reconstructed when the CLI arguments match the
training architecture:

```bash
python examples/predict_from_cif.py \
  --checkpoint runs/cgcnn_ib_readout_lam001/best_model.pt \
  --cif data/cifs/Si.cif \
  --model cgcnn \
  --num-rbf 64 \
  --hidden-dim 128 \
  --num-layers 3 \
  --readout-type implicit_bias \
  --ib-lambda 0.01 \
  --ib-sigma-slope 1.0 \
  --ib-fixed-point-iters 8 \
  --ib-coupling ring
```

Add `--ib-trainable-lambda` and `--use-edge-weight` when those options were used during
training.

## Compare training runs

Use `examples/compare_runs.py` to flatten completed run directories into one sortable
table. It reads `experiment_config.json`, `training_history.json`, and
`test_predictions.csv`, then reports readout parameters, validation metrics, recomputed
test MAE/RMSE/R2, and timing fields.

```bash
python examples/compare_runs.py \
  --root runs \
  --output-csv runs/run_comparison.csv \
  --group-by readout_type
```

Compare only selected runs:

```bash
python examples/compare_runs.py \
  runs/cgcnn_baseline \
  runs/cgcnn_ib_readout_lam001 \
  --sort-by test_mae \
  --output-csv runs/cgcnn_readout_comparison.csv
```

## Core graph fields

A bond graph contains:

```python
{
    "z": atomic_numbers,                 # [num_atoms]
    "edge_index": edge_index,            # [2, num_edges], directed i -> j
    "edge_vec": edge_displacement_vecs,  # [num_edges, 3]
    "distance": bond_distances,          # [num_edges]
    "edge_attr": distance_features,      # [num_edges, num_rbf]
    "edge_unit_vec": edge_directions,     # [num_edges, 3]
    "atom_attr": optional_atom_features,  # [num_atoms, num_atom_features]
    "num_nodes": num_atoms,
}
```

Strategies such as Voronoi or adaptive shells may also attach:

```python
{
    "edge_weight": geometric_edge_weights,  # [num_edges]
}
```

`edge_weight` is optional. `CGCNNModel` and `ALIGNNLikeModel` consume it only when
constructed with `use_edge_weight=True` or trained with `--use-edge-weight`.

An ALIGNN-like graph additionally contains:

```python
{
    "line_edge_index": line_edge_index,  # [2, num_line_edges]
    "line_edge_attr": angle_features,    # [num_line_edges, num_angle_rbf]
    "angle": raw_angles_radians,          # [num_line_edges]
    "cosine": raw_angle_cosines,          # [num_line_edges]
}
```

## Materials-science concepts in the code

Crystals need periodic neighbor graphs because the unit cell repeats infinitely in three dimensions. A neighbor of an atom may live in the same unit cell or in a translated periodic image. The graph therefore stores periodic displacement vectors, not just atom indices.

Distances are expanded with radial basis functions because the raw scalar distance is a compact but difficult representation. Smooth basis functions make it easy for an MLP to learn distance-dependent filters such as short-bond, medium-range, and cutoff-shell interactions.

ALIGNN-style models use a line graph because pairwise distances alone do not capture local geometry. In the line graph, each directed bond is a node, and each path `i -> j -> k` creates a line-graph edge carrying the bond angle at `j`. Updating bond states through this graph injects three-body angular information before atom message passing.

Graph pooling creates a crystal-level embedding from atom-level states. Mean pooling is the default because it is simple and stable for scalar intensive properties. Sum pooling is also included and may be preferable for extensive properties.

Scalar property prediction is performed by an MLP readout head applied to the pooled crystal embedding.

## Extension points

The current package is deliberately small, but the interfaces are designed for research iteration:

- Compare cutoff, k-nearest-neighbor, Voronoi, adaptive-shell, and strain-consensus graph construction under identical models.
- Add or remove elemental descriptors from `DEFAULT_ELEMENTAL_FEATURES`, or provide site-specific `atom_attr`.
- Swap Gaussian RBFs for Bessel, Fourier, or learned Gaussian bases.
- Add attention, Set2Set, or gated pooling.
- Add multi-task heads, classification heads, or uncertainty heads.
- Add Matbench task adapters and standardized benchmark splits.
- Add equivariant layers that use directions, not only distances and angles.

## Run tests

```bash
pytest
```

Tests that require `pymatgen` are skipped if `pymatgen` is not installed.
