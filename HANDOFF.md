# Project Handoff

## 1. Project Overview

`materials-gnn` is a research-oriented Python package for materials property prediction with crystal graph neural networks. It is intended for materials informatics researchers, scientific machine learning engineers, and developers who want an extensible platform for converting crystal structures into periodic graphs, training GNN models, and predicting scalar material properties such as formation energy, band gap, elastic modulus, stability metrics, or related regression/classification targets.

The package currently provides a clear raw-PyTorch prototype inspired by CGCNN and ALIGNN. It is not meant to be a highly optimized reproduction of either paper. The main goal is modular experimentation: swap graph builders, atom/bond/angle featurizers, message-passing layers, pooling/readout modules, losses, metrics, and benchmark datasets without rewriting the whole stack.

Current package version in `pyproject.toml`: `0.4.2`.

Current active branch for the implicit-bias research direction: `codex/implicit-bias-readout`.

## 2. Current Architecture

This is a Python research package, not a web application. There is no frontend, backend server, database service, authentication layer, or production API server.

### Core Python Library

The package is organized around a CIF/`pymatgen.Structure` to graph to model to metrics workflow:

1. Load crystal structures with `pymatgen`.
2. Build periodic atom/bond graphs using configurable neighbor strategies.
3. Optionally build ALIGNN-style line graphs for bond-angle message passing.
4. Encode atomic numbers, optional elemental descriptors, bond distances, and bond angles.
5. Train raw PyTorch CGCNN-style or ALIGNN-like models for scalar prediction.
6. Evaluate with MAE, RMSE, R2, parity/residual plots, and CSV prediction export.

### Storage

There is no database. Storage is file-based:

- Input datasets are CSV files with columns such as `material_id,cif_path,target`.
- CIF files are read from disk through paths listed in the CSV.
- Persistent precomputed graph cache entries are saved with `torch.save` under a user-provided cache directory, usually `.cache/materials_gnn_graphs/graphs/*.pt`. Cache keys include only graph-defining inputs, not target labels or material IDs.
- Training checkpoints are saved as PyTorch checkpoint dictionaries, usually under `runs/<model_name>/best_model.pt`.
- Prediction outputs are CSV files, usually `runs/<model_name>/test_predictions.csv`.
- The repository now has a `.gitignore` for local graph caches, generated CIF collections, run outputs, package build artifacts, Python bytecode, and checkpoint/tensor files.

### APIs

The project exposes both Python APIs and CLI example scripts.

Important Python APIs:

- `materials_gnn.featurization.structure_to_bond_graph`
- `materials_gnn.featurization.add_line_graph`
- `materials_gnn.featurization.make_neighbor_strategy`
- `materials_gnn.featurization.make_basis_expansion`
- `materials_gnn.data.CrystalGraphDataset`
- `materials_gnn.data.collate_graphs`
- `materials_gnn.models.CGCNNModel`
- `materials_gnn.models.ALIGNNLikeModel`
- `materials_gnn.training.train_model`
- `materials_gnn.training.evaluate_model`
- `materials_gnn.inference.predict_cif`

Current CLI entry points are example scripts, not installed console commands:

- `examples/train_cgcnn.py`
- `examples/train_alignn_like.py`
- `examples/predict_from_cif.py`
- `examples/analyze_graph_dataset.py`
- `examples/precompute_graph_cache.py`
- `examples/compare_runs.py`

### Background Jobs / Workers

There are no application-level background jobs. PyTorch `DataLoader` worker processes are supported through `--num-workers`. These workers may concurrently load CIFs and build/cache graphs. Persistent cache writes are atomic via temporary files plus `os.replace`, but multiple workers can still redundantly build the same graph before one cache result wins.

### Third-Party Services and Libraries

No external hosted services are used. Major dependencies are:

- `torch` for models, tensors, training, checkpointing, and graph-cache serialization.
- `pymatgen` for crystal structure loading and periodic neighbor/Voronoi operations.
- `numpy` and `pandas` for preprocessing and CSV data handling.
- `scikit-learn` listed as a dependency for future evaluation/splitting extensions.
- `matplotlib` as an optional dev/plotting dependency.
- `pytest` as an optional dev/test dependency.

### Infrastructure / Deployment Setup

There is no deployment infrastructure yet. The intended current usage is local, Colab, workstation, or cluster execution with editable installation via `pip install -e .`.

## 3. Repository Structure

```text
materials_gnn_project/
├── README.md
├── HANDOFF.md
├── pyproject.toml
├── examples/
│   ├── train_cgcnn.py
│   ├── train_alignn_like.py
│   └── predict_from_cif.py
├── materials_gnn/
│   ├── data/
│   ├── evaluation/
│   ├── featurization/
│   ├── inference/
│   ├── models/
│   ├── training/
│   └── utils/
└── tests/
```

Important folders and files:

- `pyproject.toml`  
  Package metadata, build system, dependencies, pytest config, and lint config. Current version is `0.4.2`.

- `README.md`  
  User-facing project overview, install instructions, graph-strategy descriptions, featurization options, CUDA/cache notes, memory controls, and example commands.

- `materials_gnn/featurization/basis.py`  
  Configurable scalar basis expansions for distances and angles. Supports static Gaussian, Bessel/sine, Fourier, and model-level learnable Gaussian basis modules.

- `materials_gnn/featurization/elemental_features.py`  
  Atomic-number embedding, optional elemental descriptor featurizer, descriptor parsing/aliases, descriptor normalization, missing indicators, and `AtomFeatureEncoder`.

- `materials_gnn/featurization/neighbor_strategies.py`  
  Neighbor construction strategy layer. Supports cutoff, KNN, Voronoi, adaptive shell, and strain-consensus strategies. Defines `NeighborList` and `make_neighbor_strategy`.

- `materials_gnn/featurization/crystal_graph.py`  
  Converts a `pymatgen.Structure` or CIF path into a directed periodic atom/bond graph with distance features, edge vectors, positions, optional atom descriptors, and optional edge weights.

- `materials_gnn/featurization/line_graph.py`  
  Builds ALIGNN-style line graphs where directed bonds become nodes and line-graph edges encode `i -> j -> k` bond angles. Includes memory caps for dense angular graphs.

- `materials_gnn/data/datasets.py`  
  CSV-backed `CrystalGraphDataset`, RAM graph cache support, persistent graph-cache integration, target access/normalization helper methods, and `collate_graphs` for variable-size graph batches.

- `materials_gnn/data/graph_cache.py`  
  File-backed graph cache keyed by CIF fingerprint and graph/feature/line-graph configuration.

- `materials_gnn/data/cache_precompute.py`  
  Utility for warming persistent graph caches and reporting cache hits, newly built graphs, and per-row failures. Supports multiprocessing with `num_workers` / `--num-workers`.

- `materials_gnn/data/datamodules.py`  
  Lightweight DataLoader factory. No PyTorch Lightning dependency.

- `materials_gnn/data/splits.py` and `materials_gnn/data/transforms.py`  
  Reproducible train/validation/test splits and scalar target normalization.

- `materials_gnn/models/layers.py`  
  `GatedGraphConv`, a reusable edge-gated message-passing layer for atom/bond graphs and line graphs. It supports optional atom-graph `edge_weight` aggregation when callers pass weights. Also contains `build_mlp`.

- `materials_gnn/models/cgcnn.py`  
  Distance-only atom/bond graph model for scalar prediction.

- `materials_gnn/models/alignn.py`  
  ALIGNN-inspired model that alternates line-graph bond/angle updates and atom/bond updates.

- `materials_gnn/models/readout.py`  
  Mean/sum graph pooling, the standard MLP readout head, and a `make_readout` helper for standard vs implicit-bias readouts.

- `materials_gnn/models/implicit_bias.py`  
  Optional implicit-bias readout components: `ImplicitBiasActivation` and `ImplicitBiasMLPReadout`. This branch only uses the implicit-bias neuron idea after crystal pooling, not inside message passing.

- `materials_gnn/training/device.py`  
  CPU/CUDA device resolution, recursive batch movement, DataLoader pinned-memory kwargs, and optional float32 matmul precision setting.

- `materials_gnn/training/trainer.py`  
  Minimal supervised training loop with AdamW, optional CUDA AMP, validation metrics, checkpointing, target inverse normalization, and per-epoch timing/throughput records.

- `materials_gnn/training/metrics.py`  
  MAE, RMSE, and R2 metrics.

- `materials_gnn/evaluation/parity_plots.py`  
  Parity plot, residual plot, and prediction CSV export utilities.

- `materials_gnn/evaluation/matbench.py` and `materials_gnn/evaluation/uncertainty.py`  
  Placeholder extension hooks.

- `materials_gnn/inference/predict.py`  
  Single-structure and CIF prediction helpers.

- `materials_gnn/utils/config.py`, `logging.py`, `registry.py`  
  Early lightweight utilities for dataclass configs, logging, and extension registries.

- `tests/`  
  Unit tests for basis features, line graph construction, model forward passes, dataset collation, graph cache, device helpers, graph construction, and neighbor strategies.

## 4. Current State

| Area | Status | Notes |
|---|---|---|
| Package skeleton | Working | Standard Python package layout with editable install support. |
| CIF/Structure loading | Working | Uses `pymatgen`. Tests requiring `pymatgen` skip if unavailable. |
| Periodic cutoff graph construction | Working | `CutoffNeighborStrategy` is the backward-compatible default. |
| KNN graph construction | Working | `KNearestNeighborStrategy` uses periodic candidates up to `max_radius`; controls outgoing edge count. |
| Voronoi graph construction | Working | Uses `pymatgen.analysis.local_env.VoronoiNN`; provides optional `edge_weight`. Models can consume it when `use_edge_weight=True` / `--use-edge-weight`. |
| Adaptive-shell graph construction | Working / Experimental | Local distance-gap strategy. Intended as a research baseline, not validated. |
| Strain-consensus graph construction | Working / Experimental | Keeps cutoff edges stable under small virtual strains and emits `edge_weight`. Novelty not guaranteed; treat as speculative. |
| Distance basis features | Working | Static Gaussian, Bessel/sine, Fourier preprocessing; model-level learnable Gaussian supported. |
| Angle basis features | Working | Static Gaussian, Bessel/sine, Fourier preprocessing; model-level learnable Gaussian supported. |
| Elemental descriptors | Working / Partial | Descriptor lookup through `pymatgen` where available. Missing values get indicators. Magnetic moment is especially environment-dependent and often missing. |
| Equivariant-ready geometric fields | Partial | Graphs include `pos`, `edge_vec`, `edge_unit_vec`, `distance`, `angle`, and `cosine`; no equivariant model layers yet. |
| CGCNN-style model | Working | Plain PyTorch atom/bond graph model with gated message passing and pooling. Optional atom-graph edge weighting is available via `use_edge_weight=True` or `--use-edge-weight`. |
| ALIGNN-like model | Working | Alternates line-graph bond/angle update and atom/bond update. Optional `edge_weight` affects only atom/bond updates, not line-graph updates. Can run out of memory on dense line graphs without caps. |
| Implicit-bias readout | Working / Experimental | Optional post-pooling readout head for `CGCNNModel` and `ALIGNNLikeModel` via `readout_type="implicit_bias"` or `--readout-type implicit_bias`. Default remains the original MLP readout. |
| Line-graph memory controls | Working | `max_outgoing_neighbors`, `max_line_edges`, and `line_neighbor_selection` are implemented and exposed in `train_alignn_like.py`. |
| Dataset and collation | Working | Supports variable-size graph batching, node/edge/line-edge index offsets, optional `atom_attr`, `edge_weight`, `edge_unit_vec`, raw angles/cosines. |
| Target normalization | Working | Train-set mean/std via `TargetNormalizer`. Validation/test metrics are inverse-transformed when normalizer is supplied. |
| Training loop | Working | CPU-first, CUDA-compatible, optional AMP, AdamW, checkpointing, validation metrics, and training-history timing fields such as `train_seconds`, `val_seconds`, `epoch_seconds`, `elapsed_seconds`, and throughput estimates. |
| CUDA support | Working / Basic | `device='auto'`, batch movement, pinned memory, optional AMP. Multi-GPU/distributed training not implemented. |
| Persistent graph cache | Working | File-backed CPU tensor cache keyed by CIF fingerprint and graph configuration. Target labels, target column names, material IDs, and training-only settings are excluded from cache identity. Cache precomputation and graph analysis can use multiprocessing workers. |
| Evaluation utilities | Working / Basic | MAE, RMSE, R2, parity/residual plots, CSV export, and run comparison via `examples/compare_runs.py`. Matbench and uncertainty are placeholders. |
| Inference from CIF | Working | New checkpoints store model and graph/inference config so `predict_from_cif.py` can reconstruct settings automatically. Legacy CLI fallbacks now include edge-weight and implicit-bias readout flags. |
| Classification targets | Not started | Current pipeline assumes scalar regression. |
| Multi-task prediction | Not started | `output_dim` exists, but dataset/training/evaluation are scalar-oriented. |
| Attention/gated pooling | Not started | Only mean and sum pooling are implemented. |
| Config-driven experiments | Partial / Working | Example scripts still use argparse, but each run now writes `experiment_config.json` and embeds full experiment metadata in checkpoints. |
| Deployment | Not started | No package publishing, Docker, CI/CD, or cluster launch scripts. |

Last full local test result recorded before the current implicit-bias/timing branch work:

```text
28 passed, 2 skipped
```

The skipped tests are `pymatgen`-dependent in environments where `pymatgen` is not installed or unavailable.

Current branch verification note for the implicit-bias checkpoint fallback update:

```text
C:\Users\mscoo\miniforge3\envs\CGCNN\python.exe -m pytest tests\test_experiment_config.py tests\test_training_cli_config.py --basetemp .codex_tmp_pytest -p no:cacheprovider
6 passed, 2 warnings
```

The warnings are PyTorch `torch.load(..., weights_only=False)` future warnings.

## 5. Key Decisions Made

- **Use raw PyTorch first, not PyTorch Geometric or DGL.**  
  Reasoning: the first prototype should be readable, dependency-light, and easy to modify. Scatter/index operations are implemented directly with `index_add_`.

- **Use `pymatgen` for structure parsing and periodic geometry.**  
  Reasoning: CIF parsing, periodic neighbor lists, lattice operations, and Voronoi coordination are specialized and already well supported by `pymatgen`.

- **Represent graphs as dictionaries of tensors.**  
  Reasoning: this keeps the first version framework-agnostic and easy to inspect. The canonical atom/bond graph contains `z`, `edge_index`, `edge_vec`, `distance`, `edge_attr`, `edge_unit_vec`, `pos`, and `num_nodes`.

- **Use directed periodic edges.**  
  Reasoning: crystal neighbor interactions cross unit-cell boundaries. Each edge stores the displacement vector to the periodic image, not just atom indices.

- **Separate topology from featurization.**  
  Reasoning: graph topology experiments, such as cutoff vs KNN vs Voronoi, should be independent from distance/angle/atom feature experiments.

- **Expose graph construction through strategy objects.**  
  Reasoning: new graph builders only need to implement `build(structure) -> NeighborList`. This makes graph-construction research safer and cleaner.

- **Add experimental graph builders but label them clearly.**  
  Reasoning: `adaptive_shell` and `strain_consensus` are useful hypotheses, but should not be treated as validated best practices.

- **Support both preprocessing bases and model-level learnable bases.**  
  Reasoning: static Gaussian/Bessel/Fourier features are cheap and cacheable; learnable Gaussian bases should live inside the model checkpoint and operate on raw `distance`/`angle` tensors.

- **Append missing-value indicators to elemental descriptors by default.**  
  Reasoning: not all descriptor values are known or well-defined for every element, and missingness itself may carry useful information.

- **Keep `edge_weight` optional and make weighted aggregation explicit.**  
  Reasoning: Voronoi and strain-consensus strategies can emit geometric weights, but default model behavior must remain unchanged. `CGCNNModel` and `ALIGNNLikeModel` therefore ignore `graph["edge_weight"]` unless `use_edge_weight=True` or `--use-edge-weight` is set. `ALIGNNLikeModel` applies this only to atom/bond graph updates; line-graph edge weights are intentionally not implemented.

- **Prototype implicit-bias neurons only in the readout first.**  
  Reasoning: the current branch adds an optional implicit-bias hidden activation after crystal pooling, leaving graph construction and `GatedGraphConv` message passing unchanged. This isolates the research variable and keeps default model behavior unchanged.

- **Record timing in training history.**  
  Reasoning: model and graph changes should be compared on speed as well as metrics. Epoch records now include train/validation/total elapsed time and train throughput estimates, and checkpoints retain the same history payload.

- **Use mean pooling by default.**  
  Reasoning: many scalar materials properties are intensive. Sum pooling is available and may be better for extensive targets.

- **Cache precomputed graphs as CPU tensors.**  
  Reasoning: graph construction is CPU/pymatgen-heavy, and CPU tensor cache files are portable across machines and GPU types. Batches are moved to CUDA only after collation.

- **Enable CUDA only after stabilizing schema/collation/forward paths.**  
  Reasoning: current stage is appropriate for CUDA because the graph data schema and models now run, but graph construction remains CPU-bound.

- **Add line-graph caps instead of assuming ALIGNN-like graphs fit in memory.**  
  Reasoning: line-graph edge count scales roughly with squared local coordination, so dense structures can exceed GPU memory quickly.

## 6. Important Context

- This project is a **research platform**, not a production-optimized benchmark implementation.

- The current model names are intentionally `CGCNNModel` and `ALIGNNLikeModel`. The ALIGNN-like model is inspired by ALIGNN's line-graph idea, but it is not a faithful reproduction.

- The current graph schema should not be changed casually. Many components expect these keys:

  ```python
  "z", "edge_index", "edge_vec", "distance", "edge_attr", "edge_unit_vec", "pos", "num_nodes"
  ```

  ALIGNN-like workflows additionally expect:

  ```python
  "line_edge_index", "line_edge_attr", "angle", "cosine"
  ```

- `edge_index` is directed and has shape `[2, num_edges]`, with columns `source -> destination`.

- `edge_vec` points from the source atom to the destination atom's periodic image. This is important for correct angle construction.

- `build_line_graph` treats each directed bond as a line-graph node. A line-graph edge connects bond `i -> j` to bond `j -> k`, and its feature is the bond angle centered at `j`.

- `skip_backtracking=True` is the default for line graphs, so immediate paths `i -> j -> i` are skipped unless explicitly enabled.

- The ALIGNN-like model can OOM even on large GPUs if graphs are dense. Use these knobs first:

  ```bash
  --batch-size 1
  --max-line-neighbors 6
  --hidden-dim 64
  --num-layers 2
  --num-rbf 32
  --num-angle-rbf 16
  --cutoff 4.0
  --amp
  ```

- In Colab or CUDA environments with allocator fragmentation, set the allocator environment variable before importing torch:

  ```python
  import os
  os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
  ```

  Then restart the runtime and rerun from the top.

- A previous CUDA AMP bug in `GatedGraphConv` was fixed. Aggregation buffers must be created from the tensors being accumulated, not from the original float32 node tensor. Do not revert this pattern:

  ```python
  aggregate = weighted_messages.new_zeros((num_nodes, self.node_dim))
  gate_sum = gates.new_zeros((num_nodes, self.node_dim))
  ```

- The CIF parser warning about fractional coordinates being rounded to ideal values is usually not the cause of CUDA OOM. OOMs seen so far have come from dense line graphs.

- Elemental descriptors are optional. Atomic-number embeddings remain the safest default. Some requested descriptors are not universally available or well-defined; the featurizer fills missing values and appends missing indicators.

- Magnetic moment is not generally an environment-independent elemental property. The current implementation only uses a tabulated value when available and otherwise marks it missing.

- Persistent graph cache keys include CIF fingerprint and graph/feature/line-graph settings. Changing `num_rbf`, atom descriptors, neighbor strategy, line-graph caps, or similar settings should automatically create a different cache entry. Target labels, target column names, material IDs, splits, and training-only settings do not affect the cache key. During debugging, `--overwrite-graph-cache` is still safest.

- New checkpoints store model weights, optimizer-independent inference metadata, target normalizer state, and the full experiment configuration. `predict_from_cif.py` reads these settings automatically for new checkpoints; legacy checkpoints still require CLI fallbacks, including readout and implicit-bias flags when the saved model used `readout_type="implicit_bias"`.

- `edge_weight` is collated and can be used by `CGCNNModel` and `ALIGNNLikeModel` when explicitly enabled. The layer accepts `[num_edges]` or `[num_edges, 1]` weights, casts them to the message dtype/device, and otherwise leaves values unchanged. Graph builders are responsible for meaningful weights.

- Optional implicit-bias readout is controlled by model config fields and CLI flags: `readout_type`, `ib_lambda`, `ib_sigma_slope`, `ib_fixed_point_iters`, `ib_coupling`, and `ib_trainable_lambda`. Existing commands default to `readout_type="mlp"`. Training scripts and the `predict_from_cif.py` fallback path now expose these flags.

- `training_history.json` now records timing fields per epoch. For comparing architecture changes, start with `train_samples_per_second` and use `epoch_seconds` as a sanity check.

- `num_workers > 0` can help hide graph construction time, but can also increase CPU/RAM pressure. With persistent graph caching, the first epoch may still be expensive because graphs are built lazily.

## 7. Active Workstream

The active workstream is improving the package from a minimal CGCNN/ALIGNN-like prototype into a durable materials-GNN research platform.

Immediate recent goals:

1. Add graph-construction strategy layer.
2. Upgrade atom, bond, and angle featurization.
3. Add CUDA support and persistent graph caching.
4. Address CUDA AMP dtype mismatch.
5. Address ALIGNN-like line-graph CUDA OOM with memory controls.
6. Produce this handoff document so another developer or AI agent can continue without losing context.
7. Add an optional implicit-bias readout head after crystal pooling for first implicit-bias experiments.
8. Add training-history timing and throughput fields for speed comparisons across model variants.
9. Add optional atom-graph edge-weight aggregation for CGCNN/ALIGNN-like experiments.
10. Keep CIF prediction compatible with implicit-bias checkpoints through both checkpoint metadata and explicit legacy fallback CLI arguments.

Files likely involved in the next iteration:

- `materials_gnn/featurization/neighbor_strategies.py`
- `materials_gnn/featurization/crystal_graph.py`
- `materials_gnn/featurization/line_graph.py`
- `materials_gnn/featurization/basis.py`
- `materials_gnn/featurization/elemental_features.py`
- `materials_gnn/models/layers.py`
- `materials_gnn/models/implicit_bias.py`
- `materials_gnn/models/cgcnn.py`
- `materials_gnn/models/alignn.py`
- `materials_gnn/models/readout.py`
- `materials_gnn/data/datasets.py`
- `materials_gnn/data/graph_cache.py`
- `materials_gnn/data/cache_precompute.py`
- `materials_gnn/training/trainer.py`
- `examples/train_alignn_like.py`
- `examples/train_cgcnn.py`
- `examples/predict_from_cif.py`
- `examples/analyze_graph_dataset.py`
- `examples/precompute_graph_cache.py`
- `examples/compare_runs.py`
- `tests/`

Current implementation approach:

- Keep public graph dictionaries backward-compatible.
- Add optional fields instead of replacing existing ones.
- Make new behavior configurable through constructor args and CLI flags.
- Add focused tests for each new capability.
- Prefer clarity and correctness over performance until benchmark infrastructure is added.

What remains to be done:

- Add proper config files and reproducible experiment runners.
- Add benchmark dataset adapters, especially Matbench-style workflows.
- Add additional pooling methods such as attention pooling.
- Add classification and multi-task support.
- Add first equivariant/geometric layer that consumes `edge_vec` or `edge_unit_vec`.

Blockers or uncertainties:

- Need representative real datasets to validate memory settings and graph strategy behavior.
- Need decide whether to stay raw PyTorch or introduce PyG/DGL for performance and batching.
- Need decide how much compatibility to preserve once model/config serialization is improved.
- Need decide whether experimental graph strategies should remain in core package or move behind an `experimental` namespace.

## 8. Known Issues and Technical Debt

| Issue | Impact | Suggested next step |
|---|---|---|
| Legacy checkpoints may lack training/inference config | Old `best_model.pt` files may still require manually matching architecture, featurization, edge-weight, and readout flags | Prefer new checkpoints with embedded `experiment_config`; keep CLI fallback path for old checkpoints |
| Atom-graph `edge_weight` is opt-in | Voronoi and strain-consensus weights still have no effect unless users pass `use_edge_weight=True` or `--use-edge-weight` | Compare weighted vs unweighted runs with identical seeds/splits before treating the weights as beneficial |
| Implicit-bias readout is experimental | It may affect accuracy, stability, and speed in target-dependent ways | Compare baseline vs implicit-bias runs using identical seeds/splits and inspect both metrics and timing fields |
| ALIGNN-like line graph can still OOM | Dense structures or large cutoffs can exceed GPU memory | Add graph statistics logging before training, optional dataset filtering, and automatic warnings when line-edge counts exceed thresholds |
| Graph construction is lazy before cache warms | First epoch can be slow if cache was not precomputed | Use `examples/precompute_graph_cache.py --num-workers N` to populate the persistent cache before training; future work can add pruning/indexing and duplicate-key de-duplication |
| Persistent graph cache has no pruning/index | Cache directory can grow indefinitely across experiments | Add cache manifest, size reporting, and cleanup utilities by namespace/date/config |
| Raw PyTorch `index_add_` message passing is simple but not optimized | Training may be slower than PyG/DGL/scatter backends for large datasets | Benchmark bottlenecks; consider optional PyG/DGL backend only after correctness baselines are stable |
| Matbench integration is a placeholder | No standardized benchmark splits/tasks yet | Implement `MatbenchAdapter` and a CLI that exports standardized results |
| Uncertainty module is a placeholder | No uncertainty estimates despite research goal | Start with deep ensembles and MC dropout before evidential heads |
| Only scalar regression is fully supported | Classification and multi-task objectives are not ready | Generalize dataset labels, losses, metrics, output heads, and evaluation export |
| Pooling is limited to mean/sum | May underperform for variable-size or site-importance-sensitive properties | Add attention pooling and gated pooling with tests |
| Elemental descriptors are static and site-independent | Oxidation state, spin state, charge, and local chemistry are not represented | Add optional site features and oxidation-state-aware featurizers when data is available |
| Magnetic moment descriptor is weakly defined | Can mislead users if interpreted as material/site magnetism | Keep missing indicators; document clearly; prefer site/property-specific magnetic inputs later |
| Graph statistics report is local-only | Users can inspect OOM risks, but there is no centralized report dashboard or automatic training gate yet | Use `analyze_graph_dataset.py --graph-cache-dir ... --num-workers N` before training; future work can add dataset filtering and warnings inside training |
| No reproducibility manifest | Results depend on CLI args, code version, PyTorch version, CUDA, and cache state | Save config, dependency versions, git commit if available, random seeds, and cache namespace in run directory |
| No CI config | Regressions may be missed outside manual pytest runs | Add GitHub Actions or equivalent with CPU tests and optional pymatgen install |
| Formatting/linting not enforced | Style can drift | Add `ruff`/`black` commands to developer docs and CI |
| No package publishing workflow | Users install from local folder only | Add build/publish instructions later if needed |

## 9. Setup and Running Instructions

### Prerequisites

- Python `>=3.10`
- `pip`
- CPU-only PyTorch or CUDA-enabled PyTorch appropriate for the machine
- `pymatgen` for CIF/Structure graph construction
- Optional NVIDIA GPU for CUDA training

### Install

From the repository root:

```bash
pip install -e .
```

Install minimal dependencies manually if needed:

```bash
pip install torch pymatgen numpy pandas scikit-learn
```

Install dev/test/plot dependencies:

```bash
pip install pytest matplotlib
```

Or install the package dev extra if supported by the environment:

```bash
pip install -e '.[dev]'
```

### Environment Variables

No environment variable is required for normal CPU training.

Optional CUDA allocator setting for Colab or fragmented CUDA memory situations:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

In notebooks, set this before importing `torch`, then restart the runtime:

```python
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

### Expected Dataset Format

CSV example:

```text
material_id,cif_path,target
mp-149,data/cifs/Si.cif,-5.42
mp-13,data/cifs/Fe.cif,-8.31
```

Relative `cif_path` values are resolved relative to the CSV file location unless a different dataset root is supplied in Python code.

### Train CGCNN-Style Model

```bash
python examples/train_cgcnn.py \
  --csv data/id_prop.csv \
  --target target \
  --cutoff 5.0 \
  --epochs 50 \
  --batch-size 16
```

With graph cache and CUDA:

```bash
python examples/train_cgcnn.py \
  --csv data/id_prop.csv \
  --target target \
  --device cuda \
  --amp \
  --graph-cache-dir .cache/materials_gnn_graphs
```

### Train ALIGNN-Like Model

Basic:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --cutoff 5.0 \
  --epochs 50 \
  --batch-size 16
```

Memory-safer CUDA command:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --device cuda \
  --amp \
  --num-workers 2 \
  --batch-size 1 \
  --hidden-dim 64 \
  --num-layers 2 \
  --num-rbf 32 \
  --num-angle-rbf 16 \
  --max-line-neighbors 8 \
  --graph-cache-dir .cache/materials_gnn_graphs
```

Use KNN graph construction:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --neighbor-strategy knn \
  --neighbor-k 12 \
  --neighbor-max-radius 8.0 \
  --rbf-cutoff 8.0
```

Use upgraded featurization:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --distance-basis bessel \
  --angle-basis fourier \
  --atom-features default
```

Use learnable distance and angle bases:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --learnable-distance-basis \
  --learnable-angle-basis
```

Use optional atom-graph edge weights from graph builders that emit `edge_weight`:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --neighbor-strategy voronoi \
  --neighbor-max-radius 10.0 \
  --use-edge-weight
```

Use the current implicit-bias readout experiment while leaving graph construction and
message passing unchanged:

```bash
python examples/train_cgcnn.py \
  --csv data/id_prop.csv \
  --target target \
  --readout-type implicit_bias \
  --ib-lambda 0.01 \
  --ib-sigma-slope 1.0 \
  --ib-fixed-point-iters 8 \
  --ib-coupling ring
```

After training, compare speed through per-epoch fields in `training_history.json`,
especially `train_samples_per_second`, `train_seconds`, and `epoch_seconds`.

For parameter sweeps, use `examples/compare_runs.py` to flatten run directories into a
sortable table/CSV with readout settings, validation metrics, recomputed test metrics,
and timing fields:

```bash
python examples/compare_runs.py \
  --root runs \
  --output-csv runs/run_comparison.csv \
  --group-by readout_type
```

### Predict from a CIF

New checkpoints store model and graph configuration, so prediction can usually reconstruct
the training settings from the checkpoint. Legacy checkpoints may still need explicit
model, graph, edge-weight, and readout flags.

```bash
python examples/predict_from_cif.py \
  --model alignn_like \
  --checkpoint runs/alignn_like/best_model.pt \
  --cif data/cifs/Si.cif \
  --cutoff 5.0
```

For a legacy implicit-bias checkpoint without metadata, pass the readout arguments that
match training:

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

Add `--ib-trainable-lambda` and `--use-edge-weight` if those were enabled in the saved
model.

### Tests

```bash
pytest
```

### Build Commands

No build step is required for local development. To build a wheel/sdist later, use standard Python packaging tools such as:

```bash
python -m build
```

`build` is not currently listed as a project dependency.

### Database Migration / Seed Commands

Not applicable. There is no database.

## 10. Testing Strategy

### Framework

The project uses `pytest`.

### Test Location

Tests live in `tests/`:

- `test_dataset_collate.py`
- `test_device.py`
- `test_featurization.py`
- `test_graph_cache.py`
- `test_graph_construction.py`
- `test_line_graph.py`
- `test_models.py`
- `test_neighbor_strategies.py`
- `test_experiment_config.py`
- `test_training_cli_config.py`
- `test_compare_runs.py`

### How to Run

From repository root:

```bash
pytest
```

Last full-suite result recorded before the current implicit-bias/timing branch work:

```text
28 passed, 2 skipped
```

Focused verification for the implicit-bias checkpoint fallback update:

```text
C:\Users\mscoo\miniforge3\envs\CGCNN\python.exe -m pytest tests\test_experiment_config.py tests\test_training_cli_config.py --basetemp .codex_tmp_pytest -p no:cacheprovider
6 passed, 2 warnings
```

Focused verification for the run-comparison utility:

```text
C:\Users\mscoo\miniforge3\envs\CGCNN\python.exe -m pytest tests\test_compare_runs.py --basetemp .codex_tmp_pytest -p no:cacheprovider
3 passed
```

### What Is Covered

- Scalar basis expansion shapes and finite values.
- Learnable Gaussian basis trainable parameters.
- Elemental descriptor featurizer when `pymatgen` is available.
- Line graph angle computation, backtracking skip behavior, and memory caps.
- CGCNN and ALIGNN-like forward-pass shapes.
- Model-level learnable distance/angle basis forward paths.
- Implicit-bias activation shape/finite/gradient behavior and zero-lambda equivalence to SiLU.
- CGCNN and ALIGNN-like forward paths with `readout_type="implicit_bias"`.
- Optional `GatedGraphConv` edge-weight shape handling and weighted aggregation behavior.
- CGCNN and ALIGNN-like `use_edge_weight` opt-in behavior.
- Training CLI `--use-edge-weight` propagation into model config.
- Checkpoint metadata round trip for implicit-bias CGCNN inference reconstruction.
- `predict_from_cif.py` smoke coverage for implicit-bias checkpoints loaded from metadata and from explicit legacy fallback CLI arguments.
- Run-comparison flattening, metric recomputation from `test_predictions.csv`, and sorted CSV output.
- AMP-safe `GatedGraphConv` dtype behavior through CPU autocast.
- Batch collation offsets for atom and line-graph indices.
- Optional `edge_weight`, `edge_unit_vec`, and `atom_attr` collation.
- Device resolution and recursive batch movement.
- Persistent graph cache round trip and cache invalidation when graph kwargs change.
- Neighbor strategy factory and custom strategy integration.
- Basic `pymatgen.Structure` graph construction when dependency is available.

### Important Test Gaps

- No real dataset training smoke test with actual CIF files and multiple epochs.
- No GPU/CUDA integration test.
- No full training smoke test proving timing fields on a real CIF dataset.
- No tests for Voronoi strategy against real structures in environments with `pymatgen`.
- No performance or memory regression tests for line-graph size.
- No CLI end-to-end tests for example scripts.
- No Matbench, uncertainty, classification, or multi-task tests.

### Manual Testing Notes

For new graph strategies or line-graph changes, manually inspect per-structure statistics before training:

- number of atoms
- number of directed edges
- number of line-graph edges
- max/min/mean distance
- max local coordination
- whether edge count changes with chosen strategy

Large `line_edge_index` is the most likely cause of CUDA OOM.

## 11. Deployment Notes

There is no production deployment setup.

Intended current environments:

- Local development machine.
- Google Colab.
- GPU workstation.
- HPC node or batch job, provided dependencies are installed.

### Hosting Platform

Not applicable.

### Build Process

Current development install:

```bash
pip install -e .
```

No Dockerfile, CI pipeline, or package publishing workflow exists yet.

### Required Secrets

None.

### Deployment Commands

Not applicable.

### Production Caveats

- This package is not yet production-hardened.
- There is no API server.
- There is no versioned model registry.
- There is no standardized experiment tracking.
- There is no distributed training launcher.
- Graph cache files are local and unmanaged.
- New checkpoints store full experiment configs and inference settings; legacy checkpoints still need CLI fallback arguments, including readout flags for implicit-bias models.

## 12. Next Recommended Steps

- [x] Save full experiment configuration with every training run and checkpoint, including model architecture, graph strategy, featurization settings, target column, split seed, and package version.
- [x] Add graph statistics utilities and an `examples/analyze_graph_dataset.py` script to report atom counts, edge counts, line-edge counts, distance ranges, and likely OOM risks before training.
- [x] Decide how `edge_weight` should enter message passing, then implement optional weighted aggregation in `GatedGraphConv` with tests.
- [x] Add a cache precomputation CLI so large datasets can build graphs before training instead of during the first epoch. Multiprocessing is available through `--num-workers`.
- [x] Add a run-comparison CLI for readout and parameter sweeps.
- [ ] Add attention or gated pooling as a configurable alternative to mean/sum pooling.
- [ ] Add Matbench-style dataset adapters and standardized split/evaluation workflows.
- [x] Add checkpoint round-trip and example-script smoke tests for implicit-bias prediction loading.
- [ ] Add basic classification and multi-task support across dataset, loss, metrics, and readout.
- [ ] Add first equivariant feature/model experiment using `edge_vec` or `edge_unit_vec`, while preserving current invariant CGCNN/ALIGNN-like baselines.
- [ ] Add CI with CPU tests, linting, and an optional `pymatgen` test environment.
