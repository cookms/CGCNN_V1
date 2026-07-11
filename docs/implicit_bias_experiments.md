# Implicit-Bias Experiments

Run these commands from the repository root. Replace `data/id_prop.csv` and `target`
with the dataset path and target column for the study. Keep the same data, seed, graph
settings, and training settings across runs so the convolution activation is the only
experimental variable.

## Experiment 2: convolution hidden activations

Experiment 2 applies the optional implicit-bias activation independently across hidden
feature channels for each edge or node sample inside `GatedGraphConv`. Graph construction,
pooling, the default MLP readout, and the edge gate remain unchanged. The gate is still:

```text
Linear(edge_dim, node_dim) -> Sigmoid
```

The `conv_`-prefixed flags are independent of the experiment-1 readout flags. These
commands explicitly keep `--readout-type mlp` to isolate the convolution experiment.

## CGCNN Target Isolation

### 1. Baseline: normal SiLU

```bash
python examples/train_cgcnn.py \
  --csv data/id_prop.csv \
  --target target \
  --seed 42 \
  --readout-type mlp \
  --conv-activation-type silu \
  --output-dir runs/cgcnn_baseline
```

### 2. Implicit bias in `node_mlp` only

```bash
python examples/train_cgcnn.py \
  --csv data/id_prop.csv \
  --target target \
  --seed 42 \
  --readout-type mlp \
  --conv-activation-type implicit_bias \
  --conv-ib-targets node \
  --conv-ib-lambda 0.01 \
  --conv-ib-sigma-slope 1.0 \
  --conv-ib-fixed-point-iters 8 \
  --conv-ib-coupling ring \
  --output-dir runs/cgcnn_conv_ib_node_lam001
```

### 3. Implicit bias in `message_mlp` only

```bash
python examples/train_cgcnn.py \
  --csv data/id_prop.csv \
  --target target \
  --seed 42 \
  --readout-type mlp \
  --conv-activation-type implicit_bias \
  --conv-ib-targets message \
  --conv-ib-lambda 0.01 \
  --conv-ib-sigma-slope 1.0 \
  --conv-ib-fixed-point-iters 8 \
  --conv-ib-coupling ring \
  --output-dir runs/cgcnn_conv_ib_message_lam001
```

### 4. Implicit bias in `edge_mlp` only

```bash
python examples/train_cgcnn.py \
  --csv data/id_prop.csv \
  --target target \
  --seed 42 \
  --readout-type mlp \
  --conv-activation-type implicit_bias \
  --conv-ib-targets edge \
  --conv-ib-lambda 0.01 \
  --conv-ib-sigma-slope 1.0 \
  --conv-ib-fixed-point-iters 8 \
  --conv-ib-coupling ring \
  --output-dir runs/cgcnn_conv_ib_edge_lam001
```

### 5. Implicit bias in all three MLPs

```bash
python examples/train_cgcnn.py \
  --csv data/id_prop.csv \
  --target target \
  --seed 42 \
  --readout-type mlp \
  --conv-activation-type implicit_bias \
  --conv-ib-targets all \
  --conv-ib-lambda 0.01 \
  --conv-ib-sigma-slope 1.0 \
  --conv-ib-fixed-point-iters 8 \
  --conv-ib-coupling ring \
  --output-dir runs/cgcnn_conv_ib_all_lam001
```

## Small All-Target Sweep

After the target-isolation runs, this Bash loop executes the full 4 x 3 x 2 sweep
sequentially with implicit bias enabled in all three convolution MLPs:

```bash
for conv_ib_lambda in 0.001 0.003 0.01 0.03; do
  for conv_ib_fixed_point_iters in 3 5 8; do
    for conv_ib_coupling in ring dense; do
      lambda_tag=${conv_ib_lambda/./p}
      run_name="cgcnn_conv_ib_all_lam${lambda_tag}_iters${conv_ib_fixed_point_iters}_${conv_ib_coupling}"
      python examples/train_cgcnn.py \
        --csv data/id_prop.csv \
        --target target \
        --seed 42 \
        --readout-type mlp \
        --conv-activation-type implicit_bias \
        --conv-ib-targets all \
        --conv-ib-lambda "$conv_ib_lambda" \
        --conv-ib-sigma-slope 1.0 \
        --conv-ib-fixed-point-iters "$conv_ib_fixed_point_iters" \
        --conv-ib-coupling "$conv_ib_coupling" \
        --output-dir "runs/${run_name}"
    done
  done
done
```

Sweep values:

```text
conv_ib_lambda in [0.001, 0.003, 0.01, 0.03]
conv_ib_fixed_point_iters in [3, 5, 8]
conv_ib_coupling in ["ring", "dense"]
```

## ALIGNN-Like Check

The same convolution settings apply to both line-graph and bond-graph convolutions.

Baseline:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --seed 42 \
  --readout-type mlp \
  --conv-activation-type silu \
  --output-dir runs/alignn_baseline
```

All convolution targets:

```bash
python examples/train_alignn_like.py \
  --csv data/id_prop.csv \
  --target target \
  --seed 42 \
  --readout-type mlp \
  --conv-activation-type implicit_bias \
  --conv-ib-targets all \
  --conv-ib-lambda 0.01 \
  --conv-ib-sigma-slope 1.0 \
  --conv-ib-fixed-point-iters 8 \
  --conv-ib-coupling ring \
  --output-dir runs/alignn_conv_ib_all_lam001
```

## Measurements

Track validation MAE, validation RMSE, validation R2, training time per epoch,
stability or NaNs, and whether validation improves earlier than the baseline.
