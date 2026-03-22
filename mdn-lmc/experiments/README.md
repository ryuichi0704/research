# `k1_raw_lmc.tex` numerical experiment

`paper/k1_raw_lmc.tex` の single-Gaussian mean/variance head を、`research_tmp` の K=1 synthetic 設定に寄せて検証する実験です。

実装しているもの:

- 2 層 NN
  - 1 層目: `Linear(1, width, bias=True)` + `sigmoid`
  - 2 層目: `Linear(width, 2, bias=False)`
  - 出力は mean-field 風に `1 / width` でスケーリング
- synthetic dataset
  - `x ~ Unif[-1.05, 1.05]`
  - `y = mu_*(x) + sigma_*(x) epsilon`
  - `epsilon ~ N(0, 1)`
  - `dataset_pattern` で `mu_*, sigma_*` を切り替え可能
- パラメータ化
  - `meanvar`: `(mu, s_raw) -> (mu, var=exp(s_raw)+variance_min)` を既定に使用
  - `natural`: 比較用に維持
- 学習設定
  - `Adam`
  - cosine LR (`learning_rate -> learning_rate_min`)
  - backward は `loss * width` で mean-field scaling に寄せる
- LMC barrier の計測
  - 独立学習した 2 モデルを線形補間
  - hidden permutation は Hungarian matching
  - width sweep では複数 seed-pair を集約

## セットアップ

```bash
cd experiments
uv sync
```

repo root からまとめて再生成する場合:

```bash
make reproduce-patterns
```

この target は `experiments/results` を一度空にしてから、5 つの `dataset_pattern` について

- 小幅 preview (`results/pattern_small_cases/<pattern>/`)
- width sweep (`results/pattern_sweeps/<pattern>/`)

を順に実行します。

## 単発実験

デフォルトでは `meanvar + exp` の synthetic 実験を回します。

```bash
uv run python main.py --config configs/config.yaml
```

一部だけ上書き:

```bash
uv run python main.py \
  --parameterization meanvar \
  --width 1024 \
  --epochs 10000 \
  --learning-rate 0.1 \
  --learning-rate-min 0.001 \
  --barrier-points 41
```

比較用に `natural` を回す場合:

```bash
uv run python main.py --parameterization natural
```

分散構造を変えたい場合:

```bash
uv run python main.py --config configs/config.yaml --dataset-pattern hetero_sigmoid
uv run python main.py --config configs/config.yaml --dataset-pattern hetero_periodic
uv run python main.py --config configs/config.yaml --dataset-pattern hetero_edges
uv run python main.py --config configs/config.yaml --dataset-pattern hetero_bumps
```

repo root の `Makefile` には個別 target もあります。

```bash
make preview-one PATTERN=hetero_edges
make sweep-one PATTERN=hetero_edges
make preview-all
make sweep-all
```

## 幅 sweep

`research_tmp` 側に合わせて、複数 seed-pair をまとめて可視化します。

```bash
uv run python width_sweep_experiment.py \
  --config configs/config.yaml \
  --output-dir results_width_sweep \
  --width-exponents 9 10 11 12 \
  --n-seeds 10 \
  --time-grid-points 401 \
  --max-parallel-workers 4
```

この sweep はデフォルトで seed `0..9` を学習し、pair `(0,1), (2,3), ...` を評価します。
`--max-parallel-workers` を増やすと、各 seed-pair 実験を独立 worker として並列実行します。

## 出力

単発実験では `experiments/results/<parameterization>/` に以下を保存します。

- `summary.json`
- `model_a_parameters.png`
- `model_b_matched_parameters.png`
- `lmc_barrier.png`
- `predictive_fit.png`
- `training_history.png`
- `dataset_preview.png`
- `matching_permutation.npy`

幅 sweep では `experiments/results_width_sweep/` 以下に以下を保存します。

- `width_sweep_summary.json`
- `loss_barriers.png`: width ごとの naive/aligned 補間損失
- `barrier_scaling.png`: seed-pair 平均 barrier の width scaling
- `loss_curves.png`: 各 width・seed の学習曲線
- `width_sweep_primary_vs_observed.png`: 実測 barrier と主たる理論 bound の比較
- `width_<N>/summary.json`: 各幅の pair-level 詳細
- `width_<N>/pair_<a>_<b>/summary.json`: 各 seed-pair の詳細
- `width_<N>/pair_<a>_<b>/model_a_parameters.png`
- `width_<N>/pair_<a>_<b>/model_b_matched_parameters.png`
- `width_<N>/pair_<a>_<b>/lmc_barrier.png`
- `width_<N>/pair_<a>_<b>/predictive_fit.png`
- `width_<N>/pair_<a>_<b>/training_history.png`

## YAML キー

現在の推奨構成は以下です。

```yaml
experiment:
  parameterization: meanvar
  output_dir: results
  device: cuda

model:
  width: 1024
  variance_min: 0.001
  variance_max: 10.0
  precision_activation: exp

data:
  dataset_pattern: homoskedastic_wave
  train_size: 1000
  test_size: 1000
  x_max: 1.05

train:
  epochs: 10000
  batch_size: 1000
  learning_rate: 0.1
  learning_rate_min: 0.001
  weight_decay: 0.0

evaluation:
  barrier_points: 25
  probe_points: 2048

visualization:
  plot_points: 2048

seeds:
  data: 2026
  model_a: 0
  model_b: 1
```

主な意味:

- `data.dataset_pattern`: synthetic データの mean/std パターン
- `model.precision_activation`: `meanvar` 側では `exp` を既定にして `research_tmp` の K=1 synthetic 設定へ寄せる
- `train.learning_rate_min`: cosine schedule の終点
- `evaluation.probe_points`: `Delta_raw_N` や exact modulus の評価用グリッド
- `visualization.plot_points`: 予測曲線の描画グリッド

利用できる `dataset_pattern`:

- `homoskedastic_wave`: 現在の基準設定。波状の平均に対して標準偏差は一定 `0.1`
- `hetero_sigmoid`: 左から右へ標準偏差が単調に大きくなる
- `hetero_periodic`: 平均も分散も周期的に揺れる
- `hetero_edges`: 中央は低ノイズ、両端で高ノイズ
- `hetero_bumps`: 局所的に分散が大きくなるバンプを 2 箇所に持つ
