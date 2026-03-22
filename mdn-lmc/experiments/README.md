# `k1.tex` numerical experiment

`paper/k1.tex` の single-Gaussian natural-softplus head を、1 次元 synthetic dataset 上で確認するための実験です。

実装しているもの:

- 2 層 NN
  - 1 層目: `Linear(1, width, bias=True)`
  - 活性化: `sigmoid`
  - 2 層目: `Linear(width, 2, bias=False)`
  - 出力は mean-field 風に `1 / width` でスケーリング
- パラメータ化の切り替え
  - `natural`: `k1.tex` に沿って `(u, s_raw) -> lambda=softplus(s_raw)+1/(2*variance_max), eta2=-lambda`
  - `meanvar`: `(mu, s_raw) -> (mu, var=softplus(s_raw)+variance_min)`
- 学習後のパラメータ可視化
  - 縦軸: 1 層目 weight
  - 横軸: 2 層目 weight の mean component / variance component
  - 色: 1 層目 bias
- LMC barrier の計測
  - 2 つの独立学習モデルを線形補間
  - `max_t [L(theta_t) - ((1-t)L(theta_A) + tL(theta_B))]` を計測
- マッチング
  - 1 隠れ層では hidden neuron permutation が 1 回の線形割当問題に落ちるので、
    `Git Re-Basin` の weight matching を簡略化した形で Hungarian assignment を使っています
  - コストは neuron tuple `(a_i, c_i, w_i, b_i)` の Euclidean 距離です
- `k1.tex` の量の計測
  - `B_N`
  - `Delta_{s,N}`
  - `M_V^infty`
  - `natural` モードでは理論バウンドの右辺も計算

## セットアップ

```bash
cd experiments
uv sync
```

## 実行

YAML 設定ファイルから実行:

```bash
uv run python main.py --config configs/example.yaml
```

YAML をベースに一部だけ CLI で上書き:

```bash
uv run python main.py --config configs/example.yaml --epochs 800 --output-dir results_override
```

YAML は階層化していて、訓練・評価・可視化の設定を分けています。`--config` を使った場合も、明示的に CLI で渡した値が優先されます。

`natural` だけ:

```bash
uv run python main.py --parameterization natural
```

`meanvar` だけ:

```bash
uv run python main.py --parameterization meanvar
```

両方まとめて:

```bash
uv run python main.py --parameterization both
```

幅 sweep で、`2^8` から `2^13` までの width に対して

- baseline の `k1.tex` bound
- best theorem-consistent exact-modulus bound
- 実測 matched barrier

を比較するには:

```bash
uv run python width_sweep_experiment.py \
  --config configs/config.yaml \
  --output-dir results_width_sweep \
  --width-exponents 8 9 10 11 12 13 \
  --time-grid-points 401
```

この sweep は `natural` parameterization を前提にしています。

## 出力

デフォルトでは `experiments/results/<parameterization>/` に以下を保存します。

- `summary.json`: 指標まとめ
- `model_a_parameters.png`: モデル A の neuron 分布図
- `model_b_matched_parameters.png`: マッチング後モデル B の neuron 分布図
- `lmc_barrier.png`: 補間損失と barrier
- `predictive_fit.png`: 真の平均・分散との比較
- `training_history.png`: 学習曲線
- `matching_permutation.npy`: 推定 permutation

幅 sweep では `experiments/results_width_sweep/` 以下に

- `width_sweep_summary.json`: 幅ごとの集約結果
- `width_sweep_primary_vs_observed.png`: 実測 barrier と主たる理論 bound の比較図
- `width_<N>/summary.json`: 各幅の詳細

を保存します。

`--parameterization both` の場合は `results/comparison.json` も保存します。

## 主要オプション

```bash
uv run python main.py \
  --parameterization natural \
  --width 128 \
  --epochs 800 \
  --learning-rate 5e-3 \
  --evaluation-probe-points 4096 \
  --plot-points 1024 \
  --barrier-points 41
```

必要に応じて `--device cpu` か `--device cuda` を指定できます。

## YAML キー

推奨構成は以下です。

```yaml
experiment:
  parameterization: both
  output_dir: results_from_yaml
  device: cpu

model:
  width: 128
  variance_min: 0.05
  variance_max: 10.0

data:
  train_size: 1024
  val_size: 512
  test_size: 2048
  x_max: 3.0

train:
  epochs: 400
  batch_size: 256
  learning_rate: 0.01
  weight_decay: 0.000001
  patience: 60

evaluation:
  barrier_points: 41
  probe_points: 2048

visualization:
  plot_points: 2048

seeds:
  data: 2026
  model_a: 0
  model_b: 1
```

意味としては:

- `model.variance_min`: `meanvar` 側の分散下限
- `model.variance_max`: `natural` 側で `lambda_min = 1 / (2 * variance_max)` に変換して使う共有スケール
- `evaluation.probe_points`: `Delta_{s,N}` などの評価用グリッド点数
- `visualization.plot_points`: 予測曲線の描画用グリッド点数
