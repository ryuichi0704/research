# Numerical Experiments

This directory currently contains two self-contained oracle experiment runners for the `neurips_2026.tex` draft.

- `run_closed_form.py`
  - main rare-event and library-collapse validations from exact Beta-Bernoulli calculations
- `run_imbalance_oracle.py`
  - TabPFN-free imbalance validations for prompt balancing, prevalence recovery, and ranking reversal

Shared utilities live in:

- `core_closed_form.py`
- `core_imbalance.py`

## Setup

Install the standard dependencies:

```bash
pip install -r codes/requirements.txt
```

Both runners are fully self-contained once those packages are installed.

## Run The Main Closed-Form Experiments

From the repository root:

```bash
python codes/run_closed_form.py
```

Outputs are written to:

```text
codes/results/closed_form/
```

Expected artifacts:

- `figure1_global_hedge.png` and `.pdf`
- `figure2_boundary_failure_and_fix.png` and `.pdf`
- `figure3_phase_transition.png` and `.pdf`
- `summary_closed_form.csv`
- detailed per-experiment CSV files

Figure mapping:

- `figure1_global_hedge`
  - broad mixtures hedge OOD family shift
- `figure2_boundary_failure_and_fix`
  - collapsed broad mixtures fail near rare-event boundaries, and boundary-aware buckets repair the failure
- `figure3_phase_transition`
  - validates the phase-transition rate law by sweeping across interior, critical, and singular boundary regimes

## Run The Imbalance Oracle Experiments

From the repository root:

```bash
python codes/run_imbalance_oracle.py
```

Useful options:

```bash
python codes/run_imbalance_oracle.py \
  --c 0.5 \
  --d 49.5 \
  --support-sizes 8,16,32,64,128,256,512 \
  --noisy-token-sigma 1.0
```

Outputs are also written to:

```text
codes/results/closed_form/
```

Expected artifacts:

- `figure4_balancing_prevalence_curves.png` and `.pdf`
- `figure5_balancing_small_count.png` and `.pdf`
- `figure6_balancing_ranking_reversal.png` and `.pdf`
- `experiment5_balancing_prevalence_curves.csv`
- `experiment5_balancing_prevalence_calibration.csv`
- `experiment6_balancing_small_count.csv`
- `experiment6_balancing_small_count_excess.csv`
- `experiment7_balancing_ranking.csv`
- `summary_imbalance_oracle.csv`

Figure mapping:

- `figure4_balancing_prevalence_curves`
  - natural prompts improve with support size, balanced prompts stay prevalence-blind, and prevalence tokens recover the signal
- `figure5_balancing_small_count`
  - excess risk over the natural prompt is largest on rare small-count strata
- `figure6_balancing_ranking_reversal`
  - the exact case-control aliasing counterexample from the imbalance section, with a visible ranking reversal under family `A`

## Notes

- `run_imbalance_oracle.py` records prevalence RMSE and prevalence-calibration tables because ordinary outcome calibration is largely trivial for these oracle predictors.
- All outputs are deterministic except for the noisy-token Monte Carlo terms, which are controlled by the script seed.
