# Numerical Experiments

This directory contains two experiment layers for the `neurips_2026.tex` draft.

- `run_closed_form.py`
  - main-text experiments using exact Bayes predictors in Beta-Bernoulli toy worlds
- `run_learned_tfm.py`
  - appendix corroboration using `NanoTabPFNModel` from [TFM-Playground](https://github.com/automl/TFM-Playground/)
- `test_closed_form.py`
  - lightweight regression checks for the closed-form utilities

## Setup

Install the standard dependencies:

```bash
pip install -r codes/requirements.txt
```

The closed-form experiments are fully self-contained once those packages are installed.

## Run The Main Closed-Form Experiments

From the repository root:

```bash
python codes/test_closed_form.py
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

### Figure mapping

- `figure1_global_hedge`
  - broad mixtures hedge OOD family shift
- `figure2_boundary_failure_and_fix`
  - collapsed broad mixtures fail near rare-event boundaries, and boundary-aware buckets repair the failure
- `figure3_phase_transition`
  - validates the phase-transition rate law by sweeping across interior, critical, and singular boundary regimes

## Run The Optional Learned Corroboration

The learned appendix experiment uses TFM-Playground as an optional backend.

Official repository:

- [automl/TFM-Playground](https://github.com/automl/TFM-Playground/)

Install it separately, for example:

```bash
git clone https://github.com/automl/TFM-Playground.git
cd TFM-Playground
pip install -e .
cd ..
```

Then run:

```bash
python codes/run_learned_tfm.py
```

Outputs are written to:

```text
codes/results/learned_tfm/
```

Expected artifacts:

- `figure_appendix_learned_boundary_corrob.png` and `.pdf`
- `learned_training_log.csv`
- `learned_boundary_logloss.csv`
- `learned_boundary_profiles.csv`
- saved checkpoints for the two trained mixtures

The learned script is designed as appendix corroboration. The main paper should still rely on the exact closed-form experiments for its primary evidence.

## How The Figures Validate The Theory

- Figures 1 and 2 validate the main design claims:
  - broad mixing improves worst-case OOD robustness
  - collapsed mixtures fail on boundary-heavy deployments
  - boundary-aware bucketing fixes that failure
- Figure 3 validates the rate-level phase transition:
  - the left panel shows the slope change across `c > 1`, `c = 1`, and `0 < c < 1`
  - the right panel shows the task-level prevalence priors whose boundary mass drives the phase transition
