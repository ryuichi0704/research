SHELL := /bin/bash

EXPERIMENTS_DIR := experiments
CONFIG := configs/config.yaml
UV_RUN := uv run python
PARAMETERIZATION ?= meanvar
DEVICE ?= cuda
PATTERN ?= homoskedastic_wave
PATTERNS := homoskedastic_wave hetero_sigmoid hetero_periodic hetero_edges hetero_bumps

PREVIEW_WIDTH ?= 128
PREVIEW_EPOCHS ?= 200

WIDTH_EXPONENTS ?= 9 10 11 12 13
N_SEEDS ?= 10
TIME_GRID_POINTS ?= 401
MAX_PARALLEL_WORKERS ?= 4

.PHONY: help clean-results preview-one preview-all sweep-one sweep-all reproduce-patterns

help:
	@echo "Targets:"
	@echo "  make clean-results"
	@echo "  make preview-one PATTERN=<dataset_pattern>"
	@echo "  make preview-all"
	@echo "  make sweep-one PATTERN=<dataset_pattern>"
	@echo "  make sweep-all"
	@echo "  make reproduce-patterns"

clean-results:
	cd $(EXPERIMENTS_DIR) && python -c 'from pathlib import Path; import shutil; root = Path("results"); shutil.rmtree(root, ignore_errors=True); root.mkdir(parents=True, exist_ok=True); print(f"Reset {root.resolve()}")'

preview-one:
	cd $(EXPERIMENTS_DIR) && $(UV_RUN) main.py \
		--config $(CONFIG) \
		--output-dir results/pattern_small_cases/$(PATTERN) \
		--parameterization $(PARAMETERIZATION) \
		--dataset-pattern $(PATTERN) \
		--width $(PREVIEW_WIDTH) \
		--epochs $(PREVIEW_EPOCHS) \
		--device $(DEVICE)

preview-all:
	cd $(EXPERIMENTS_DIR) && for pattern in $(PATTERNS); do \
		$(UV_RUN) main.py \
			--config $(CONFIG) \
			--output-dir results/pattern_small_cases/$$pattern \
			--parameterization $(PARAMETERIZATION) \
			--dataset-pattern $$pattern \
			--width $(PREVIEW_WIDTH) \
			--epochs $(PREVIEW_EPOCHS) \
			--device $(DEVICE) || exit $$?; \
	done

sweep-one:
	cd $(EXPERIMENTS_DIR) && $(UV_RUN) width_sweep_experiment.py \
		--config $(CONFIG) \
		--output-dir results/pattern_sweeps/$(PATTERN) \
		--parameterization $(PARAMETERIZATION) \
		--dataset-pattern $(PATTERN) \
		--width-exponents $(WIDTH_EXPONENTS) \
		--n-seeds $(N_SEEDS) \
		--time-grid-points $(TIME_GRID_POINTS) \
		--max-parallel-workers $(MAX_PARALLEL_WORKERS) \
		--device $(DEVICE)

sweep-all:
	cd $(EXPERIMENTS_DIR) && for pattern in $(PATTERNS); do \
		$(UV_RUN) width_sweep_experiment.py \
			--config $(CONFIG) \
			--output-dir results/pattern_sweeps/$$pattern \
			--parameterization $(PARAMETERIZATION) \
			--dataset-pattern $$pattern \
			--width-exponents $(WIDTH_EXPONENTS) \
			--n-seeds $(N_SEEDS) \
			--time-grid-points $(TIME_GRID_POINTS) \
			--max-parallel-workers $(MAX_PARALLEL_WORKERS) \
			--device $(DEVICE) || exit $$?; \
	done