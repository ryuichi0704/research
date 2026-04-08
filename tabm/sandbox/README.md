# Sandbox Outputs (2026-04-05)

## Main theory note

- `contrast_mode_theory.tex`
- `contrast_mode_theory.pdf`

Standalone note containing:

- consensus/contrast linear decomposition around the collapsed BatchEnsemble baseline;
- covariance formula for leading-order disagreement;
- second-order invariance of ensemble risk under centered contrast perturbations;
- `1/M` no-go for CLT-scale asymmetry;
- optimal antithetic/rank-one contrast design theorem;
- practitioner consequences.

## Finite-K geometry note

- `finite_k_geometry_note.tex`
- `finite_k_geometry_note.pdf`

Standalone note on why diversity amplitude alone is not the right object for finite `K`, including:

- exact amplitude = mean-drift + disagreement decomposition;
- pairwise-correlation formula;
- distinction between trace-optimal rank-one diversity and robust multi-direction coverage;
- tetrahedral principle for `K=4`;
- exact simplex realization of any target rank-`K-1` local covariance.

## Stationary-law design note

- `stationary_law_design_note.tex`
- `stationary_law_design_note.pdf`

Note explaining why initialization-only optimality is generally transient under nonlinear training, and how to redesign diversity from the converged collapsed law.

## Boundary fast-weight persistence note

- `boundary_fast_weight_persistence_note.tex`
- `boundary_fast_weight_persistence_note.pdf`

Note clarifying that differing boundary fast weights do not universally disappear in the mean-field limit; disappearance requires a contraction condition.

## Paper-ready insertion

- `paper_insert_contrast_modes.tex`

Compact subsection written in the notation of `paper/neurips_2026.tex`.

## Full paper preview in sandbox

- `neurips_2026_with_contrast_extension.tex`
- `neurips_2026_with_contrast_extension.pdf`

Sandbox-only preview of the current paper with the new contrast-mode subsection inserted.

## Syntax-check wrapper for the insertion

- `contrast_insert_wrapper.tex`
- `contrast_insert_wrapper.pdf`

Only for local compilation/sanity checking of the subsection fragment.

## Literature positioning

- `literature_notes_2026-04-05.md`

Short memo on what was checked on the web and how the new results differ from the closest prior theory.

## Verification memo

- `diversity_extension_verification_2026-04-05.md`

Theorem-by-theorem review of the proposed contrast/diversity extension, including what is correct as stated and what needs caveats.

## Practitioner guidance

- `practitioner_recipe.md`
- `practical_protocol_finite_k.md`

Implementation-oriented recipes for contrast-aware BatchEnsemble initialization and evaluation.

## Next theorem targets

- `next_theory_directions.md`

Concrete follow-up theorem candidates after the current local contrast-mode analysis.
