# Next Theory Directions After the Current Contrast-Mode Results

The current sandbox note gives a local theory around the collapsed diagonal. The strongest next results to target are:

## 1. Nonlinear escape theorem

Goal:

- go beyond linearized contrast growth;
- prove conditions under which a centered contrast perturbation leaves an `O(eps)` neighborhood of the collapse diagonal within finite time;
- characterize whether the resulting nonlinear trajectory still preserves a small shift in the ensemble mean.

Why this matters:

- the current note explains the *first-order* design of diversity;
- a nonlinear escape theorem would explain when that diversity becomes genuinely order-one.

## 2. Finite-width fluctuation theorem

Goal:

- connect the `1/M` disagreement prediction to actual finite-width BatchEnsemble;
- show that, absent explicit order-one contrast initialization, wide finite models inherit only vanishing spread on fixed horizons;
- identify the precise finite-width scaling assumptions under which this holds.

Why this matters:

- this would turn the current local no-go principle into a concrete asymptotic statement about practical networks.

## 3. Contrast-instability criterion

Goal:

- characterize the sign and size of the top singular/eigen growth of the contrast transfer operator;
- relate that growth to optimization choices such as learning rate, regularization, or architecture.

Why this matters:

- practitioners need to know not only how to initialize contrast, but also how to keep it from collapsing during training.

## 4. Data-averaged design theorem with estimation error

Goal:

- turn the integrated Gram matrix design rule into a finite-sample theorem;
- quantify how much disagreement is lost when the dominant eigendirection is estimated from minibatches or a validation subset.

Why this matters:

- this is the missing bridge between the current exact optimization result and a deployable initialization algorithm.

## 5. Contrast-subspace training theorem

Goal:

- parameterize member-specific boundary variables in a low-dimensional zero-mean contrast subspace;
- prove that this preserves the first-order mean-invariance property while concentrating diversity budget on useful directions.

Why this matters:

- this would convert the current analysis from an initialization prescription into an architecture/training prescription.
