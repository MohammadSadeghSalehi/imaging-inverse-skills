---
name: imaging-optimisation
description: >
  Choose a convergent first-order algorithm for a variational imaging
  objective and set its step sizes: proximal gradient, FISTA, ADMM,
  Chambolle-Pock primal-dual (PDHG), stochastic and adaptive PDHG,
  preconditioning, total variation, total generalised variation, and
  bilevel parameter learning. Use when the user asks for optimisation or
  optimization of an imaging or inverse-problem objective, a proximal map,
  a step size, PDHG, SPDHG, plug-and-play convergence, RED, or runs
  /imaging-optimisation.
---

# Imaging optimisation

Fit a solver to an imaging objective. The regulariser and the loss are chosen by `../inverse-problems/SKILL.md`. The operator and the noise are chosen by `../computational-imaging/SKILL.md`. Step-size theorems, the algorithm table, and the deepinv class for each algorithm are in `references/algorithms.md`. Citations are in `../inverse-problems/references/lineage.md` and `../inverse-problems/references/communities.md`.

This skill covers continuous optimisation for imaging, in the sense of Chambolle and Pock's Acta Numerica introduction (2016) and the primal-dual algorithm (JMIV, 2011). It is the wrong tool for training-hyperparameter search, combinatorial solvers, and generic neural-network optimisers, except where those appear as the outer loop of a bilevel imaging model.

## Procedure

1. Write `min_x f(x; y) + λ R(x)` with `f(x; y) = d(A(x), y)`. Record convexity, smoothness, and whether a prox is available, for `f` and for `R` separately.
2. If both terms are nonsmooth, or a linear map sits inside a nonsmooth fidelity, rewrite as the saddle point `min_x max_u ⟨Kx, u⟩ + G(x) - F*(u)` and identify `K`.
3. Pick the row of the algorithm table in `references/algorithms.md`. In deepinv the Chambolle-Pock iteration is `deepinv.optim.PDCP`.
4. Estimate `‖A‖²` or `‖K‖²` and set the step from that table. For a deepinv physics operator use `physics.compute_norm`.
5. Log the quantity the proof controls: the objective when it is explicit, the primal-dual gap for a saddle point. Stop on that quantity, and treat the change in the image as a secondary check.

## Rules that decide the algorithm

- Smooth data term, prox-friendly regulariser: proximal gradient, step `γ < 1/L` with `L = ‖A‖²` for a squared L2 fidelity.
- Same structure, and the user asked for acceleration: FISTA, same step restriction. Quote an `O(1/k²)` rate on function values. Quote iterate convergence only with the Chambolle-Dossal momentum (`a > 2`).
- Two proxes and a linear map: Chambolle-Pock (`PDCP`), extrapolation `beta=1`. The product condition on the two steps is the PDHG row of the algorithms reference. Log the product you used.
- The linear map is a sum of many blocks (projection angles, PET subsets): stochastic PDHG (Chambolle, Ehrhardt, Richtárik, Schönlieb, 2018), with the block probabilities inside the step-size condition. Use adaptive SPDHG (Chambolle, Delplancke, Ehrhardt, Schönlieb, Tang, JMIV 2024) when the primal/dual ratio is the part you cannot set.
- Rows of `K` have very different norms: diagonal preconditioning (Pock and Chambolle) instead of a single scalar step.
- Squared fidelity plus TV only: Chambolle's 2004 dual projection, with the step bound in the algorithms reference. General `A`: proximal gradient on `TVPrior`, or `PDCP` with `K` equal to `A`.
- TV staircasing on ramps: move to second-order TGV and solve the saddle point. The definition is in the algorithms reference.
- Poisson counts and a nonnegative image: `MLEM`, `OSEM`, or `BSREM`, or a Bregman geometry (`BurgEntropy`) rather than an L2 gradient step on the raw counts.
- Nonconvex smooth term plus a prox: iPiano, inside the paper's step and inertia bounds (Ochs, Chen, Brox, Pock).
- Weakly convex learned regulariser: stay inside the modulus assumed by Goujon, Neumayer, and Unser, or by Shumaylov, Budd, Mukherjee, and Schönlieb for PDHG. The objective is then handled as weakly convex, and the claim is critical-point convergence.
- The data term is a long sum and a denoiser replaces the prox: one subset of measurements per data step (online plug-and-play). The convergence claim is still one of the three hypotheses in `../inverse-problems/references/communities.md`.
- Landweber, iterated Tikhonov, or a deep image prior: these are semi-convergent. Stop when the residual reaches the noise level. Running to a tiny residual fits noise. The discrepancy principle is in the Italian section of `../inverse-problems/references/communities.md`.

Plug-and-play keeps the data-term step of proximal gradient, HQS, ADMM, or Douglas-Rachford and replaces the prox. RED replaces the prior gradient by `x - D(x)`. Quote a convergence or "this is a prior gradient" claim only under the matching hypothesis in `../inverse-problems/references/communities.md`. Langevin and diffusion samplers are not rows of this skill. A total-variation density inside a sampler is not discretization-invariant; deterministic `TVPrior` minimisation is a different estimator.

## Bilevel parameters

`λ`, a Fields-of-Experts filter bank, or a sampling mask is an outer variable. The inner problem stays the variational reconstruction. Implement the derivative by unrolling (`unfold=True`, `trainable_params` in deepinv) or by the implicit-function route (`DEQConfig` on `GD`, `PGD`, or `HQS`). A variational network is this construction with learned filters and learned activations in each gradient step.

## Implementation

```python
import deepinv as dinv

norm_A2 = physics.compute_norm(y, tol=1e-4, verbose=False).item()
model = dinv.optim.PGD(
    prior=dinv.optim.TVPrior(),
    data_fidelity=dinv.optim.L2(),
    stepsize=0.99 / norm_A2,
    lambda_reg=0.05,
    max_iter=200,
)
x_hat = model(y, physics, init=physics.A_dagger(y))
```

Swap the class using the table in the reference when the structure is not proximal-gradient. For `PDCP`, pass `K` and `K_adjoint` (the default is the identity), set `stepsize` and `stepsize_dual`, keep `beta=1`, and check the product condition from the algorithms reference on the numbers you actually run. Backtracking (`BacktrackingConfig`) is available when the prior has an explicit cost; it is disabled for denoiser priors.

Parameters may be a scalar or a list with one value per iteration, which is how a continuation schedule on `λ` or on a denoiser noise level is written.

Confirm class names against the installed deepinv. The reference is the 0.4.2 surface.
