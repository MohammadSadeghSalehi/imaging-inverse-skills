---
name: imaging-optimisation
description: >
  Choose a convergent first-order solver for a variational imaging
  objective, set its step sizes from a stated condition, and decide what
  convergence claim is allowed: proximal gradient, FISTA, ADMM, HQS,
  Chambolle-Pock PDHG, stochastic PDHG, preconditioning, total variation
  and TGV, EM for Poisson counts, early stopping, and bilevel learning of
  regularisation parameters. Use when the user asks how to optimise or
  solve an imaging objective, which step size to use, why an iteration
  diverges or stalls, how to wire PDHG, whether plug-and-play or RED
  converges, or how to learn lambda from data.
---

# Imaging optimisation

Fit a solver to an imaging objective. The regulariser and the loss are chosen by `../inverse-problems/SKILL.md`. The operator and the noise are chosen by `../computational-imaging/SKILL.md`. Step-size theorems, the algorithm table, and the deepinv class for each algorithm are in `references/algorithms.md`. Citations are in `../inverse-problems/references/lineage.md` and `../inverse-problems/references/communities.md`.

This skill covers continuous optimisation for imaging, in the sense of Chambolle and Pock's Acta Numerica introduction (2016) and the primal-dual algorithm (JMIV, 2011). It is the wrong tool for training-hyperparameter search, combinatorial solvers, and generic neural-network optimisers, except where those appear as the outer loop of a bilevel imaging model.

## Procedure

1. Write `min_x f(x; y) + λ R(x)` with `f(x; y) = d(A(x), y)`. Record convexity, smoothness, and whether a prox is available, for `f` and for `R` separately.
2. If both terms are nonsmooth, or a linear map sits inside a nonsmooth fidelity, rewrite as the saddle point `min_x max_u ⟨Kx, u⟩ + G(x) - F*(u)` and identify `K`.
3. Pick the row of the algorithm table in `references/algorithms.md`. In deepinv the Chambolle-Pock iteration is `deepinv.optim.PDCP`.
4. Estimate `‖A‖²` or `‖K‖²` and set the step from that table. For a deepinv physics operator use `physics.compute_sqnorm(x)` with a batched tensor shaped like the image, not the measurement, and keep a small margin because the power iteration undershoots.
5. Log the quantity the proof controls: the objective when it is explicit, the primal-dual gap for a saddle point. Stop on that quantity, and treat the change in the image as a secondary check.

## Rules that decide the algorithm

- Smooth data term, prox-friendly regulariser: proximal gradient, step `0 < γ < 2/L` with `L = ‖A‖²` for a squared L2 fidelity. `γ = 1/L` is the safe default.
- Same structure, and the user asked for acceleration: FISTA, with the stricter step `γ ≤ 1/L`. Quote an `O(1/k²)` rate on function values. Quote iterate convergence only with the Chambolle-Dossal momentum (`a > 2`).
- Two proxes and a linear map: Chambolle-Pock (`PDCP`), extrapolation `beta=1`. `PDCP` does not check the step product; assert `τ σ ‖K‖² < 1` yourself. The data term of `PDCP` already contains the `physics` it is called with, so `K=physics.A` must be called with an identity physics, or `A` is applied twice. The wiring is in the algorithms reference.
- The linear map is a sum of many blocks (projection angles, PET subsets): stochastic PDHG (Chambolle, Ehrhardt, Richtárik, Schönlieb, 2018), with the block probabilities inside the step-size condition. Use adaptive SPDHG (Chambolle, Delplancke, Ehrhardt, Schönlieb, Tang, JMIV 2024) when the primal/dual ratio is the part you cannot set.
- Rows of `K` have very different norms: diagonal preconditioning (Pock and Chambolle) instead of a single scalar step.
- Squared fidelity plus TV only: Chambolle's 2004 dual projection, with the step bound in the algorithms reference. General `A`: proximal gradient on `TVPrior`, or `PDCP` with `K` equal to `A`.
- TV staircasing on ramps: move to second-order TGV and solve the saddle point. The definition is in the algorithms reference.
- Poisson counts and a nonnegative image: `MLEM` or `OSEM`, or a Bregman geometry (`BurgEntropy`, mirror descent `MD`/`PMD`) rather than an L2 gradient step on the raw counts. A `prior` passed to `MLEM` or `OSEM` gives Green's one-step-late MAP-EM, a heuristic without a convergence proof. For a penalised Poisson solve you need to call convergent, use a primal-dual method on the Kullback–Leibler term (CIL's `KullbackLeibler` has the conjugate prox) with a nonnegativity constraint.
- The unknown is a PDE coefficient: penalty formulation in the PDE-constrained section of `../inverse-problems/references/communities.md`. The reduced adjoint-state objective is the one with the local minima. Subset tomography with an explicit convex objective is CIL's `SPDHG`, under the stochastic PDHG step-size row. The library choice is `../inverse-problems/references/libraries.md`.
- Nonconvex smooth term plus a prox: iPiano, inside the paper's step and inertia bounds (Ochs, Chen, Brox, Pock).
- Weakly convex learned regulariser: stay inside the modulus assumed by Goujon, Neumayer, and Unser, or by Shumaylov, Budd, Mukherjee, and Schönlieb for PDHG. The objective is then handled as weakly convex, and the claim is critical-point convergence.
- The data term is a long sum and a denoiser replaces the prox: one subset of measurements per data step (online plug-and-play). The convergence claim is still one of the three hypotheses in `../inverse-problems/references/communities.md`.
- Landweber, iterated Tikhonov, dual diagonal descent, or a deep image prior: these are semi-convergent. The iteration count is the regularisation parameter. Stop when the residual reaches the noise level. Dual diagonal descent and its inertial form cover a general convex penalty and a Kullback–Leibler fidelity; an inexact prox keeps the rate only under the error control in the iterative-regularisation section of `../inverse-problems/references/communities.md`.

Plug-and-play keeps the data-term step of proximal gradient, HQS, ADMM, or Douglas-Rachford and replaces the prox. RED replaces the prior gradient by `x - D(x)`. Quote a convergence or "this is a prior gradient" claim only under the matching hypothesis in `../inverse-problems/references/communities.md`. Langevin and diffusion samplers are not rows of this skill. A total-variation density inside a sampler is not discretization-invariant; deterministic `TVPrior` minimisation is a different estimator.

## Bilevel parameters

`λ`, a Fields-of-Experts filter bank, or a sampling mask is an outer variable. The inner problem stays the variational reconstruction. If that inner problem is nonsmooth, differentiate a fixed run of primal-dual iterations (Ochs, Ranftl, Brox, Pock); the procedure is in `references/algorithms.md`. If it is smooth, unroll it (`unfold=True`, `trainable_params`) or use the implicit-function route (`DEQConfig` on `GD`, `PGD`, or `HQS`). The implicit route computes the hypergradient only approximately; when the inner problem is smooth and strongly convex, set the inner and linear-solve tolerances adaptively from a posteriori bounds (MAID) rather than fixing an iteration count, as in the inexact-hypergradient part of the reference.

## Implementation

```python
import torch
import deepinv as dinv

torch.manual_seed(0)
x = torch.rand(1, 1, 32, 32)
physics = dinv.physics.BlurFFT(
    img_size=(1, 32, 32),
    filter=dinv.physics.blur.gaussian_blur(sigma=1.5),
    noise_model=dinv.physics.GaussianNoise(sigma=0.02),
)
y = physics(x)

# ‖A‖² from a batched tensor shaped like x (not y); the estimate undershoots, so keep a margin.
L = float(physics.compute_sqnorm(x, verbose=False))
model = dinv.optim.PGD(
    prior=dinv.optim.TVPrior(),
    data_fidelity=dinv.optim.L2(),
    stepsize=0.9 / L,  # proximal gradient allows γ < 2/L; FISTA needs γ ≤ 1/L
    lambda_reg=0.02,
    max_iter=50,
)
x_hat = model(y, physics, init=physics.A_dagger(y))
```

Swap the class using the table in the reference when the structure is not proximal-gradient. For `PDCP`, follow the wiring in the algorithms reference: `K=physics.A`, the identity physics at call time, `stepsize` and `stepsize_dual` set from `‖A‖²`, `beta=1`, and the step product asserted on the numbers you actually run. Backtracking (`BacktrackingConfig`) is available when the prior has an explicit cost; it is disabled for denoiser priors.

Parameters may be a scalar or a list with one value per iteration, which is how a continuation schedule on `λ` or on a denoiser noise level is written.

Confirm class names against the installed deepinv. The reference is the 0.4.2 surface.
