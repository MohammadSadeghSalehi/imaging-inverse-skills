# Algorithms for imaging objectives

Lookup for the imaging-optimisation skill. DeepInverse class names match 0.4.2. The reconstruction-method decision and the loss table live in the inverse-problems skill. The forward operator lives in the computational-imaging skill.

The objective these algorithms exist for is

```
min_x  f(x; y) + λ R(x)
```

with `f(x; y) = d(A(x), y)`. When both `f` and `R` are nonsmooth, or `A` sits inside a nonsmooth fidelity, write the saddle point

```
min_x max_u  ⟨Kx, u⟩ + G(x) - F*(u)
```

and use a primal-dual method. `K` is whatever linear map was inside the conjugate: a finite-difference gradient for TV, the forward operator for an L1 fidelity, or both stacked.

## Step-size rules

Estimate norms numerically. For a deepinv linear physics, `physics.compute_sqnorm(x)` estimates `‖A‖²` by power iteration. Pass a batched tensor with the shape of the image `x`, never the measurement `y`: on tomography a measurement-shaped input raises, and on other operators it only works when `x` and `y` happen to share a shape. The power iteration approaches `‖A‖²` from below, so an estimate at the default `tol=1e-3` can sit a few per cent low; tighten `tol` or keep a margin in the step (for example `0.9 / L` rather than `1 / L`). For a finite-difference gradient, use the bound for that discretisation rather than a power iteration you forgot to run.

| Algorithm | Condition | What you may claim |
| --- | --- | --- |
| Gradient descent on an `L`-smooth convex function | step `γ < 2/L` for descent; `γ = 1/L` is the safe textbook choice | Objective decrease. |
| Proximal gradient, data term `L`-smooth, prior has a prox | `0 < γ < 2/L`. For `f = ½‖Ax - y‖²`, `L = ‖A‖²`. `γ = 1/L` is the safe default. | Convergence of the iterates and `O(1/k)` on the objective for convex `f` and `R`. |
| FISTA | `γ ≤ 1/L`, stricter than proximal gradient. | `O(1/k²)` on the objective values. Iterate convergence needs the Chambolle–Dossal momentum (parameter `a > 2`), not only the original Beck–Teboulle sequence. Chambolle's profile paper is "On the convergence of the iterates of the fast iterative shrinkage/thresholding algorithm". |
| Chambolle–Pock PDHG | `τ σ ‖K‖² ≤ 1` with extrapolation `beta = 1` (`θ = 1`). `deepinv.optim.PDCP` states this condition in its docstring but does not check it; compute the product and assert it yourself. The 2011 ergodic-rate proof uses a strict inequality, so keep the product below 1 when you cite that rate. | Ergodic `O(1/N)` on the primal-dual gap when `G` and `F*` are convex and the product is below 1. Cite Chambolle and Pock, JMIV 2011. |
| SPDHG | Per-block condition from Chambolle, Ehrhardt, Richtárik, Schönlieb, SIAM J. Optim. 2018. Serial sampling uses the block probabilities `p_i` and the block norms `‖K_i‖`. Applying the PDHG product condition to the full stacked operator is sufficient and looser than the block condition. | Same convex saddle-point guarantee, with subset updates. |
| Adaptive SPDHG | Keep the product of the adapted steps inside the 2018 regime. Chambolle, Delplancke, Ehrhardt, Schönlieb, Tang, JMIV 2024. | Use when the primal/dual ratio is not known. Demonstrated on CT. |
| Chambolle 2004 dual TV | Projected gradient on the dual ball, step `τ ≤ 1/8` for the standard 2D forward-difference gradient (`‖div‖² ≤ 8`). | Convergence for the ROF problem. |
| iPiano | Inertial proximal step on smooth nonconvex plus prox-friendly nonsmooth. Ochs, Chen, Brox, and Pock, SIAM Journal on Imaging Sciences, 2014. | Convergence to a critical point under that paper's step and inertia restrictions. |
| PDHG with a weakly convex regulariser | Shumaylov, Budd, Mukherjee, Schönlieb, ICML 2024. | Convergence of the iterates to a critical point, and an ergodic rate under a Kurdyka–Łojasiewicz condition. |

Diagonal preconditioning (Pock and Chambolle, "Diagonal preconditioning for first order primal-dual algorithms") replaces the scalar steps by diagonal matrices built from the absolute row and column sums of `K`, so the same product condition holds mode-wise. Use it for optical flow, multi-term TV, and any `K` whose rows have very different scales.

## Which algorithm

| Structure you can verify | Algorithm | deepinv class |
| --- | --- | --- |
| `f` and `R` both smooth | Gradient descent | `GD` |
| `f` smooth, `R` prox-friendly | Proximal gradient | `PGD` |
| Same, and you want the accelerated rate on the objective | FISTA | `FISTA` |
| Both terms prox-friendly, `A` inside `f` | Half-quadratic splitting or ADMM or Douglas–Rachford | `HQS`, `ADMM`, `DRS` |
| Saddle point, two proxes, a linear map `K` | Chambolle–Pock | `PDCP` |
| Poisson counts, nonnegative image, no extra prior | EM | `MLEM` |
| Same, data split into subsets | Ordered-subset EM | `OSEM` |
| EM with a differentiable prior | One-step-late MAP-EM (Green, 1990). A heuristic: no convergence proof, and it breaks down when `λ ∇R` makes the denominator nonpositive. | `MLEM` or `OSEM` with `prior=` |
| Tomography with diagonal weights | Simultaneous iterative reconstruction | `SIRT` |
| Mirror map / Bregman geometry (simplex, positive orthant) | Mirror descent | `MD`, `PMD` |
| Poisson deconvolution, known kernel | Richardson–Lucy, which is MLEM on a blur operator | `MLEM` with a `Blur` physics |

`PDCP` is the library's Chambolle–Pock iteration, dual update first:

```
u <- prox_{σ F*}(u + σ K z)
x <- prox_{τ λ G}(x - τ Kᵀ u)
z <- x + β (x - x_prev)
```

Pass `stepsize` as `τ`, `stepsize_dual` as `σ`, and `beta` as the extrapolation. Set `beta=1` to recover the over-relaxed form the 2011 convergence proof uses.

Bregman potentials for mirror descent: `BregmanL2`, `BurgEntropy` (positive intensities), `NegEntropy` (simplex), `Bregman_ICNN` (a learned convex mirror map, `deepinv.models.ICNN`).

## Total variation and TGV

Isotropic TV is `‖Dx‖_{1,2}`. In deepinv this is `TVPrior`, whose prox is what `PGD` calls. Anisotropic TV is `TVL1Prior`.

`PDCP` wiring. The dual step of `PDCP` takes the conjugate prox of the whole data term `x ↦ d(A x, y)`, and `A` there is the `physics` passed at call time. So `PDCP(K=physics.A, ...)` called with that same `physics` applies `A` twice and minimises `d(A A x, y) + λ R(x)`, a different problem, with no error raised. Two correct wirings:

- Put `A` in `K` and an identity operator in the data term: `K=physics.A`, `K_adjoint=physics.A_adjoint`, and call `model(y, dinv.physics.Denoising())`. The dual step is then the conjugate prox of `d(·, y)`, which is closed form for `L2`, `L1`, and `IndicatorL2`.
- Keep the default `K` (the identity) and call with the real `physics`. The dual step then needs the conjugate prox of `d(A·, y)`, which for `L2` is a linear solve inside every iteration.

On a Gaussian deblurring problem with TV, the first wiring reaches the proximal-gradient objective to five significant figures; `K=physics.A` with the real physics stops about 4% higher, at the minimiser of the wrong problem. With `K` equal to the finite-difference gradient instead, the dual variable is the TV dual field of the 2011 saddle point.

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

norm_K2 = float(physics.compute_sqnorm(x, tol=1e-6, verbose=False))  # ‖K‖², K = A
tau = sigma = 0.99 / norm_K2 ** 0.5
assert tau * sigma * norm_K2 < 1  # PDCP does not check this itself

model = dinv.optim.PDCP(
    K=physics.A,
    K_adjoint=physics.A_adjoint,
    data_fidelity=dinv.optim.L2(),
    prior=dinv.optim.TVPrior(),
    lambda_reg=0.02,
    stepsize=tau,
    stepsize_dual=sigma,
    beta=1.0,
    max_iter=50,
)
x_hat = model(y, dinv.physics.Denoising())  # A lives in K, so the data term sees the identity
```

The dual formulation, which is what Chambolle 2004 solves, is the ROF problem

```
min_x ½‖x - y‖² + λ ‖Dx‖_{1,2}
= max_{‖p‖_∞ ≤ 1} ½‖y‖² - ½‖y - λ div p‖²
```

with the reconstruction `x = y - λ div p`. Use the dual projection when the data term is a plain squared norm. Use PDHG when the data term is `½‖Ax - y‖²` for a general `A` or when a second nonsmooth term is present.

Second-order total generalised variation (Bredies, Kunisch, Pock) is

```
TGV²_α(x) = min_w  α₁ ‖Dx - w‖_{1,2} + α₀ ‖Ew‖_{1,2}
```

with `E` the symmetrised derivative. Affine regions have zero cost, so ramps survive. Put `w` in the primal (or its dual fields in the dual) and solve the saddle point with PDHG. Reach for TGV when a TV reconstruction has produced flat staircases on a smoothly shaded object.

## Plug-and-play and RED inside a solver

Replacing `prox_{γλR}` by a denoiser, or replacing `∇R` by `x - D(x)`, keeps the data-term step of the row already chosen. The hypotheses under which that iteration converges, and the papers to cite, are in `../../inverse-problems/references/communities.md`. A generic pretrained denoiser meets none of them until you check.

Langevin (`ULA`, `SKRock`) and diffusion samplers are not rows of the table above. They target a posterior. Hand them to the inverse-problems skill.

## Bilevel parameters

Learning `λ`, a filter bank, or a sampling mask is the outer problem

```
min_θ  ℓ(x(θ), x_ref)   subject to   x(θ) = argmin_x f(x; y) + R(x; θ)
```

When the lower-level problem is nonsmooth, as total variation is, the solution map is not a gradient you can write down. Ochs, Ranftl, Brox, and Pock, "Techniques for Gradient-Based Bilevel Optimization with Non-smooth Lower Level Problems", Journal of Mathematical Imaging and Vision, 2016: replace the minimiser by a differentiable iterative algorithm (a Bregman proximal or primal-dual map) and differentiate those iterations. The parameters are then optimal for that fixed iteration count, which is an unroll, not a claim about the infinite-iteration minimiser. Smoothing the lower level and differentiating the smoothed problem solves the smoothed problem. Say which one you did.

A nonsmooth nonconvex penalty, such as an `ℓᵖ` penalty with `p < 1` on the gradient, is the iteratively reweighted scheme of Ochs, Dosovitskiy, Brox, and Pock, SIAM Journal on Imaging Sciences, 2015. The convex primal-dual rate does not apply. iPiano, in the table above, is the inertial proximal step for smooth nonconvex plus a prox.

Two implementations that stay faithful to a smooth or prox-friendly lower level:

- Unroll a fixed number of the inner algorithm and differentiate through it. In deepinv, `unfold=True` and `trainable_params`. This is the variational-network construction (Hammernik, Pock, and coauthors) when each step is a gradient step on a Fields-of-Experts energy. The tomography form, with both proximal maps learned, is the learned primal-dual in `../../inverse-problems/references/communities.md`.
- Implicit differentiation of the optimality condition, when the inner solver runs to convergence and the optimality condition is differentiable. In deepinv this is the deep-equilibrium path, `DEQConfig`, on `GD`, `PGD`, or `HQS`.

Keep the inner problem convex, or weakly convex with a convergent algorithm, if the learned parameter is going to be reused at test time inside that same algorithm. An outer loss alone does not make an arbitrary unrolled network a minimiser of the variational model.

Inexact hypergradients. The implicit route needs the lower-level minimiser and the solution of a linear system in the Hessian, and both are computed only approximately. A fixed small number of inner iterations gives a biased hypergradient with no descent guarantee on the outer loss. When the lower level is smooth and strongly convex, the distance to its minimiser is certified a posteriori by `‖x − x*‖ ≤ ‖∇ₓh(x)‖ / μ`, and the hypergradient error can be bounded from that and from the residual of the conjugate-gradient solve. The method of adaptive inexact descent (MAID) sets both tolerances from those bounds, tightens them only when a backtracking line search on the outer loss fails, and so keeps a sufficient-decrease guarantee while spending little lower-level work early on: Salehi, Mukherjee, Roberts, and Ehrhardt, "An Adaptively Inexact First-Order Method for Bilevel Optimization with Application to Hyperparameter Learning", SIAM Journal on Mathematics of Data Science 7(3), 2025, doi:10.1137/24M1653513. Use it when the lower-level solve dominates the cost of learning `λ`, filters, or the parameters of a convex regulariser. It does not apply to a nonsmooth lower level; that is the unrolling case above.

## Checks before handing code back

1. State `f`, `R`, and whether each is convex, smooth, and prox-friendly.
2. Compute `‖A‖²` or `‖K‖²` and set the step from the table. For PDHG, print `τ σ ‖K‖²` and keep it inside the PDHG row of the table above.
3. Log the objective `f + λR` when both terms have an explicit cost, or the primal-dual gap for a saddle point. Image change between iterations is a stopping test, not a convergence certificate.
4. For a nonnegative physical quantity (PET, CT attenuation, photon flux), put the indicator of the positive orthant in `G` and use its prox, which is projection, or use `MLEM` / `BurgEntropy`.
5. Match the quoted rate to the table row you actually satisfied. Function-value rates, iterate convergence, and critical-point convergence are different statements.
