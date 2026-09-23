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

### Choosing the hypergradient method

The hypergradient of `ℓ(x(θ))` is where bilevel learning spends its compute, and each way of computing it supports a different claim. Survey for imaging: Crockett and Fessler, "Bilevel Methods for Image Reconstruction", Foundations and Trends in Signal Processing 15(2–3), 2022, doi:10.1561/2000000111. The variational-model form is Kunisch and Pock, "A Bilevel Optimization Approach for Parameter Learning in Variational Models", SIAM Journal on Imaging Sciences 6(2), 2013, doi:10.1137/120882706.

| Lower level, and what you have | Hypergradient | What you may claim |
| --- | --- | --- |
| Nonsmooth, or you will deploy a fixed number of iterations | Iterative differentiation: unroll `K` steps and backpropagate (Franceschi, Donini, Frasconi, Pontil, ICML 2017; Ochs, Ranftl, Brox, Pock above for nonsmooth steps) | Parameters optimal for that `K`-step algorithm, not for the minimiser. |
| Smooth and strongly convex, solved to convergence | Implicit differentiation: solve `∇²ₓₓh q = ∇ₓℓ` by conjugate gradient, then `−∇²_θₓh q` | The exact hypergradient in the limit. Grazzi, Franceschi, Pontil, Salzo, "On the Iteration Complexity of Hypergradient Computation", ICML 2020, give rates for both this and the unrolled route in the number of inner iterations. |
| Same, and the lower-level solve dominates the cost | Adaptively inexact implicit differentiation: MAID (below). The fixed-schedule predecessor is HOAG, which drives the tolerances down along a summable sequence (Pedregosa, "Hyperparameter optimization with approximate gradient", ICML 2016). | Sufficient decrease on the upper loss under the paper's assumptions, without a hand-tuned tolerance schedule. |
| Same, and no Hessian-vector products are available | Derivative-free bilevel with inexact lower-level solves: Ehrhardt and Roberts, "Inexact Derivative-Free Optimization for Bilevel Learning", Journal of Mathematical Imaging and Vision 63(5), 2021, doi:10.1007/s10851-021-01020-8 | Convergence for a small number of parameters; the cost grows with the dimension of `θ`. |
| Many training pairs, sampled in minibatches | Inexact stochastic hypergradients (below), or stocBiO (Ji, Yang, Liang, "Bilevel Optimization: Convergence Analysis and Enhanced Design", ICML 2021). Warm-start the inner solve and the linear system across outer steps (Arbel and Mairal, "Amortized Implicit Differentiation for Stochastic Bilevel Optimization", ICLR 2022). | Convergence in expectation to a stationary point, at the rate the schedule supports. |
| Convex but nonsmooth, solved by a primal-dual method, learning an operator or a convex regulariser | Piggyback primal-dual differentiation with an a posteriori bound (below). For a general nonsmooth solution map, conservative Jacobians (Bolte, Le, Pauwels, Silveti-Falls, "Nonsmooth Implicit Differentiation for Machine Learning and Optimization", NeurIPS 2021). | A hypergradient whose error is bounded by the computed tolerances. |
| A fixed-point network (deep equilibrium) | Implicit differentiation of the fixed point (Bai, Kolter, Koltun, "Deep Equilibrium Models", NeurIPS 2019; imaging: Gilton, Ongie, Willett, IEEE Transactions on Computational Imaging 7, 2021, doi:10.1109/TCI.2021.3118944). Jacobian-free backpropagation drops the inverse (Fung and coauthors, AAAI 2022). | With the Jacobian-free shortcut, a descent direction under that paper's conditions, not the hypergradient. |

How large the hypergradient error is, for each of these estimators, is analysed in Ehrhardt and Roberts, "Analyzing inexact hypergradients for bilevel learning", IMA Journal of Applied Mathematics 89(1), 2024, doi:10.1093/imamat/hxad035. A fixed small number of inner iterations gives a biased hypergradient with no descent guarantee on the upper loss; say so if that is what you ran.

### Adaptively inexact hypergradients (MAID)

When the lower level is smooth and `μ`-strongly convex, the distance to its minimiser is certified a posteriori, `‖x − x*‖ ≤ ‖∇ₓh(x)‖ / μ`, and the conjugate-gradient residual is computable, so the hypergradient error can be bounded from quantities the code already has. The method of adaptive inexact descent (MAID) sets the lower-level and linear-solve tolerances from those bounds, runs a backtracking line search on the upper loss that accounts for the inexactness, and tightens the tolerances only when that line search fails. Early outer steps are therefore cheap, and accuracy is bought only when progress needs it. Salehi, Mukherjee, Roberts, and Ehrhardt, "An Adaptively Inexact First-Order Method for Bilevel Optimization with Application to Hyperparameter Learning", SIAM Journal on Mathematics of Data Science 7(3), 2025, doi:10.1137/24M1653513.

The same group extends the idea in three directions:

- Stochastic upper level, where the sampling of training pairs makes the hypergradient inexact and stochastic: convergence under mild assumptions, with speed-ups and better generalisation than adaptive deterministic methods on denoising and deblurring. Salehi, Mukherjee, Roberts, and Ehrhardt, "Bilevel Learning with Inexact Stochastic Gradients", Scale Space and Variational Methods in Computer Vision (SSVM 2025), doi:10.1007/978-3-031-92366-1_27.
- Rates for inexact stochastic gradient descent under decaying accuracy and step-size schedules, `O(k^{-1/4})` in expectation with the best configuration. Their experiments, with convex-ridge regularisers and input-convex networks, find the accuracy schedule matters more than the step-size schedule. Salehi, Mukherjee, Roberts, and Ehrhardt, "Bilevel Learning via Inexact Stochastic Gradient Descent", arXiv:2511.06774, 2025.
- Nonsmooth convex lower levels solved by primal-dual: the piggyback iteration differentiates the primal-dual algorithm, an a posteriori bound sets the tolerances of both, and the upper step size is adaptive. Used to learn linear operators and input-convex regularisers. Bogensperger, Ehrhardt, Pock, Salehi, and Wong, "An Adaptively Inexact Method for Bilevel Learning Using Primal–Dual-Style Differentiation", Journal of Mathematical Imaging and Vision 67(5), 2025, doi:10.1007/s10851-025-01262-w.

The core of MAID is a hypergradient whose accuracy is certified rather than assumed. DeepInverse 0.4.2 has no bilevel solver, so write it with `torch.autograd` and `deepinv.optim.linear.conjugate_gradient`. The example below learns a total-variation weight: it stops the lower-level solve on the certificate, solves the Hessian system by conjugate gradient, and checks the residual itself, because that function's `tol` is relative to `‖b‖`. The printed hypergradient converges to a central finite difference of the upper loss (`0.67552`) as `eps` shrinks. MAID's contribution is choosing `eps`, and the step on `θ`, adaptively; this example fixes the sequence to show what `eps` controls.

```python
import torch
import deepinv as dinv
from deepinv.optim.linear import conjugate_gradient

torch.manual_seed(0)
x_true = torch.rand(1, 1, 16, 16, dtype=torch.float64)
physics = dinv.physics.BlurFFT(
    img_size=(1, 16, 16),
    filter=dinv.physics.blur.gaussian_blur(sigma=1.0).double(),
    noise_model=dinv.physics.GaussianNoise(sigma=0.05),
)
y = physics(x_true)
mu, nu = 1e-2, 5e-2  # strong convexity and TV smoothing: the lower level is smooth and mu-strongly convex
norm_A2 = float(physics.compute_sqnorm(x_true, tol=1e-6, verbose=False))


def grad2d(x):
    return torch.stack((torch.roll(x, -1, -1) - x, torch.roll(x, -1, -2) - x))


def h(x, theta):  # lower level, with lambda = exp(theta)
    tv = torch.sqrt(grad2d(x).pow(2).sum(0) + nu**2).sum()
    return 0.5 * (physics.A(x) - y).pow(2).sum() + theta.exp() * tv + 0.5 * mu * x.pow(2).sum()


def grad_x(x, theta, create_graph=False):
    x = x if x.requires_grad else x.detach().requires_grad_()
    return torch.autograd.grad(h(x, theta), x, create_graph=create_graph)[0]


def solve_lower(theta, tol, x):
    """Accelerated gradient until the certificate ||x - x*|| <= ||grad_x h(x)|| / mu drops below tol."""
    L = norm_A2 + float(theta.detach().exp()) * 8 / nu + mu
    k = (L / mu) ** 0.5
    beta, z, x_prev = (k - 1) / (k + 1), x.clone(), x.clone()
    while float(grad_x(x, theta).norm()) / mu > tol:
        x_prev, x = x, (z - grad_x(z, theta) / L).detach()
        z = x + beta * (x - x_prev)
    return x


def hypergradient(theta, eps, x0):
    """Hypergradient of 0.5 ||x*(theta) - x_true||^2 from an eps-accurate lower-level solve."""
    x = solve_lower(theta, eps, x0).requires_grad_()
    g = grad_x(x, theta, create_graph=True)
    hess = lambda v: torch.autograd.grad(g, x, v, retain_graph=True)[0]
    b = (x - x_true).detach()  # gradient of the upper loss at x
    q = conjugate_gradient(hess, b, max_iter=1000, tol=eps / float(b.norm()), parallel_dim=None)
    residual = float((hess(q) - b).norm())  # certify the linear solve too; CG's tol is relative to ||b||
    return float(-torch.autograd.grad(g, theta, q)[0]), x.detach(), residual


theta = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
x = physics.A_adjoint(y)
for eps in (1e-1, 1e-2, 1e-3):  # MAID chooses eps adaptively; a fixed sequence shows what it controls
    hg, x, residual = hypergradient(theta, eps, x)
    print(f"eps={eps:.0e}  hypergradient={hg:+.6f}  CG residual={residual:.1e}")
```

## Checks before handing code back

1. State `f`, `R`, and whether each is convex, smooth, and prox-friendly.
2. Compute `‖A‖²` or `‖K‖²` and set the step from the table. For PDHG, print `τ σ ‖K‖²` and keep it inside the PDHG row of the table above.
3. Log the objective `f + λR` when both terms have an explicit cost, or the primal-dual gap for a saddle point. Image change between iterations is a stopping test, not a convergence certificate.
4. For a nonnegative physical quantity (PET, CT attenuation, photon flux), put the indicator of the positive orthant in `G` and use its prox, which is projection, or use `MLEM` / `BurgEntropy`.
5. Match the quoted rate to the table row you actually satisfied. Function-value rates, iterate convergence, and critical-point convergence are different statements.
