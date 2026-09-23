---
name: inverse-problems
description: >
  Choose, implement, and report a reconstruction method for an ill-posed
  imaging inverse problem: Tikhonov and total variation, sparsity,
  plug-and-play and RED, unrolled and learned primal-dual networks, convex
  and weakly convex learned regularisers, self-supervised and equivariant
  training, deep image prior, diffusion and Langevin posterior sampling,
  and nonlinear problems such as impedance and optical tomography. Picks
  DeepInverse, ASTRA, CIL, or LION. Use when the user wants to reconstruct
  an image from indirect or corrupted data, choose a regulariser or its
  weight, train without ground truth, quantify uncertainty, or decide
  whether a learned method is justified.
---

# Inverse problems

Recover `x` from `y = N(A(x))` when `A` is not stably invertible. The job of this skill is to choose the reconstructor and the training loss. The forward model is specified by `../computational-imaging/SKILL.md`. The step size and the convergence claim are specified by `../imaging-optimisation/SKILL.md`. Read both before returning solver code.

Citations for the variational line: `references/lineage.md`. Method choices beyond that line: `references/communities.md`. Which of DeepInverse, ASTRA, CIL, and LION to call: `references/libraries.md`. DeepInverse class names (0.4.2; check the installed package): `references/deepinv.md`.

Organise the method the way Arridge, Maass, Öktem, and Schönlieb do in their 2019 Acta Numerica survey. A learned piece replaces one identified slot of a variational problem: the regulariser, the proximal map, the gradient step, or the whole inverse. Name the slot.

## Choose the method

Walk down this list and take the first row whose hypothesis you can actually check. If the deliverable is a posterior draw or an uncertainty map, use the diffusion or Langevin row even when an earlier row also matches.

| Situation | Method |
| --- | --- |
| `A` is well conditioned; noise is the only defect | Denoise, or Tikhonov / Wiener. A reconstruction network is unnecessary. |
| The unknown is a conductivity or an optical absorption or scattering field from boundary measurements | Diffuse optical tomography or electrical impedance tomography, in `references/communities.md`. A regularised D-bar reconstruction or a Bayesian estimate comes before a pixel network. |
| Limited-angle tomography, with edges the angular range cannot determine | Shearlet sparsity on the visible wavefront, and a network only for the invisible coefficients (Bubba, with Siltanen). Or a pseudodifferential correction of the normal operator (ΨDONet). Both are in `references/communities.md`. |
| The unknown is a PDE coefficient, and the data are partial observations of the PDE solution | Penalty formulation (van Leeuwen and Herrmann), in the PDE-constrained section of `references/communities.md`. A reduced adjoint-state loop is the local-minimum formulation unless the initial coefficient is already close. |
| The unknown takes a few known material values | Discrete algebraic reconstruction (DART, Batenburg and Sijbers), not continuous TV. |
| The object is piecewise constant, piecewise smooth, or sparse in a known basis, and you need a convex guarantee | Explicit regulariser: `TVPrior`, second-order TGV, `WaveletPrior`, or `L1Prior`, solved by the imaging-optimisation skill. TGV when a TV result staircases on ramps. Deterministic TV only. A TV density is not a discretization-invariant Bayesian prior. |
| A registered anatomical image should share edge directions | Parallel level sets (Ehrhardt), as the structural prior inside the same variational problem. |
| Counts, nonnegative image, no extra prior | `MLEM` or `OSEM`. With a regulariser, a convergent penalised Poisson solve from the imaging-optimisation skill; `MLEM` or `OSEM` with `prior=` is the one-step-late heuristic and carries no convergence claim. |
| A pretrained denoiser exists and there is no paired data for this `A` | Plug-and-play (`PnP`) or RED, with `DRUNet` as the usual pretrained denoiser, or `DPIR` for the prebuilt schedule. Quote convergence only under one of the three hypotheses in `references/communities.md`; `GSDRUNet` is the gradient-step denoiser built for the third. |
| The regulariser must stay convex | Convex-ridge network (Goujon, Neumayer, Bohra, Ducotterd, Unser, IEEE TCI 2023), trained as a denoiser and deployed as `R`, or an input-convex network trained bilevel. Hertrich and coauthors, "Learning regularization functionals for inverse problems: a comparative study", Handbook of Numerical Analysis, 2026, doi:10.1016/bs.hna.2026.04.001, compare the learned-regulariser families in one framework with practical guidelines; read it before choosing among them. |
| A nonconvex prior is worth it and the iteration must still converge | Weakly convex regulariser with an explicit modulus (Goujon, Neumayer, Unser, SIAM J. Imaging Sci. 2024). Keep the solver inside that paper's step restriction, or the PDHG extension (Shumaylov, Budd, Mukherjee, Schönlieb, ICML 2024). |
| A small paired set should set `λ`, the filters, or the potentials of a variational model | Bilevel learning. The hypergradient method follows the lower level: unroll a fixed number of primal-dual iterations when it is nonsmooth (Ochs, Ranftl, Brox, Pock), or use implicit differentiation with adaptively set accuracy (MAID) when it is smooth and strongly convex. The decision table is in the bilevel section of the imaging-optimisation algorithms reference. A variational network is the MRI form of the same unroll. |
| Paired simulated measurements exist and test-time `A` will match training `A` | Unroll a short iteration that still calls `A` and `A*`. Tomography: learned primal-dual (Adler and Öktem). Multi-coil MRI: `VarNet` or `MoDL`. Otherwise unfolded `PGD`, `HQS`, or `ADMM`. Which of these is post-processing rather than an unroll is the unrolled-reconstructions section of `references/communities.md`. |
| One measurement and no training set | Deep image prior (`dinv.models.DeepImagePrior`). Stop when the residual reaches the noise level. The weights do not transfer. |
| Only measurements exist | Self-supervised loss from the table in `references/deepinv.md`. Equivariant imaging (`EILoss`; Chen, Tachella, Davies) when a group symmetry of the images is real and `A` breaks it. Measurement splitting when the noise is independent across coordinates. For noisy tomography, split projection angles (Noise2Inverse), not pixels of the reconstruction. SURE when `σ` is known. |
| The deliverable is a posterior draw or an uncertainty map, and a denoiser trained across noise levels exists | Diffusion posterior sampling. Pick `DDRM`, `DiffPIR`, or `DPS` from the sampling section of `references/deepinv.md`. Average several draws for a mean. |
| The deliverable is a posterior draw and the log posterior is an explicit imaging model | Langevin. `ULA` for a short chain. `SKRock` when the posterior is ill-conditioned. MYULA when the potential is convex and nonsmooth. These target a posterior, and MYULA targets its Moreau smoothing; the distinctions are in `references/communities.md`. |
| A Lipschitz functional should score "looking like data" inside a variational objective | Adversarial regulariser (Lunz, Öktem, Schönlieb). This is a penalty `Ψ_θ(x)`, not a GAN that outputs `x`. |

An end-to-end network that never receives `A` is the right row only when `A` is frozen between train and test. The moment the mask, the blur, the dose, or the coil maps can change, put `A` back into the architecture (unrolled data-consistency step, or a variational model).

A strong learned baseline costs one line: `dinv.models.RAM` is a pretrained reconstructor that takes `(y, physics)` for a range of standard operators. Run it next to whatever you build. Its training operators and images are listed in its documentation; check it on your data before treating it as the reference.

## Choose the regularisation weight

`λ` is part of the method, and a hand-tuned `λ` makes a comparison meaningless. In order of preference when there is no paired data:

- Discrepancy principle (Morozov): choose `λ` so that `‖A x_λ − y‖²` matches the expected noise energy, `m σ²` for `m` Gaussian measurements. Needs `σ`. The same rule stops Landweber, deep image prior, and every other semi-convergent iteration.
- Stein's unbiased risk estimate: with Gaussian noise of known `σ`, SURE estimates the mean-squared error in the measurement domain without ground truth. Evaluate it over a grid of `λ`; `SureGaussianLoss` computes it for any deepinv model. `SurePoissonLoss` and `SurePGLoss` cover the other noise models.
- Generalised cross-validation, when `σ` is unknown and the estimator is linear in `y` (Tikhonov, truncated SVD).
- The L-curve is a heuristic with known failure cases. Report it as such.
- With a small paired set, learn `λ` bilevel, as in the imaging-optimisation skill.

Report `λ` and the rule that chose it.

## Training

- Paired data: `SupLoss`, and pass `physics` into the model so the data-consistency step sees `A`.
- Measurements only: pick the loss from `references/deepinv.md` by the noise and by whether several operators are observed.
- The group in an equivariant-imaging loss is a symmetry of the object distribution. Rotations fit cells and textures. They do not fit an exam whose orientation is anatomical.
- Control the Lipschitz constant while training a denoiser you will insert into plug-and-play (`JacobianSpectralNorm`, `FNEJacobianSpectralNorm`).

## What to report with the reconstruction

State which row of the table you used and which hypothesis it required. State whether the number you quote is a distortion score (PSNR, SSIM) or a perceptual one (LPIPS), and compute it with `dinv.metric` so the data range is handled consistently. State `λ` and the rule that chose it. Run the residual and null-space checks from the computational-imaging skill. Run the step-size and objective-log checks from the imaging-optimisation skill before calling an iteration convergent.

A spread across posterior draws is uncertainty under the model and the prior, not a calibrated error bar. Before quoting it as one, check empirical coverage on held-out ground truth: the fraction of pixels whose true value falls inside the reported interval. Diffusion samplers built on an approximate likelihood score are routinely overconfident.

## Implementation

Pick the package from `references/libraries.md`, then use its classes. The DeepInverse skeleton and class tables are in `references/deepinv.md`. Read the installed signature if the import fails. The DeepInverse names are the 0.4.2 surface.

```python
import torch
import deepinv as dinv

torch.manual_seed(0)
x = torch.rand(1, 1, 32, 32)
sigma = 0.05
physics = dinv.physics.Inpainting(
    img_size=(1, 32, 32), mask=0.6, noise_model=dinv.physics.GaussianNoise(sigma=sigma)
)
y = physics(x)

L = float(physics.compute_sqnorm(x, verbose=False))  # shape of x, not y
model = dinv.optim.PGD(
    prior=dinv.optim.TVPrior(),
    data_fidelity=dinv.optim.L2(),
    stepsize=0.9 / L,
    lambda_reg=0.05,
    max_iter=50,
)
x_hat = model(y, physics)

# Discrepancy check: the residual should be at the noise level, not far below it.
residual = float((physics.A(x_hat) - y).pow(2).sum())
noise_energy = y.numel() * sigma**2
print(f"residual / noise energy = {residual / noise_energy:.2f}")
```

Set `stepsize` from the imaging-optimisation rule for whichever solver you swapped in. `0.9 / ‖A‖²` is a safe proximal-gradient step; FISTA and `PDCP` have their own conditions.
