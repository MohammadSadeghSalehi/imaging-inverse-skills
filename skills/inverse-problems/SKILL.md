---
name: inverse-problems
description: >
  Choose and implement a reconstruction method for an ill-posed imaging
  inverse problem: variational regularisation, plug-and-play, unrolled
  networks, convex and weakly convex neural regularisers, adversarial
  regularisers, equivariant and self-supervised losses, and posterior
  sampling. Distilled from Schönlieb, Ehrhardt, Pock, Neumayer, and
  Chambolle, with DeepInverse (deepinv) as the library. Use when the user
  says inverse problem, image reconstruction, regulariser, regularizer,
  plug-and-play, RED, diffusion posterior sampling, Langevin, MYULA,
  deep image prior, diffuse optical tomography, electrical impedance
  tomography, statistical inversion, or runs /inverse-problems.
---

# Inverse problems

Recover `x` from `y = N(A(x))` when `A` is not stably invertible. The job of this skill is to choose the reconstructor and the training loss. The forward model is specified by `../computational-imaging/SKILL.md`. The step size and the convergence claim are specified by `../imaging-optimisation/SKILL.md`. Read both before returning solver code.

Citations for the variational line: `references/lineage.md`. Plug-and-play, RED, diffusion, Langevin, and the French and American lines: `references/communities.md`. DeepInverse class names (0.4.2; check the installed package): `references/deepinv.md`.

Organise the method the way Arridge, Maass, Öktem, and Schönlieb do in their 2019 Acta Numerica survey. A learned piece replaces one identified slot of a variational problem: the regulariser, the proximal map, the gradient step, or the whole inverse. Name the slot.

## Choose the method

Walk down this list and take the first row whose hypothesis you can actually check. If the deliverable is a posterior draw or an uncertainty map, use the diffusion or Langevin row even when an earlier row also matches.

| Situation | Method |
| --- | --- |
| `A` is well conditioned; noise is the only defect | Denoise, or Tikhonov / Wiener. A reconstruction network is unnecessary. |
| The unknown is a conductivity or an optical absorption or scattering field from boundary measurements | Diffuse optical tomography or electrical impedance tomography, in `references/communities.md`. A D-bar reconstruction or a Bayesian estimate comes before a pixel network. |
| The object is piecewise constant, piecewise smooth, or sparse in a known basis, and you need a convex guarantee | Explicit regulariser: `TVPrior`, second-order TGV, `WaveletPrior`, or `L1Prior`, solved by the imaging-optimisation skill. TGV when a TV result staircases on ramps. Deterministic TV only. A TV density is not a discretization-invariant Bayesian prior. |
| A registered anatomical image should share edge directions | Parallel level sets (Ehrhardt), as the structural prior inside the same variational problem. |
| Counts, nonnegative image, no extra prior | `MLEM` or `OSEM`. Add `BSREM` when a regulariser is present. |
| A pretrained denoiser exists and there is no paired data for this `A` | Plug-and-play (`PnP`) or RED. Quote convergence only under one of the three hypotheses in `references/communities.md`. |
| The regulariser must stay convex | Convex-ridge network (Goujon, Neumayer, Bohra, Ducotterd, Unser, IEEE TCI 2023), trained as a denoiser and deployed as `R`. |
| A nonconvex prior is worth it and the iteration must still converge | Weakly convex regulariser with an explicit modulus (Goujon, Neumayer, Unser, SIAM J. Imaging Sci. 2024). Keep the solver inside that paper's step restriction, or the PDHG extension (Shumaylov, Budd, Mukherjee, Schönlieb, ICML 2024). |
| A small paired set should set `λ`, the filters, or the potentials of a variational model | Bilevel learning. Implement it as a short unroll (`unfold=True`) or as a deep equilibrium. A variational network (Hammernik, Pock and coauthors) is the MRI form: unrolled gradient steps on a Fields-of-Experts energy. |
| Paired simulated measurements exist and test-time `A` will match training `A` | Unrolled `PGD`, `HQS`, or `ADMM` with a small fixed iteration count, or `VarNet` / `MoDL` for multi-coil MRI. |
| One measurement and no training set | Deep image prior. Stop when the residual reaches the noise level. The weights do not transfer. |
| Only measurements exist | Self-supervised loss from the table in `references/deepinv.md`. Equivariant imaging (`EILoss`; Chen, Tachella, Davies) when a group symmetry of the images is real and `A` breaks it. Measurement splitting when the noise is independent across coordinates. SURE when `σ` is known. |
| The deliverable is a posterior draw or an uncertainty map, and a denoiser trained across noise levels exists | Diffusion posterior sampling. Pick `DDRM`, `DiffPIR`, or `DPS` from the sampling section of `references/deepinv.md`. Average several draws for a mean. |
| The deliverable is a posterior draw and the log posterior is an explicit imaging model | Langevin. `ULA` for a short chain. `SKRock` when the posterior is ill-conditioned. MYULA when the potential is convex and nonsmooth. These target a posterior, and MYULA targets its Moreau smoothing; the distinctions are in `references/communities.md`. |
| A Lipschitz functional should score "looking like data" inside a variational objective | Adversarial regulariser (Lunz, Öktem, Schönlieb). This is a penalty `Ψ_θ(x)`, not a GAN that outputs `x`. |

An end-to-end network that never receives `A` is the right row only when `A` is frozen between train and test. The moment the mask, the blur, the dose, or the coil maps can change, put `A` back into the architecture (unrolled data-consistency step, or a variational model).

## Training

- Paired data: `SupLoss`, and pass `physics` into the model so the data-consistency step sees `A`.
- Measurements only: pick the loss from `references/deepinv.md` by the noise and by whether several operators are observed.
- The group in an equivariant-imaging loss is a symmetry of the object distribution. Rotations fit cells and textures. They do not fit an exam whose orientation is anatomical.
- Control the Lipschitz constant while training a denoiser you will insert into plug-and-play (`JacobianSpectralNorm`, `FNEJacobianSpectralNorm`).

## What to report with the reconstruction

State which row of the table you used and which hypothesis it required. State whether the number you quote is a distortion score (PSNR, SSIM) or a perceptual one. Run the residual and null-space checks from the computational-imaging skill. Run the step-size and objective-log checks from the imaging-optimisation skill before calling an iteration convergent.

## Implementation

Use deepinv. The working skeleton and the class tables are in `references/deepinv.md`. Read the installed signature if the import fails; the documented surface is version 0.4.2.

```python
import deepinv as dinv

data_fidelity = dinv.optim.L2()
prior = dinv.optim.TVPrior()
norm_A2 = physics.compute_norm(y, tol=1e-4, verbose=False).item()
model = dinv.optim.PGD(
    prior=prior,
    data_fidelity=data_fidelity,
    stepsize=0.99 / norm_A2,
    lambda_reg=0.05,
    max_iter=100,
)
x_hat = model(y, physics)
```

Set `stepsize` from the imaging-optimisation rule for whichever solver you swapped in. `0.99 / ||A||²` is the proximal-gradient case only.
