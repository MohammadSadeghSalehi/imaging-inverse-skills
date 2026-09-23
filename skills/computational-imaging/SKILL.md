---
name: computational-imaging
description: >
  Write and check the measurement model of a computational imaging system
  before reconstructing: forward operator, noise likelihood, sampling or
  mask design, null space, adjoint test, and evaluation against the noise
  level. Covers MRI and k-space, CT and tomography, PET, deblurring,
  inpainting, super-resolution, phase retrieval, ptychography, spectral
  imaging, compressed sensing, and single-pixel imaging, implemented in
  DeepInverse, ASTRA, or CIL. Use when the user builds or debugs a forward
  model, simulates measurements, picks a noise model or sampling pattern,
  checks an adjoint, or evaluates a reconstruction without ground truth.
---

# Computational imaging

Write the measurement model before any network. The reconstruction method is chosen by the inverse-problems skill, and the solver by the imaging-optimisation skill. Read those files when the task crosses the boundary:

- `../inverse-problems/SKILL.md`
- `../imaging-optimisation/SKILL.md`

Who those names refer to, and which paper to cite, is in `../inverse-problems/references/lineage.md`. Operator and noise class names are in `references/forward-models.md` next to this file. Optical tomography, statistical inversion, spectral filtering, and PDE-constrained coefficients are in `../inverse-problems/references/communities.md`. DeepInverse, ASTRA, CIL, and LION are chosen in `../inverse-problems/references/libraries.md`.

## Procedure

1. Write `y = N(A(x))` with the domain of `x` (pixel image, nonnegative activity, complex coil images) and of `y`.
2. Choose the library from `../inverse-problems/references/libraries.md`. In DeepInverse, implement `A` as a `deepinv.physics` operator and attach `N` from `references/forward-models.md`. A measured tomography geometry uses ASTRA or CIL. If the operator is missing from all four libraries, subclass `Physics` or `LinearPhysics`.
3. Name the null space of `A`. That is the part of `x` the measurements do not determine.
4. Hand the pair `(A, N)` to the inverse-problems skill for the reconstructor. Come back here to judge the result.
5. Evaluate the data residual against the noise level, and evaluate the null-space error separately from the row-space error.

Before step 4, test the adjoint of any linear operator you wrote or wrapped (`physics.adjointness_test(x)` should be at floating-point precision). A wrong adjoint breaks every gradient and primal-dual step silently.

## Modelling rules

Match the data-fidelity term to the likelihood of `N`. Complex MRI noise is Gaussian in k-space. Magnitude MRI is Rician. PET and photon-limited optical data are Poisson, and the image is nonnegative. Pre-log CT is Poisson; post-log CT is a weighted Gaussian approximation. Speckle is Gamma or Fisher-Tippett, depending on the domain. Impulse noise uses an L1 fidelity. The class table is in the reference.

A second registered modality enters as a structural prior. Parallel level sets (Ehrhardt) couple edge directions and leave intensities free, which is the PET-guided-by-MRI model. Use them when edges should coincide and contrasts need not.

A mask, an angular schedule, or a spectral code is a parameter of `A`. Compare designs with the reconstructor held fixed. Learning the design is a bilevel problem: the inner problem reconstructs, the outer problem scores the design on a measurement budget. Ehrhardt's learned MRI sampling pattern is the prototype. Relax the discrete mask during training and project it onto the budget before evaluation.

Phase retrieval and ptychography are nonlinear. Report several initialisations. Compressed sensing needs an explicit measurement budget. For a sparse basis, the budget tracks the sparsity and the sensing operator has to be incoherent with that basis (Candès, Romberg, Tao; variable-density k-space as in Lustig, Donoho, Pauly). For a generative prior `G`, the budget tracks the latent dimension and the unknown is the code `z` in `min_z ‖A G(z) - y‖` (Bora, Jalali, Price, Dimakis). Papers and the optics-design line (Wetzstein; Waller) are in `../inverse-problems/references/communities.md`.

For a compact linear operator, read the recoverable components off the singular spectrum before training anything (Bertero and Boccacci). For deblurring, the boundary condition is part of `A`: anti-reflective conditions when the object reaches the frame edge (Donatelli, Serra-Capizzano, Estatico). Diffuse optical tomography and electrical impedance tomography are nonlinear boundary-measurement problems; they are specified in the communities reference, and they are not rows of the deepinv physics table.

If the lens, the coded aperture, or the illumination can be changed, those parameters belong to `A` and are differentiated jointly with the reconstructor. Compare two optical designs with the reconstructor held fixed. A learned denoiser does not replace the detector likelihood: Poisson counts stay a Poisson data term (Bouman's model-based iterative reconstruction).

## Evaluation

- Residual `‖A(x_hat) - y‖` at the scale of the noise. Far below the noise level means noise fitting. Far above means the prior or the stopping rule is dominating.
- With ground truth and linear `A`, split the error into the row space of `A` and the null space. Quote both.
- Without ground truth, score held-out measurements (left-out k-space lines, a left-out projection angle).
- Stability probe: reconstruct `y` and `y + δ` and report `‖x(y) - x(y+δ)‖ / ‖δ‖`.
- Model error: if `A` is an approximation (coarse mesh, calibrated geometry, assumed coil maps), a residual far above the noise level on a good reconstruction points to `A`, not the prior. The approximation-error model in `../inverse-problems/references/communities.md` is the principled fix.
- PSNR and SSIM are distortion scores (`dinv.metric.PSNR`, `dinv.metric.SSIM`). Say so when the method was built to produce a posterior sample instead of a conditional mean.

## Implementation

Prefer `deepinv` over a handwritten Fourier mask or Radon transform. Confirm every class against the installed package; the reference records the 0.4.2 names.

```python
import torch
import deepinv as dinv

torch.manual_seed(0)
mask = dinv.physics.generator.GaussianMaskGenerator(img_size=(64, 64), acceleration=4).step()["mask"]
physics = dinv.physics.MRI(
    mask=mask,
    img_size=(64, 64),  # spatial size, not a batched shape
    noise_model=dinv.physics.GaussianNoise(sigma=0.01),
)
x = torch.rand(1, 2, 64, 64)  # complex image as two channels: real, then imaginary
y = physics(x)                # one noisy measurement
x0 = physics.A_dagger(y)      # least-squares start

# <A u, v> = <u, A* v> must hold before any solver uses A_adjoint.
assert abs(float(physics.adjointness_test(x))) < 1e-3
```

`img_size` on `MRI` is the spatial size, default `(320, 320)`. It is not a batched tensor shape. Simulate a small batch and check the shape of `y` before trusting a reconstruction.
