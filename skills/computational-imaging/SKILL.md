---
name: computational-imaging
description: >
  Formulate, simulate, and evaluate a computational imaging system whose
  measurements are not the image: MRI, CT, PET, ultrasound, phase retrieval,
  ptychography, spectral imaging, deblurring, inpainting, and compressed
  sensing. Covers forward operators, noise, sampling design, structural
  priors, and null-space evaluation, from the variational line and from
  sparse MRI, compressed sensing, and designable optics (Lustig, Candès,
  Bouman, Fessler, Waller, Wetzstein). Implemented in deepinv. Use when
  the user says computational imaging, forward model, k-space, tomography,
  compressed sensing, diffuse optical tomography, electrical impedance
  tomography, measurement operator, or runs /computational-imaging.
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
- PSNR and SSIM are distortion scores. Say so when the method was built to produce a posterior sample instead of a conditional mean.

## Implementation

Prefer `deepinv` over a handwritten Fourier mask or Radon transform. Confirm every class against the installed package; the reference records the 0.4.2 names.

```python
import deepinv as dinv

physics = dinv.physics.MRI(
    mask=mask,  # (H, W) or broadcastable to (B, 2, H, W)
    noise_model=dinv.physics.GaussianNoise(sigma=0.01),
)
y = physics(x)            # x is (B, 2, H, W): real, then imaginary
z = physics.A(x)          # noiseless forward map
x0 = physics.A_dagger(y)  # least-squares start
```

`img_size` on `MRI` is the spatial size, default `(320, 320)`. It is not a batched tensor shape. Simulate a small batch, check the shape of `y`, and check `A_adjoint` against `<A x, y> ≈ <x, A* y>` before trusting a reconstruction.
