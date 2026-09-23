# Forward models

Lookup for the computational-imaging skill. Class names match DeepInverse 0.4.2. Confirm them in the installed package. Docs: https://deepinv.org/user_guide/physics/physics.html

The measurement model is `y = N(A(x))`. In deepinv, `A` is a `deepinv.physics.Physics` and `N` is attached with the noise-model argument. `physics(x)` draws one noisy measurement. `physics.A(x)` is the noiseless operator. Linear operators expose `A_adjoint` and `A_dagger`.

Build a custom operator by subclassing `Physics` or `LinearPhysics` and implementing `A` and, for linear maps, `A_adjoint`. Check a hand-written adjoint with `physics.adjointness_test(x)`, which returns a number that should be at the level of floating-point precision. Automatic differentiation through `A` and through operator parameters is how blind problems and learned sampling are implemented. Guide: https://deepinv.org/auto_examples/basics/demo_custom_physics.html

## Operators

| Family | Classes |
| --- | --- |
| Pixelwise | `Denoising`, `Inpainting`, `Demosaicing`, `Decolorize`, `SpatialUnwrapping` |
| Blur and scale | `Blur`, `BlurFFT`, `SpaceVaryingBlur`, `TiledSpaceVaryingBlur`, `Downsampling`, `Upsampling`, `DownsamplingMatlab` |
| MRI | `MRI`, `MultiCoilMRI`, `DynamicMRI`, `SequentialMRI` (Cartesian; `three_d=True` for 3D). Non-Cartesian (radial, spiral) sampling has no class in 0.4.2: wrap a NUFFT from `mri-nufft` or `torchkbnufft` as a `LinearPhysics`. |
| Tomography | `Tomography` (parallel beam, or fan beam with `fan_beam=True`), `TomographyWithAstra` (measured cone-beam, fan-beam, and electron-tomography geometries), `XrayTransform`, `PET` (needs `parallelproj`) |
| Wave | `RadioInterferometry`, `SinglePhotonLidar`, `Scattering`. There is no ultrasound operator in 0.4.2. |
| Spectral | `Pansharpen`, `HyperSpectralUnmixing`, `CompressiveSpectralImaging` |
| Compressive | `CompressedSensing`, `StructuredRandom`, `SinglePixelCamera` |
| Phase | `PhaseRetrieval`, `RandomPhaseRetrieval`, `StructuredRandomPhaseRetrieval`, `Ptychography`, `PtychographyLinearOperator` |
| Other | `Haze` |
| Wrappers | `PhysicsMultiScaler`, `LinearPhysicsMultiScaler`, `PhysicsCropper` |

Mask and kernel generators live in `deepinv.physics.generator`: `BernoulliSplittingMaskGenerator`, `GaussianSplittingMaskGenerator`, `GaussianMaskGenerator`, `RandomMaskGenerator`, `EquispacedMaskGenerator`, `PolyOrderMaskGenerator` (MRI, including k-t), `GaussianBlurGenerator`, `MotionBlurGenerator`, `DiffractionBlurGenerator`. Phase helpers: `deepinv.physics.phase_retrieval.build_probe`, `generate_shifts`.

## Noise

Set the noise on any operator. The default is `ZeroNoise`.

| Class | Use when the data are |
| --- | --- |
| `GaussianNoise(sigma)` | Thermal or complex k-space noise. Data fidelity `deepinv.optim.L2`. |
| `PoissonNoise` | Photon counts (PET, low-dose CT pre-log, photon-limited optical). Data fidelity `PoissonLikelihood`. Keep nonnegative intensities. |
| `PoissonGaussianNoise` | Sensor with photon noise plus read noise. Data fidelity `SurePGLoss` at training time; weighted least squares is the usual variational surrogate. |
| `RicianNoise` | Magnitude MRI. Prefer fitting the complex Gaussian in k-space when the raw data exist. |
| `LogPoissonNoise` | Log-intensity Poisson models. Data fidelity `LogPoissonLikelihood`. |
| `GammaNoise`, `FisherTippettNoise` | Speckle, including log-domain ultrasound and SAR. |
| `LaplaceNoise` | Impulsive residuals. Data fidelity `L1`. |
| `SaltPepperNoise` | Saturated impulse noise. Data fidelity `L1`. |
| `UniformNoise` | Quantisation-style bounded noise. |

## Modality defaults

MRI. `A` is a partial Fourier operator, multi-coil when sensitivity maps exist (`MultiCoilMRI`). Single-coil `MRI` stores a complex image as two channels, shape `(B, 2, ..., H, W)`, real then imaginary. `img_size` is the spatial size (default `(320, 320)`), and `mask` broadcasts to that layout. Noise is complex Gaussian in k-space. A Cartesian mask keeps a fully sampled low-frequency block and a variable-density or equispaced high-frequency pattern (`GaussianMaskGenerator`, `EquispacedMaskGenerator`, `PolyOrderMaskGenerator`). The null space is the unsampled Fourier coefficients. The evaluation procedure in `SKILL.md` asks for the error of `x_hat` on those coefficients, not only image-domain PSNR.

CT. `Tomography` is the parallel-beam ray transform. `TomographyWithAstra` is the ASTRA backend for cone-beam, fan-beam, and electron-tomography geometries that have to match a measured scan. CIL is the variational layer on that same projector. The library choice and the citations are in `../../inverse-problems/references/libraries.md`. Angle splits and discrete materials are in the measured-tomography part of `../../inverse-problems/references/communities.md`. Limited angle produces a directional null space; acquiring the missing angles changes the problem more than a stronger prior. Post-log data are often treated with weighted least squares; pre-log data stay Poisson.

PET. `PET` (built on `parallelproj`), Poisson likelihood, nonnegative image. Solvers: `MLEM` and `OSEM` in `deepinv.optim`. Passing a `prior` to either turns it into Green's one-step-late MAP-EM, a heuristic with no convergence proof whose denominator must stay positive, so keep `lambda_reg` small or use a convergent penalised solver from the imaging-optimisation skill. A co-registered MRI or CT enters as a parallel-level-sets structural prior (edge directions shared), which is Ehrhardt's joint-reconstruction model. Parallel level sets are the right coupling when edges should align and contrasts may differ. Channel-stacked total variation is the right coupling when the contrasts themselves are proportional.

Ultrasound. No operator ships in 0.4.2; subclass `Physics` with the beamforming model you actually use. Speckle is not Gaussian; pick `GammaNoise` or `FisherTippettNoise` to match the beamforming domain.

Phase retrieval and ptychography. The map is nonlinear (modulus of a Fourier or windowed Fourier transform). Data fidelity in deepinv for the amplitude residual is `AmplitudeLoss`. Local minima are part of the problem; initialise from a spectral or autocorrelation method and report more than one random start.

Compressed sensing and single-pixel. `CompressedSensing`, `StructuredRandom`, `SinglePixelCamera`. Sparsity is a property of a basis (`WaveletPrior`, `L1Prior` on coefficients), and the number of measurements has to match that sparsity. A learned denoiser prior does not remove the need to state the measurement budget.

Inpainting, demosaicing, deblurring, super-resolution. Pixelwise or blur classes above. Deblurring kernels from `GaussianBlurGenerator` or `MotionBlurGenerator`. `MLEM` on a `Blur` operator is Richardson–Lucy, the Poisson deconvolution baseline. A blind kernel has no dedicated class in 0.4.2: make the kernel a parameter of `A` and differentiate it jointly with the image, and report the kernel as well as the image, since the pair is only determined up to the blur–image ambiguity.

## Sampling as a design variable

A mask, an angle set, or a coil sensitivity is a parameter of `A`. To learn it, relax the discrete constraint during training (continuous weights on candidate k-space lines or views), penalise the measurement budget, and project back onto the feasible design at evaluation. Ehrhardt's "Learning the sampling pattern for MRI" is the bilevel version of this: the inner problem reconstructs, the outer problem scores the mask. Keep the reconstruction method fixed while comparing masks, otherwise the mask and the network trade off in a way you cannot interpret.

Evaluation (residual against the noise level, null-space versus row-space error, held-out measurements, stability probe, distortion versus a posterior sample) is the procedure in `SKILL.md`.
