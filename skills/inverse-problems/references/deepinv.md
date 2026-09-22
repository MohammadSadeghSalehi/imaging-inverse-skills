# DeepInverse reconstruction API

Lookup for the inverse-problems skill. Names match the 0.4.2 docs at https://deepinv.org. Confirm every symbol in the installed package before using it. Physics classes and noise models live in `../../computational-imaging/references/forward-models.md`. Algorithm step-size rules live in `../../imaging-optimisation/references/algorithms.md`.

Install: `pip install deepinv` (Python 3.10+). Optional extras: `pip install "deepinv[dataset,denoisers]"`. Cite Tachella et al., JOSS, 2025, doi:10.21105/joss.08923.

Every reconstructor subclasses `deepinv.models.Reconstructor` and maps `(y, physics)` to `x_hat`.

```python
import deepinv as dinv
import torch

physics = dinv.physics.Inpainting(
    (1, 512, 512), mask=0.5, noise_model=dinv.physics.GaussianNoise(sigma=0.01)
)
data_fidelity = dinv.optim.L2()
prior = dinv.optim.PnP(denoiser=dinv.models.MedianFilter())
norm_A2 = physics.compute_norm(y, tol=1e-4, verbose=False).item()
model = dinv.optim.PGD(
    prior=prior,
    data_fidelity=data_fidelity,
    stepsize=0.99 / norm_A2,
    lambda_reg=0.1,
    sigma_denoiser=0.1,
    max_iter=100,
)
x_hat = model(y, physics)
```

`compute_norm` estimates `||A||^2` by power iteration. Use it before choosing a step. The legal step for each algorithm is in the imaging-optimisation reference.

## Data fidelity

`d(A(x), y)`, class `deepinv.optim.DataFidelity`.

| Class | Matches |
| --- | --- |
| `L2` | Gaussian, `‖A(x) - y‖²₂` |
| `L1` | Laplace or impulse, `‖A(x) - y‖₁` |
| `IndicatorL2` | Hard constraint `‖A(x) - y‖₂ ≤ ε` |
| `PoissonLikelihood` | Poisson counts, `-yᵀ log(A(x) + β) + 1ᵀ A(x)` |
| `LogPoissonLikelihood` | Log-domain Poisson |
| `AmplitudeLoss` | Phase retrieval amplitude residual |
| `ItohFidelity` | Wrapped-phase finite differences |
| `ZeroFidelity` | Prior projection, no data term |

## Priors

| Class | Role | Explicit cost |
| --- | --- | --- |
| `Tikhonov` | `‖x‖²₂` | yes |
| `L1Prior` | `‖x‖₁` | yes |
| `WaveletPrior` | `‖Ψx‖_p` | yes |
| `TVPrior` | isotropic TV, `‖Dx‖_{1,2}` | yes |
| `TVL1Prior` | anisotropic TV | yes |
| `L12Prior` | group sparsity, `∑ ‖x_i‖₂` | yes |
| `PatchPrior`, `PatchNR`, `EPLL` | patch models | yes |
| `PnP` | `prox` replaced by a denoiser `D_σ` | no |
| `RED` | gradient of the prior replaced by `x - D_σ(x)` | no |
| `ScorePrior` | score ` (x - D_σ(x)) / σ² ` | no |
| `ZeroPrior` | no regulariser | yes |

`lambda_reg` multiplies the prior. `sigma_denoiser` (also `g_param`) is the denoiser noise level. Backtracking needs an explicit cost, so it switches off for `PnP`, `RED`, and `ScorePrior`.

Prebuilt iterative methods: `DPIR` (plug-and-play with a decreasing noise schedule), `EPLL`.

## Unfolding

Pass `unfold=True` and `trainable_params` (any of `stepsize`, `lambda_reg`, `sigma_denoiser`, and the denoiser weights) to `PGD`, `HQS`, `ADMM`, and the other `BaseOptim` solvers. Keep `max_iter` small and fixed.

```python
model = dinv.optim.PGD(
    unfold=True,
    data_fidelity=dinv.optim.L2(),
    prior=dinv.optim.PnP(dinv.models.DnCNN()),
    stepsize=1.0,
    sigma_denoiser=0.1,
    lambda_reg=1.0,
    max_iter=5,
    trainable_params=["stepsize", "sigma_denoiser", "lambda_reg"],
)
```

MRI-specific unrolled models: `deepinv.models.VarNet`, `deepinv.models.MoDL`. Primal-dual network blocks: `PDNet_PrimalBlock`, `PDNet_DualBlock`.

For HQS or ADMM, construct linear physics with `implicit_backward_solver=True` so the least-squares prox does not store every inner iteration on the backward pass. Raise that inner `max_iter` until the prox has actually converged, because the closed-form gradient assumes a minimiser.

Deep equilibrium is available for `GD`, `PGD`, and `HQS` by passing `DEQ` as a `deepinv.optim.DEQConfig`. The forward pass runs to a fixed point and the backward pass solves the implicit-function linear system. Anderson acceleration flags are fields of `DEQConfig`.

## Losses

Supervised: `deepinv.loss.SupLoss` with a distortion metric, on paired `(x, y)`.

Self-supervised, from measurements alone:

| Loss | Requires |
| --- | --- |
| `SureGaussianLoss` | Gaussian noise, known `σ`, general `A` |
| `SurePoissonLoss` | Poisson noise |
| `SurePGLoss` | Poisson-Gaussian noise |
| `R2RLoss` | Poisson, Gaussian, or Gamma; recorrupted measurements |
| `SplittingLoss` | Noise independent across measurements; works with a general `A` |
| `Neighbor2Neighbor` | Independent pixel noise; denoising |
| `EILoss` | Image distribution invariant under a stated group (equivariant imaging) |
| `MOILoss` | Several different operators observed |
| `MOEILoss` | Several operators and a group symmetry |
| `TVLoss` | Piecewise-smooth images, used as a training penalty |
| `mri.WeightedSplittingLoss` | Accelerated MRI, split k-space |
| `mri.RobustSplittingLoss` | Noisy MRI |
| `mri.Phase2PhaseLoss`, `mri.Artifact2ArtifactLoss` | Dynamic or sequential MRI, split along time |
| `mri.ENSURELoss` | Gaussian SURE with rank-deficient multi-operator MRI |

The group passed to `EILoss` must be a symmetry of the objects. A rotation group is appropriate for natural images and cells, and inappropriate for an oriented anatomical radiograph whose pose carries meaning.

Adversarial generator/discriminator losses (`SupAdversarialGeneratorLoss`, `UnsupAdversarialGeneratorLoss`, `UAIRGeneratorLoss`) train a network that outputs an image. They are not the Lunz–Öktem–Schönlieb adversarial regulariser, which is a Lipschitz functional inside a variational objective.

Lipschitz control during training: `JacobianSpectralNorm`. Firm nonexpansiveness: `FNEJacobianSpectralNorm`. Use these when the trained denoiser will be dropped into plug-and-play and you need the averaged-operator hypothesis.

## Sampling

Class names for posterior samplers in deepinv 0.4.2. Which scientific claim each class supports is in `communities.md`. Docs: https://deepinv.org/user_guide/reconstruction/sampling.html

Prebuilt diffusion reconstructors, one draw per call:

| Class | Use when |
| --- | --- |
| `deepinv.sampling.DDRM` | `A` is decomposable (blur, inpainting, single-coil MRI): it needs the SVD. |
| `deepinv.sampling.DiffPIR` | `A` is linear. Denoiser step, then a least-squares data step. |
| `deepinv.sampling.DPS` | `A` is any differentiable operator, including nonlinear. Backpropagates through the denoiser. |

A custom likelihood score is a `NoisyDataFidelity` inside `deepinv.sampling.PosteriorDiffusion`: `DPSDataFidelity`, `PiGDMDataFidelity` (linear Gaussian noise), `MomentMatchingDataFidelity`, `ALDDataFidelity` (no backward through the denoiser), `ScoreSDEDataFidelity`, `ILVRDataFidelity`. The SDE is `VarianceExplodingDiffusion` or `VariancePreservingDiffusion`, integrated with `EulerSolver` or `HeunSolver`. Several draws, and the mean and variance across them, come from `DiffusionSampler`.

MCMC holds the noise level fixed. Build it with `deepinv.sampling.sampling_builder(iterator="ULA"` or `"SKRock"`, `prior`, `data_fidelity`, `params_algo`, `max_iter)`. `ULA` takes `step_size`, `alpha`, `sigma`. `SKRock` adds `inner_iter` and `eta`. The prior is `ScorePrior` for a denoiser score, or an explicit potential. `ULA` is Euler-Maruyama and unadjusted. `SKRock` is the stabilised discretisation. Neither is a proximal-gradient optimiser, and neither applies the step `1/‖A‖²` from the algorithms reference.

## Training loop

Use `deepinv.Trainer` when the installed version provides it. Read `Trainer.__init__` in the installed package and pass `model`, `physics`, `optimizer`, `losses`, and the dataloaders by those names. Docs: https://deepinv.org/user_guide/training/trainer.html

Pretrained reconstructors and denoisers are listed at https://deepinv.org/user_guide/reconstruction/pretrained-models.html and https://deepinv.org/user_guide/reconstruction/denoisers.html. Load one before training a new network for a standard denoising or MRI problem.
