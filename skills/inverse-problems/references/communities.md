# Broader lines

Home for the plug-and-play, RED, diffusion, and Bayesian sampling communities beyond the five profiles in `lineage.md`. Cite the paper whose hypothesis the code actually uses.

Spellings used below: Marcelo Pereyra (Heriot-Watt), Julián Tachella (CNRS, DeepInverse), Martin Benning, Ulugbek Kamilov (Washington University in St. Louis), Peyman Milanfar, Michael Unser (EPFL).

## Plug-and-play

PnP replaces the proximal map of the regulariser with a denoiser and keeps the data-fidelity step. The original algorithm is PnP-ADMM: Venkatakrishnan, Bouman, and Wohlberg, "Plug-and-Play Priors for Model Based Reconstruction", IEEE GlobalSIP 2013. The survey to organise the variants is Kamilov, Bouman, Buzzard, and Wohlberg, "Plug-and-Play Methods for Integrating Physical and Learned Models in Computational Imaging", IEEE Signal Processing Magazine, 2023, doi:10.1109/MSP.2022.3199595. Bouman and Buzzard's MACE reading is that the iteration searches for an equilibrium between the sensor step and the denoiser, which is a fixed point, not automatically a minimiser of `f + λR`.

Use a subset of the measurements at each data step when the forward operator is a long sum (online PnP: Sun, Wohlberg, and Kamilov, IEEE Transactions on Computational Imaging, 2019). That is the denoiser analogue of stochastic PDHG, and it is still a point estimate.

A PnP iteration has a convergence claim only under one of these hypotheses:

- The denoiser is firmly nonexpansive (an averaged operator). Sufficient for forward-backward and ADMM in the monotone-operator sense used by Terris, Repetti, Pesquet, and Wiaux, "Building Firmly Nonexpansive Convolutional Neural Networks", ICASSP 2020. Proximal and Parseval proximal networks meet this by construction; that construction is the Neumayer entry in `lineage.md`. `FNEJacobianSpectralNorm` in `deepinv.md` is the training penalty for this hypothesis.
- The denoiser residual `Id - D` is contractive. Ryu, Liu, Wang, Chen, Wang, and Yin, "Plug-and-Play Methods Provably Converge with Properly Trained Denoisers", ICML 2019. A stock DnCNN or DRUNet does not satisfy this unless it was trained or projected to.
- The denoiser is a gradient step `D_σ = Id - ∇g_σ` on an explicit potential, possibly nonconvex. Hurault, Leclaire, and Papadakis, "Gradient Step Denoiser for convergent Plug-and-Play", ICLR 2022, and "Proximal Denoiser for Convergent Plug-and-Play Optimization with Nonconvex Regularization", ICML 2022. The iteration then targets a stationary point of an explicit functional. Keep the step inside the restriction in those papers; a proximal denoiser with step 1 forces a constraint on `λ` that their later relaxed proximal-gradient form is designed to loosen.

If none of the three holds, report a numerical residual and call the result an empirical fixed point. Chan, Wang, and Elgendy, "Plug-and-Play ADMM for Image Restoration: Fixed-Point Convergence and Applications", IEEE Transactions on Computational Imaging, 2017, is the fixed-point analysis for nonexpansive denoisers inside ADMM. The splitting toolkit underneath all of these is Combettes and Pesquet, "Proximal Splitting Methods in Signal Processing", 2011.

## RED

Romano, Elad, and Milanfar, "The Little Engine That Could: Regularization by Denoising (RED)", SIAM Journal on Imaging Sciences, 2017, replace the prior gradient by `x - D_σ(x)`. In deepinv that object is `deepinv.optim.RED`. It is the gradient of a prior only when the Jacobian of `D_σ` is symmetric. Reehorst and Schniter, "Regularization by Denoising: Clarifications and New Interpretations", IEEE Transactions on Computational Imaging, 2019, show that a generic denoiser fails this, so the iteration is a fixed-point method. Treat RED as an explicit prior gradient only after that symmetry is true, which is the case for the gradient-step denoisers above.

Milanfar's filtering line (kernel regression, a tour of modern image filtering) is a source of the denoiser `D`, not a reconstruction algorithm. The denoiser is interchangeable; the data term is not.

## Diffusion posterior sampling

A diffusion sampler draws from an approximation of `p(x | y)`, using a denoiser trained across noise levels as the score via Tweedie's formula. The SDE formulation is Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole, "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021.

Which approximation of the likelihood score, and which deepinv class, is the sampling section of `deepinv.md`. The paper names:

- Chung, Kim, McCann, Klasky, and Ye, "Diffusion Posterior Sampling for General Noisy Inverse Problems", ICLR 2023. Backpropagate the data residual through the denoiser. Works for a differentiable `A`, including nonlinear operators. Slow.
- Kawar, Elad, Ermon, and Song, "Denoising Diffusion Restoration Models", NeurIPS 2022. SVD of `A` in closed form. Only decomposable operators (blur, inpainting, single-coil MRI).
- DiffPIR (Zhu, Zhang, and coauthors) alternates a diffusion denoiser with a least-squares data step. Linear `A`.
- Kadkhodaie and Simoncelli, "Stochastic Solutions for Linear Inverse Problems using the Prior Implicit in a Denoiser", NeurIPS 2021, are the same idea without a trained diffusion: the denoiser's implicit prior, annealed, plus a data step.

One call returns one draw. The conditional mean and a pixelwise variance need several independent draws (`DiffusionSampler`). A sharp sample can score worse in PSNR than the mean. Bouman and Buzzard, "Generative Plug and Play", 2023, is the same posterior-sampling goal built by alternating proximal generators; implement it with the classes above, which exist in deepinv.

Do not call a diffusion sample a minimiser of `f + λR`.

## Bayesian sampling

Marcelo Pereyra's line targets the posterior of a convex imaging model, not its mode.

- Pereyra, "Proximal Markov chain Monte Carlo algorithms", Statistics and Computing, 2016: Langevin proposals that use a proximity map on a nonsmooth convex potential, with a Metropolis accept/reject step.
- Durmus, Moulines, and Pereyra, "Efficient Bayesian Computation by Proximal Markov Chain Monte Carlo: When Langevin Meets Moreau", SIAM Journal on Imaging Sciences, 2018: MYULA, the unadjusted version. The chain targets the Moreau-Yosida smoothing of the nonsmooth posterior, so the samples are from a smoothed model. Say that when you report them.
- Vargas, Pereyra, and Zygalakis, "Accelerating Proximal Markov Chain Monte Carlo by Using an Explicit Stabilized Method", SIAM Journal on Imaging Sciences, 2020, doi:10.1137/19M1283719: SK-ROCK. Same Langevin diffusion, Chebyshev stages instead of Euler-Maruyama, so the stable step is larger on an ill-conditioned posterior (deblurring, tomography).

deepinv exposes the Euler discretisation as `ULA` and the stabilised one as `SKRock`, built with `sampling_builder`. Both are unadjusted: there is no accept/reject step, so discretisation bias remains. Discard a burn-in, return the empirical mean and a spread, and do not quote an optimisation rate. The step size is the one in those papers for the Lipschitz constant of the smooth part; do not reuse the proximal-gradient step `1/‖A‖²` as if it were a Langevin step.

`ScorePrior` at a small fixed `σ` is the denoiser version of the same Langevin target. Diffusion annealing changes `σ` over time; ULA and SK-ROCK hold it fixed.

## Equivariant imaging and DeepInverse

Julián Tachella leads DeepInverse (the JOSS citation in `lineage.md`). Equivariant imaging is Chen, Tachella, and Davies, "Equivariant Imaging: Learning Beyond the Range Space", ICCV 2021, with the noisy extension "Robust Equivariant Imaging", CVPR 2022. The range of `A` is all a measurement-consistency loss can see. A group symmetry of the images transports information into the null space. That is what `EILoss` implements. The group has to be a symmetry of the objects. Multi-operator losses (`MOILoss`) are the alternative when each sample is seen through a different `A` and there is no group to exploit.

## Regularisation theory

Martin Benning and Martin Burger, "Modern Regularization Methods for Inverse Problems", Acta Numerica, 2018, is the survey of Tikhonov, iterative, and nonsmooth regularisation next to Arridge, Maass, Öktem, and Schönlieb 2019. Benning is part of the Cambridge bilevel and learned-sampling line already cited from Ehrhardt's and Schönlieb's profiles. Use the Acta Numerica paper when the question is what kind of regularisation a method is, rather than which network to train.

Mila Nikolova, "Analysis of the Recovery of Edges in Images and Signals by Minimizing Nonconvex Regularized Least-Squares", Multiscale Modeling and Simulation, 2005: a nonconvex penalty keeps edges that convex TV rounds off, and the objective then has local minima. That is why a weakly convex or gradient-step prior is allowed in the method table, and why a convex solver's rate does not transfer to it.

Michael Unser's representer theorem ("A Representer Theorem for Deep Neural Networks", Journal of Machine Learning Research, 2019) says a spline-activation network is the solution of a continuous-domain variational problem. Use it when someone treats a network prior as an arbitrary black box: a shallow spline potential is a regulariser with a null space you can name. His Parseval-convolution constraint (nonexpansive, energy-preserving filterbanks) is the same stability requirement as the firmly nonexpansive PnP hypothesis, applied layerwise. The convex-ridge regulariser with Neumayer and Goujon stays in `lineage.md`.

Gabriel Peyré's computational optimal transport is the data term when the mismatch is a displacement between histograms (spectral unmixing, domain shift), not a pixelwise squared error. It does not replace a Poisson likelihood on photon counts.

## Measurement design

These constrain `A` before any reconstructor. The forward-model procedure stays in the computational-imaging skill.

- Sparse MRI: Lustig, Donoho, and Pauly, Magnetic Resonance in Medicine, 2007. Variable-density random k-space, incoherent with the sparsity basis, fully sampled centre.
- Compressed sensing: Candès, Romberg, and Tao, IEEE Transactions on Information Theory, 2006. The measurement count tracks the sparsity, and the sensing basis has to be incoherent with the representation basis. A learned generative prior changes the count to the latent dimension of the generator: Bora, Jalali, Price, and Dimakis, "Compressed Sensing using Generative Models", ICML 2017. The optimisation is then over the latent code, `min_z ‖A G(z) - y‖`.
- Model-based iterative reconstruction, Charles Bouman: the noise model stays inside the iteration. A learned denoiser does not authorise a Gaussian data term on Poisson counts.
- Jeffrey Fessler, "Optimization Methods for Magnetic Resonance Image Reconstruction", IEEE Signal Processing Magazine, 2020: pick the solver from the imaging-optimisation skill after the MRI likelihood is written, rather than starting from a network.
- Designable optics. Gordon Wetzstein's deep-optics line puts lens or coding parameters inside `A` and differentiates them jointly with the reconstructor. Laura Waller's computational microscopy treats the illumination (LED array, Fourier ptychography in the sense of Zheng, Horstmeyer, and Yang, Nature Photonics, 2013, and its experimental extensions) as part of `A`. Compare optical designs with the reconstructor held fixed, as the computational-imaging skill already requires for masks.
- Rebecca Willett, with Gilton and Ongie, "Neumann Networks for Inverse Problems in Imaging": an unrolled Neumann series when `A` is a perturbation of a known invertible map. Use it as an unrolled architecture in that case, not as a generic U-Net.
- Stanley Osher: the ROF model is Rudin, Osher, and Fatemi, Physica D, 1992, and Bregman iteration is the iterative-regularisation form. The algorithm and the step bound stay in the imaging-optimisation reference.
- Stephen Boyd, Parikh, Chu, Peleato, and Eckstein, "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers", Foundations and Trends in Machine Learning, 2011, is the ADMM reference behind PnP-ADMM. The step condition for the imaging objective stays the ADMM row of the algorithms reference.

## Deep image prior

Ulyanov, Vedaldi, and Lempitsky, "Deep Image Prior", CVPR 2018, fit an untrained network to one measurement:

```
min_θ  d(A G_θ(z), y)
```

with `z` a fixed random tensor and `θ` the network weights. The architecture is the prior. There is no training set and the weights do not transfer to the next scan. Run long enough and the fit reaches the noise; stop when the residual matches the noise level. That is the discrepancy principle, the same early stopping used for Landweber and iterated Tikhonov in the Italian section below. The result is a point estimate, not a posterior draw. Confirm any `DIP` symbol against the installed deepinv before importing one; the formulation above does not depend on a class name.

## Simon Arridge and diffuse optical tomography

Simon Arridge (UCL) is a coauthor of the 2019 Acta Numerica survey already used to organise learned reconstructors. His own modelling line is optical tomography, not Fourier or Radon sampling.

- Arridge, "Optical tomography in medical imaging", Inverse Problems, 1999.
- Arridge and Schotland, "Optical tomography: forward and inverse problems", Inverse Problems, 2009.

The unknown is an absorption or scattering field in tissue. The forward map is the diffusion approximation or the radiative transfer equation, and the useful reconstruction Jacobian is the linearisation of that map about a background. Camera-image inpainting is the wrong problem statement. Hybrid measurements (photoacoustics, anatomy guiding optics) enter as a second operator or a structural prior, not as an extra channel stacked into a U-Net. Model-based learning for limited-view photoacoustics (Hauptmann and coauthors, IEEE Transactions on Medical Imaging, 2018, with Arridge) keeps that operator inside the learned iteration. If the mesh or the diffusion approximation is a deliberate simplification, account for it with the approximation-error model below rather than by inflating the detector noise.

There is no diffusion-approximation or electrical-impedance operator in the deepinv 0.4.2 physics table. Subclass `Physics`.

## Finnish inverse problems

The Finnish school treats the posterior as the object of reconstruction and requires the prior to survive refinement of the mesh. Kaipio and Somersalo, "Statistical and Computational Inverse Problems", Springer, 2005, is the working reference. The conditional mean is the estimator when the user asks for a reconstruction under that posterior; the MAP is one point of it. Report a spread from the same posterior, using Langevin or a Gaussian approximation only when the posterior is actually Gaussian.

Discretisation. A prior written as white noise on pixels changes its meaning when the grid is refined. Lassas and Siltanen, "Can one use total variation prior for edge-preserving Bayesian inversion?", Inverse Problems, 2004: a total-variation density is not discretization-invariant, and the conditional mean loses its edges under refinement. Deterministic TV minimisation, as in the imaging-optimisation skill, is unaffected and remains the convex variational estimator. For a Bayesian edge-preserving prior, use a wavelet Besov density. Lassas, Saksman, and Siltanen, "Discretization-invariant Bayesian inversion and Besov space priors", Inverse Problems and Imaging, 2009: Gaussian smoothness priors and Besov priors are discretization-invariant, and the `B^1_{11}` prior penalises an `ℓ1` norm of wavelet coefficients. Build the prior as the discretisation of that continuous density. Do not take `TVPrior` from deepinv, drop it into a sampler, and call the conditional mean a TV reconstruction.

Approximation error. Kaipio and Somersalo's approximation-error model puts the discrepancy of a reduced forward operator (coarse mesh, diffusion approximation instead of transport, wrong boundary condition) in the likelihood as an additional random variable, estimated offline from samples of the accurate model. Use it for Arridge's diffusion approximation and for a coarse EIT mesh. Adding an arbitrary constant to `σ` is not that estimate.

Electrical impedance tomography. The unknown is a conductivity and the data are a Dirichlet-to-Neumann map. Mueller and Siltanen, "Linear and Nonlinear Inverse Problems with Practical Applications", SIAM, 2012. A direct D-bar reconstruction is the baseline for the Calderón problem before any learned denoiser. The forward map is nonlinear; writing it as a matrix `A` means you have already linearised, and that linearisation has to be stated.

Learned reconstructions in this line keep the simulator in the loop when the acquisition geometry can change. A network trained as a post-processor for one scanner is the frozen-`A` case already in the inverse-problems skill.

## Italian inverse problems

Mario Bertero and Patrizia Boccacci, "Introduction to Inverse Problems in Imaging" (IOP, 1998), read a compact linear operator through its singular system. The singular values above the noise set the row space; those below it set the null space the computational-imaging evaluation already asks for. Truncated SVD and Tikhonov are filters on those singular values. Plot them before training a network for a linear problem.

Iterative regularisation. Landweber and iterated Tikhonov are semi-convergent: the early iterates reconstruct, the late iterates fit noise. Stop when the residual reaches the noise level (discrepancy principle). Deep image prior uses this stopping rule with the iteration index replaced by the number of optimiser steps on `θ`.

Deblurring boundary conditions are part of `A`. Zero Dirichlet, periodic, reflective, and anti-reflective conditions produce different matrices and different boundary artefacts. Donatelli and Serra-Capizzano, "Anti-reflective boundary conditions and re-blurring", Inverse Problems, 2005, and Donatelli, Estatico, Martinelli, and Serra-Capizzano, Inverse Problems, 2006: anti-reflective conditions give `C^1` matching at the border and, with re-blurring, a fast solve. Use them when the object reaches the edge of the frame. Claudio Estatico is part of this Genoa–Insubria numerical line.

The Bologna restoration group (Fiorella Sgallari, Serena Morigi, Alessandro Lanza) is the applied form of nonconvex edge-preserving penalties, including automatic choice of the regularisation weight. Elena Loli Piccolomini's group keeps the Radon or cone-beam projector as `A` for limited-data CT and puts an edge-preserving or learned regulariser in the variational problem. Massimo Fornasier connects this tradition to sparsity and to learning sparse representations. None of these replaces the singular-value check or the discrepancy stop.

Daniela Calvetti and Erkki Somersalo, "Introduction to Bayesian Scientific Computing" (Springer, 2007), are the Italian–Finnish bridge on computation: conditionally Gaussian priors with hyperpriors on the noise variance and on the regularisation weight. Use a hyperprior when `λ` and `σ` are unknown and the deliverable is a posterior. Fixing them by hand and then sampling is a different, narrower posterior.
