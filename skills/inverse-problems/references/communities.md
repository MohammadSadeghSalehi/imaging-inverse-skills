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

## Unrolled reconstructions

Ozan Öktem's line, with Jonas Adler, is the unrolling pattern for tomography. An unrolled layer still evaluates the forward operator and its adjoint. A network applied to a filtered back-projection, with no further call to `A`, is post-processing. It does not follow a change of dose, angles, or mask. Arridge, Maass, Öktem, and Schönlieb 2019 is the survey that separates these slots. The constructions:

- Learned iterative gradient scheme. Adler and Öktem, "Solving ill-posed inverse problems using iterative deep neural networks", Inverse Problems, 2017. Each layer is a learned step that is given `A` and `A*`.
- Learned primal-dual. Adler and Öktem, "Learned Primal-Dual Reconstruction", IEEE Transactions on Medical Imaging, 2018, doi:10.1109/TMI.2018.2799231. Unroll a proximal primal-dual iteration, replace both proximal maps by convolutional networks, and keep the forward and back-projection. Use a small fixed depth. Their CT experiments use ten iterations. In deepinv the blocks are `PDNet_PrimalBlock` and `PDNet_DualBlock`.
- Variational network (Hammernik, Pock, and coauthors, already in `lineage.md`): the MRI case, unrolled gradient steps on a Fields-of-Experts energy.
- MoDL (`deepinv.models.MoDL`): a shared denoiser alternating with a data-consistency least-squares step. The data-consistency step is where `A` stays exact.
- LISTA (Gregor and LeCun, ICML 2010) is the sparse-coding origin of the same idea: unroll ISTA and learn the step.

Train the unroll on paired data whose operator matches test time. If `A` will change, the layer must read `physics` rather than bake one projector into the weights. A long unroll is differentiated either through the stored steps or, once the iteration has a fixed point, by the implicit-function route in the imaging-optimisation reference. Differentiating a fixed number of iterations is also how a nonsmooth variational model gets its parameters. That use is the bilevel section of `../../imaging-optimisation/references/algorithms.md`.

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

with `z` a fixed random tensor and `θ` the network weights. The architecture is the prior. There is no training set and the weights do not transfer to the next scan. Run long enough and the fit reaches the noise; stop when the residual matches the noise level. That is the discrepancy principle, the same early stopping used for Landweber and iterated Tikhonov in the iterative-regularisation section below. The result is a point estimate, not a posterior draw. Confirm any `DIP` symbol against the installed deepinv before importing one; the formulation above does not depend on a class name.

## Simon Arridge and diffuse optical tomography

Simon Arridge (UCL) is a coauthor of the 2019 Acta Numerica survey already used to organise learned reconstructors. His own modelling line is optical tomography, not Fourier or Radon sampling.

- Arridge, "Optical tomography in medical imaging", Inverse Problems, 1999.
- Arridge and Schotland, "Optical tomography: forward and inverse problems", Inverse Problems, 2009.

The unknown is an absorption or scattering field in tissue. The forward map is the diffusion approximation or the radiative transfer equation, and the useful reconstruction Jacobian is the linearisation of that map about a background. Camera-image inpainting is the wrong problem statement. Hybrid measurements (photoacoustics, anatomy guiding optics) enter as a second operator or a structural prior, not as an extra channel stacked into a U-Net. Model-based learning for limited-view photoacoustics (Hauptmann and coauthors, IEEE Transactions on Medical Imaging, 2018, with Arridge) keeps that operator inside the learned iteration. If the mesh or the diffusion approximation is a deliberate simplification, account for it with the approximation-error model below rather than by inflating the detector noise.

There is no diffusion-approximation or electrical-impedance operator in the deepinv 0.4.2 physics table. Subclass `Physics`.

## Statistical inversion, impedance tomography, and limited-angle X-ray

The posterior is the object of reconstruction, and the prior has to survive refinement of the mesh. Kaipio and Somersalo, "Statistical and Computational Inverse Problems", Springer, 2005, is the working reference. The conditional mean is the estimator when the user asks for a reconstruction under that posterior; the MAP is one point of it. Report a spread from the same posterior, using Langevin or a Gaussian approximation only when the posterior is actually Gaussian.

Discretisation. A prior written as white noise on pixels changes its meaning when the grid is refined. Lassas and Siltanen, "Can one use total variation prior for edge-preserving Bayesian inversion?", Inverse Problems, 2004: a total-variation density is not discretization-invariant, and the conditional mean loses its edges under refinement. Deterministic TV minimisation, as in the imaging-optimisation skill, is unaffected and remains the convex variational estimator. For a Bayesian edge-preserving prior, use a wavelet Besov density. Lassas, Saksman, and Siltanen, "Discretization-invariant Bayesian inversion and Besov space priors", Inverse Problems and Imaging, 2009: Gaussian smoothness priors and Besov priors are discretization-invariant, and the `B^1_{11}` prior penalises an `ℓ1` norm of wavelet coefficients. Build the prior as the discretisation of that continuous density. Do not take `TVPrior` from deepinv, drop it into a sampler, and call the conditional mean a TV reconstruction.

Approximation error. Kaipio and Somersalo's approximation-error model puts the discrepancy of a reduced forward operator (coarse mesh, diffusion approximation instead of transport, wrong boundary condition) in the likelihood as an additional random variable, estimated offline from samples of the accurate model. Use it for Arridge's diffusion approximation and for a coarse EIT mesh. Adding an arbitrary constant to `σ` is not that estimate.

Electrical impedance tomography. The unknown is a conductivity and the data are a Dirichlet-to-Neumann map. The forward map is nonlinear; a matrix `A` means the problem has already been linearised, and that linearisation has to be stated. The baseline is a regularised D-bar method, not a variational iteration. Siltanen, Mueller, and Isaacson, Inverse Problems, 2000, is the first robust implementation of Nachman's reconstruction. Knudsen, Lassas, Mueller, and Siltanen, Inverse Problems and Imaging, 2009, is the regularised nonlinear Fourier version, with a proved regularisation property. Mueller and Siltanen, Inverse Problems, 2020, is the practical review. Code and the worked examples of their SIAM 2012 book are at https://siltanen-research.net/open-software/ and https://github.com/ssiltane/Dbar_method_for_EIT. Low-pass filtering of the nonlinear Fourier transform is the regularisation parameter. A variational solver is what you use after this baseline, or when the measurement model leaves the two-dimensional Calderón setting the D-bar theory covers. Hamilton, Hauptmann, and Siltanen, Inverse Problems and Imaging, 2014, is the edge-preserving, data-driven correction of that low-pass D-bar image.

Sparse and dynamic X-ray tomography in this line is an explicit optimisation problem with a sparsity or motion penalty, not a post-hoc denoiser. Hämäläinen, Kallonen, Kolehmainen, Lassas, Niinimäki, and Siltanen, SIAM Journal on Scientific Computing, 2013. The open walnut tomographic dataset from that group is the test to use before claiming a new limited-data CT method.

Limited-angle tomography, Tatiana Bubba with Siltanen. Microlocal analysis says which edges the angular range can determine and which it cannot. Bubba, Kutyniok, Lassas, März, Samek, Siltanen, and Srinivasan, "Learning the invisible", Inverse Problems, 2019: recover the visible wavefront by weighted shearlet sparsity, and train a network only on the invisible shearlet coefficients. Do not ask one network to invent the whole image. Bubba, Galinier, Lassas, Prato, Ratti, and Siltanen, SIAM Journal on Imaging Sciences, 2021 (ΨDONet): the learned piece is a pseudodifferential correction of `AᵀA`, concentrated near the diagonal of the normal operator, not a replacement of the projector. Bubba and Ratti, shearlet `ℓᵖ` regularisation for randomly sampled angles, extends the same sparsity penalty into the statistical-learning rate statement. A learned post-processor with no visible/invisible split is the weaker option for this geometry.

Learned reconstructions in this line keep the simulator in the loop when the acquisition geometry can change. A network trained as a post-processor for one scanner is the frozen-`A` case already in the inverse-problems skill.

## Spectral filtering, boundary conditions, and iterative regularisation

Mario Bertero and Patrizia Boccacci, "Introduction to Inverse Problems in Imaging" (IOP, 1998), read a compact linear operator through its singular system. The singular values above the noise set the row space; those below it set the null space the computational-imaging evaluation already asks for. Truncated SVD and Tikhonov are filters on those singular values. Plot them before training a network for a linear problem.

Iterative regularisation. Landweber and iterated Tikhonov are semi-convergent: the early iterates reconstruct, the late iterates fit noise. Stop when the residual reaches the noise level (discrepancy principle). Deep image prior uses this stopping rule with the iteration index replaced by the number of optimiser steps on `θ`.

Deblurring boundary conditions are part of `A`. Zero Dirichlet, periodic, reflective, and anti-reflective conditions produce different matrices and different boundary artefacts. Donatelli and Serra-Capizzano, "Anti-reflective boundary conditions and re-blurring", Inverse Problems, 2005, and Donatelli, Estatico, Martinelli, and Serra-Capizzano, Inverse Problems, 2006: anti-reflective conditions give `C^1` matching at the border and, with re-blurring, a fast solve. Use them when the object reaches the edge of the frame. Claudio Estatico is part of this Genoa–Insubria numerical line.

The Bologna restoration group (Fiorella Sgallari, Serena Morigi, Alessandro Lanza) is the applied form of nonconvex edge-preserving penalties, including automatic choice of the regularisation weight. Elena Loli Piccolomini's group keeps the Radon or cone-beam projector as `A` for limited-data CT and puts an edge-preserving or learned regulariser in the variational problem. Massimo Fornasier connects this tradition to sparsity and to learning sparse representations. None of these replaces the singular-value check or the discrepancy stop.

Daniela Calvetti and Erkki Somersalo, "Introduction to Bayesian Scientific Computing" (Springer, 2007): conditionally Gaussian priors with hyperpriors on the noise variance and on the regularisation weight. Use a hyperprior when `λ` and `σ` are unknown and the deliverable is a posterior. Fixing them by hand and then sampling is a different, narrower posterior.

Luca Calatroni, Saverio Salzo, and Silvia Villa, with Lorenzo Rosasco, treat the iteration count as the regularisation parameter. Garrigos, Rosasco, and Villa, "Iterative regularization via dual diagonal descent", Journal of Mathematical Imaging and Vision, 2018: a dual proximal method, stopped early, regularises a linear inverse problem for a general convex penalty and a general data term. Calatroni, Garrigos, Rosasco, and Villa, "Accelerated iterative regularization via dual diagonal descent", SIAM Journal on Optimization, 2021: the inertial version, with an early-stopping rate that covers Kullback–Leibler as well as an additive fidelity. Use this when you do not want to grid-search `λ`, and when the noise is Poisson rather than Gaussian. The discrepancy stop already stated for Landweber is the stopping rule. Salzo and Villa, "Inexact and accelerated proximal point algorithms", Journal of Convex Analysis, 2012: if the prox itself is iterative, as it is for total variation, the outer `1/k²` rate survives only for prox errors of the type proved there. An arbitrary fixed number of inner iterations is not that control. Calatroni, De Los Reyes, and Schönlieb, infimal convolution of L2 and L1 fidelities (SIAM Journal on Imaging Sciences, 2017): Gaussian noise plus outliers is one fidelity built as an infimal convolution, not a compromise weight on a single L2 term.

## PDE-constrained coefficients and measured tomography

Tristan van Leeuwen works on PDE-constrained coefficient estimation, not on pixel denoising. The unknown is a coefficient of a PDE, typically a wavespeed, and the data are partial observations of the PDE solution for several sources. The reduced formulation solves the PDE exactly at every coefficient update. That is full-waveform inversion, and it has local minima when the initial coefficient is far from the truth. The penalty formulation enlarges the unknown to the coefficient and the fields:

```
min_{m,u}  ½‖P u − d‖² + (λ/2) ‖A(m) u − q‖²
```

van Leeuwen and Herrmann, "Mitigating local minima in full-waveform inversion by expanding the search space", Geophysical Journal International, 2013, is the seismic form, called wavefield reconstruction inversion. van Leeuwen and Herrmann, "A penalty method for PDE-constrained optimization in inverse problems", Inverse Problems, 2016, doi:10.1088/0266-5611/32/1/015007, is the general algorithm: eliminate the field from the quadratic penalty and update the coefficient, at about the cost of one reduced iteration, on a less nonlinear objective. Raise `λ` only as the field residual allows. van Leeuwen and Yang, Inverse Problems, doi:10.1088/1361-6420/adab89, identify the relaxed problem with a weighted nonlinear least-squares problem whose weight is the Gram matrix of PDE solutions. There is no wave-equation operator in the deepinv 0.4.2 physics table. Subclass `Physics`. A denoiser on a migrated image is a different problem, applied after this inversion.

Measured tomography uses a projector that matches the scanner. The projector for cone-beam, fan-beam, and electron tomography is the ASTRA toolbox: van Aarle, Palenstijn, De Beenhouwer, Altantzis, Bals, Batenburg, and Sijbers, Ultramicroscopy, 2015, and van Aarle, Palenstijn, Cant, Janssens, Bleichrodt, Dabravolski, De Beenhouwer, Batenburg, and Sijbers, Optics Express, 2016. In deepinv that backend is `TomographyWithAstra`. Use it when the geometry has to match a measured scan. The default `Tomography` class is the parallel-beam ray transform. ASTRA's iterative algorithms include SIRT, SART, and CGLS. deepinv's `SIRT` is the same simultaneous-iterative family. Test a new CT method on measured projections, not only on a synthetic ray transform. The FleX-ray lab at CWI is that measured-data setting.

When the unknown takes a few known material values, use discrete algebraic reconstruction. Batenburg and Sijbers, "DART: A Practical Reconstruction Algorithm for Discrete Tomography", IEEE Transactions on Image Processing, 2011. Continuous TV on those pixels is the wrong prior.

When the only data are noisy projections, split angles rather than pixels. Hendriksen, Pelt, and Batenburg, "Noise2Inverse: Self-Supervised Deep Convolutional Denoising for Tomography", IEEE Transactions on Computational Imaging, 2020, doi:10.1109/TCI.2020.3019647: reconstruct statistically independent subsets of the projections and train a denoiser to map one reconstruction to another. The noise must be elementwise independent and zero-mean across the split. In DeepInverse the general mechanism is `SplittingLoss` with the split built into the projector. There is no `Noise2Inverse` class in the 0.4.2 table. LION implements that angle split as `Noise2Inverse_solver`, and the library choice is `libraries.md`. The small network used with this line is the mixed-scale dense network, Pelt and Sethian, PNAS, 2018, which is the right capacity when the training set is one scan. A pixelwise `Neighbor2Neighbor` loss on the reconstructed image is the wrong split, because the reconstruction correlates the noise. The FleX-ray lab is a measured-projection test for these methods.
