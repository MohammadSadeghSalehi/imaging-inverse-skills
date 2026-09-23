# Libraries

Which package implements the operator and the solver. Confirm every symbol in the installed version before writing code. DeepInverse class names for version 0.4.2 stay in `deepinv.md`. Step-size conditions stay in `../../imaging-optimisation/references/algorithms.md`.

| You have | Use |
| --- | --- |
| A general imaging operator: MRI, blur, inpainting, phase retrieval, ptychography, a learned denoiser, a diffusion sampler, or a self-supervised loss | DeepInverse. `pip install deepinv`. https://github.com/deepinv/deepinv |
| A measured cone-beam, fan-beam, parallel-beam, or electron-tomography geometry, and you need the projector or FBP, SIRT, SART, CGLS | ASTRA. `pip install astra-toolbox`. https://github.com/astra-toolbox/astra-toolbox |
| The same tomography geometry, plus an explicit variational solver (TV, Kullback–Leibler, subset PDHG) or a real scanner data container | CIL, with ASTRA as its projector. `pip install cil`. https://github.com/TomographicImaging/CIL |
| A learned CT experiment: learned primal-dual, learned gradient, FBPConvNet, Noise2Inverse, or a mixed-scale dense network, with a fixed dataset and a saved run | LION. It calls ASTRA through tomosipo and can call DeepInverse for a pretrained denoiser. https://github.com/CambridgeCIA/LION |

Use one library for the projector. A second copy of the same ray transform in another package will not match voxel for voxel.

## ASTRA

GPU projectors and the classical iterative algorithms. Cite van Aarle, Palenstijn, Cant, Janssens, Bleichrodt, Dabravolski, De Beenhouwer, Batenburg, and Sijbers, Optics Express, 2016, doi:10.1364/OE.24.025129, when the projector is ASTRA. Build `vol_geom` and `proj_geom` to the scan, then `astra.create_projector`. Forward projection is `astra.create_sino`. Back projection is `astra.create_backprojection`. FBP, SIRT, SART, and CGLS are `astra.algorithm` configs. DeepInverse exposes this backend as `deepinv.physics.TomographyWithAstra`. CIL exposes it as `cil.plugins.astra.operators.ProjectionOperator`.

## CIL

The Core Imaging Library wraps that projector in an optimisation object. Cite Jørgensen and coauthors, "Core Imaging Library, Part I", Philosophical Transactions of the Royal Society A, 2021, doi:10.1098/rsta.2020.0192. Acquisition and image containers describe the scan. The usual convex solve is least squares or Kullback–Leibler plus a regulariser:

```python
# fragment: needs CIL and a scan's image_geometry, acquisition_geometry, and data
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import LeastSquares, TotalVariation
from cil.optimisation.algorithms import FISTA

A = ProjectionOperator(image_geometry, acquisition_geometry)
algo = FISTA(initial=image_geometry.allocate(0), f=LeastSquares(A, data), g=lam * TotalVariation())
algo.run(100)
```

`SPDHG` in `cil.optimisation.algorithms` is the subset primal-dual method for a stack of projection blocks. Its step-size condition is the stochastic PDHG row of the algorithms reference, not a new rule. `PDHG`, `CGLS`, and `SIRT` are the other stock algorithms. Confirm `LeastSquares` against the installed CIL. The constructor has moved between releases.

## LION

Learned Iterative Optimization Networks, Cambridge Image Analysis. Install from the repository (`pip install .` after cloning). The operator layer is tomosipo on top of ASTRA (Hendriksen and coauthors, Optics Express, 2021). Models that match papers already chosen in these skills:

| Method | Where |
| --- | --- |
| Learned primal-dual (Adler and Öktem) | `LION/models/iterative_unrolled/LPD.py` |
| Learned gradient scheme (Adler and Öktem) | `LION/models/iterative_unrolled/LG.py` |
| FBP then a U-Net (Jin and coauthors) | post-processing `FBPConvNet` |
| Noise2Inverse | `LION.optimizers.Noise2Inverse_solver` |
| Mixed-scale dense network (Pelt and Sethian) | CNN models |

LION is the right place for a CT training run with its dataset loaders (`2DeteCT`, `LIDC-IDRI`) and its saved-experiment format. DeepInverse is the right place when the operator is not a tomosipo projector, or when the method is plug-and-play, diffusion, or MRI. The learned primal-dual blocks `PDNet_PrimalBlock` and `PDNet_DualBlock` in DeepInverse are the same algorithm for a general `physics`. Do not train both on the same scan and expect identical projectors.

## DeepInverse

General imaging operators, data fidelities, priors, classical and unfolded solvers, self-supervised losses, and posterior samplers. The class tables are `deepinv.md`. Cite Tachella and coauthors, JOSS, 2025, doi:10.21105/joss.08923. Prefer it to a handwritten MRI mask or a handwritten plug-and-play loop. For a measured cone-beam geometry, call ASTRA or CIL for the projector instead of `deepinv.physics.Tomography`.
