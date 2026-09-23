# Imaging inverse problems

Three agent skills for computational imaging. They tell an agent how to write the measurement model, choose a reconstructor, and choose a solver with a step-size condition it is allowed to claim.

| Skill | What it decides |
| --- | --- |
| `computational-imaging` | Forward operator, noise, sampling, null space |
| `inverse-problems` | Variational regularisation, plug-and-play, RED, unrolling, equivariant imaging, diffusion, Langevin, deep image prior, diffuse optical tomography, electrical impedance tomography |
| `imaging-optimisation` | Proximal gradient, FISTA, ADMM, Chambolle–Pock, stochastic PDHG, discrepancy stopping |

The citations cover variational imaging, plug-and-play, diffusion, statistical inversion, impedance tomography, limited-angle X-ray, PDE-constrained coefficients, and measured tomography. The implementation choice is DeepInverse, ASTRA, CIL, or LION, in `skills/inverse-problems/references/libraries.md`. These are agent instructions distilled from those sources. They are not a survey, and they are not endorsed by the authors.

Install all three. They refer to each other as sibling directories, which is how `npx skills` lays them out.

```bash
npx skills add MohammadSadeghSalehi/imaging-inverse-skills
```

From a local clone, before the repository is on GitHub:

```bash
npx skills add /absolute/path/to/imaging-inverse-skills
```

One skill at a time:

```bash
npx skills add MohammadSadeghSalehi/imaging-inverse-skills --skill inverse-problems
```

The implementation names match DeepInverse 0.4.2. Check them against the installed package before writing code.

## Layout

```
skills/
  computational-imaging/
  inverse-problems/
  imaging-optimisation/
```
