# Imaging inverse problems

[![check](https://github.com/MohammadSadeghSalehi/imaging-inverse-skills/actions/workflows/check.yml/badge.svg)](https://github.com/MohammadSadeghSalehi/imaging-inverse-skills/actions/workflows/check.yml)

Three agent skills for computational imaging. They tell an agent how to write the measurement model, choose a reconstructor, and choose a solver with a step-size condition it is allowed to claim.

| Skill | What it decides |
| --- | --- |
| `computational-imaging` | Forward operator, noise likelihood, sampling design, adjoint test, null space, evaluation without ground truth |
| `inverse-problems` | Variational regularisation, plug-and-play, RED, unrolling, learned convex regularisers, self-supervised training, deep image prior, diffusion and Langevin sampling, impedance and optical tomography, choice of the regularisation weight, uncertainty reporting |
| `imaging-optimisation` | Proximal gradient, FISTA, ADMM, Chambolle–Pock, stochastic PDHG, EM for Poisson data, discrepancy stopping, bilevel learning with exact or inexact hypergradients |

The citations cover variational imaging, plug-and-play, diffusion, statistical inversion, impedance tomography, limited-angle X-ray, PDE-constrained coefficients, and measured tomography. The implementation choice is DeepInverse, ASTRA, CIL, or LION, in `skills/inverse-problems/references/libraries.md`. These are agent instructions distilled from those sources, written from inside the variational and bilevel imaging lineage they describe. They are not a survey, and they are not endorsed by the authors cited.

## Install

```bash
npx skills add MohammadSadeghSalehi/imaging-inverse-skills
```

Install all three. They refer to each other as sibling directories, which is how `npx skills` lays them out, so a single skill on its own loses its links to the other two.

From a local clone:

```bash
npx skills add /absolute/path/to/imaging-inverse-skills
```

## What is checked

The skills cite library classes, and a wrong name is worse than none: an agent copies it into code that fails, or into code that runs and does the wrong thing. So every commit, and a weekly run, checks the skills against DeepInverse:

- every class name in the skills exists in the installed DeepInverse;
- every Python block runs as written;
- every cross-reference between skills resolves;
- the library behaviours the skills state as fact still hold, for example that `PDCP(K=physics.A)` must be called with an identity physics, and that `compute_sqnorm` needs an image-shaped tensor.

The contract is DeepInverse 0.4.2. The same tests run against the latest release, where a failure flags a sentence to update rather than blocking.

```bash
pip install "deepinv==0.4.2" pytest
python -m pytest tests -q
```

`evals/evals.json` holds prompts for checking that the skills change what an agent does, with the behaviour expected and the mistakes to avoid. Run each with and without the skills installed.

## Contributing

Corrections are welcome, particularly from the communities cited. A change that names a DeepInverse class or states how the library behaves should come with the test that pins it. Names from other libraries go in `tests/external_names.txt`, and a code block that cannot run on its own starts with `# fragment:`.

## Layout

```
skills/
  computational-imaging/    SKILL.md, references/forward-models.md
  inverse-problems/         SKILL.md, references/{deepinv,libraries,communities,lineage}.md
  imaging-optimisation/     SKILL.md, references/algorithms.md
tests/                      symbol, snippet, link, and behaviour checks
evals/                      behavioural prompts
```

MIT licence.
