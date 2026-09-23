# Imaging inverse problems: agent skills

[![check](https://github.com/MohammadSadeghSalehi/imaging-inverse-skills/actions/workflows/check.yml/badge.svg)](https://github.com/MohammadSadeghSalehi/imaging-inverse-skills/actions/workflows/check.yml)
[![DeepInverse 0.4.2](https://img.shields.io/badge/DeepInverse-0.4.2-blue)](https://deepinv.org)
[![Agent Skills](https://img.shields.io/badge/Agent%20Skills-3-8A2BE2)](https://github.com/vercel-labs/skills)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

Three skills that make a coding agent reason about imaging inverse problems the way the field does: write the measurement model first, pick a reconstructor whose assumptions hold, and run a solver with a step size and a convergence claim it can justify. Built on DeepInverse, ASTRA, CIL, and LION, and tested against DeepInverse on every commit.

```bash
npx skills add MohammadSadeghSalehi/imaging-inverse-skills
```

## Why

General-purpose agents write imaging code that runs and is wrong. These are mistakes the skills are written to prevent, each checked against the library:

- **The wrong likelihood.** A squared-error data term on photon counts, where PET, low-dose CT, and fluorescence need a Poisson likelihood and a nonnegative image.
- **A silently wrong solver.** `deepinv.optim.PDCP(K=physics.A)` called with the same `physics` applies `A` twice and minimises a different objective, with no error. The skills give the wiring that reproduces the proximal-gradient optimum.
- **An illegal step size.** Estimating `‖A‖²` from the measurement instead of an image-shaped tensor fails on tomography and gives the wrong step elsewhere.
- **Convergence claimed without its hypothesis.** A pretrained denoiser in plug-and-play meets none of the conditions under which the iteration provably converges until someone checks.
- **A regularisation weight tuned by eye,** where the discrepancy principle, SURE, or bilevel learning give a defensible choice.
- **Posterior spread reported as an error bar,** without checking that it is calibrated.

## The skills

| Skill | It decides |
| --- | --- |
| [`computational-imaging`](skills/computational-imaging/SKILL.md) | The forward operator, the noise likelihood, the sampling design, the adjoint test, the null space, and how to evaluate without ground truth |
| [`inverse-problems`](skills/inverse-problems/SKILL.md) | The reconstructor: variational regularisation, plug-and-play and RED, unrolled and learned primal-dual networks, learned convex regularisers, self-supervised training, deep image prior, diffusion and flow-matching posterior sampling (DPS, DDRM, DDNM, ΠGDM, DAPS, latent diffusion, sequential Monte Carlo, PnP-Flow, OT-ODE, D-Flow), Langevin sampling, impedance and optical tomography, off-the-grid sparse recovery, imperfect and learned forward operators, the choice of `λ`, and uncertainty reporting, including hypothesis tests and multilevel Monte Carlo |
| [`imaging-optimisation`](skills/imaging-optimisation/SKILL.md) | The solver and its step: proximal gradient, FISTA, ADMM, Chambolle–Pock, Condat–Vũ, variable-metric and majorise–minimise methods, stochastic and block-coordinate methods, multilevel optimisation, EM for Poisson data, early stopping, and bilevel learning, including which hypergradient to compute and adaptively inexact methods (MAID) |

They refer to each other as sibling folders, so install all three. An agent loads a skill when the task matches its description, and reads the reference files only when it needs them.

## Install

```bash
npx skills add MohammadSadeghSalehi/imaging-inverse-skills
```

The installer asks which agents to install into. To name them directly, for example Claude Code, Cursor, and Codex:

```bash
npx skills add MohammadSadeghSalehi/imaging-inverse-skills -a claude-code -a cursor -a codex --skill '*' -y
```

It supports most coding agents, including Gemini CLI, GitHub Copilot, Windsurf, Cline, and OpenCode; run `npx skills add --help` for the list. Without the installer, copy the three folders under `skills/` into your agent's skills directory, for example `.claude/skills/`, keeping them side by side.

## Try it

After installing, ask your agent:

- "I have PET sinogram counts and a scanner model. Set up the reconstruction in DeepInverse."
- "Write a Chambolle–Pock solver for TV-regularised deblurring and check the step sizes."
- "I plugged a pretrained DRUNet into PnP-ADMM. Can I say it converges?"
- "Learn the TV weight from 20 paired CT scans. Which hypergradient should I use?"
- "I only have noisy CT projections. How do I train a denoiser?"
- "Localise single fluorescent molecules from a blurred microscope frame."
- "Use my pretrained flow-matching model as a prior for undersampled MRI and give me uncertainty."

`evals/evals.json` lists these and more, with the behaviour to expect and the mistakes to avoid, for comparing an agent with and without the skills.

## What is tested

A wrong class name is worse than none: an agent copies it into code that fails, or into code that runs and does the wrong thing. Every push, and a weekly scheduled run, checks against DeepInverse:

- every class name cited in the skills exists;
- every Python block runs as written;
- every cross-reference between the skills resolves;
- the library behaviours the skills state as fact still hold, including the `PDCP` wiring, the norm estimate, and that the bilevel example's hypergradient matches a finite difference.

The contract is DeepInverse 0.4.2. The same suite runs against the latest release, where a failure flags a sentence to update.

```bash
pip install "deepinv==0.4.2" pytest
python -m pytest tests -q
```

## Key references

The skills cite more than a hundred papers, each next to the method it justifies. The ones they are organised around:

- Arridge, Maass, Öktem, Schönlieb. Solving inverse problems using data-driven models. *Acta Numerica*, 2019. doi:[10.1017/S0962492919000059](https://doi.org/10.1017/S0962492919000059)
- Chambolle, Pock. An introduction to continuous optimization for imaging. *Acta Numerica*, 2016.
- Chambolle, Pock. A first-order primal-dual algorithm for convex problems with applications to imaging. *Journal of Mathematical Imaging and Vision*, 2011.
- Chambolle, Ehrhardt, Richtárik, Schönlieb. Stochastic primal-dual hybrid gradient algorithm with arbitrary sampling and imaging applications. *SIAM Journal on Optimization*, 2018.
- Kamilov, Bouman, Buzzard, Wohlberg. Plug-and-play methods for integrating physical and learned models in computational imaging. *IEEE Signal Processing Magazine*, 2023. doi:[10.1109/MSP.2022.3199595](https://doi.org/10.1109/MSP.2022.3199595)
- Kaipio, Somersalo. *Statistical and Computational Inverse Problems*. Springer, 2005.
- Combettes, Pesquet. Proximal splitting methods in signal processing. In *Fixed-Point Algorithms for Inverse Problems in Science and Engineering*, Springer, 2011. doi:[10.1007/978-1-4419-9569-8_10](https://doi.org/10.1007/978-1-4419-9569-8_10)
- Peyré, Cuturi. Computational optimal transport. *Foundations and Trends in Machine Learning*, 2019. doi:[10.1561/2200000073](https://doi.org/10.1561/2200000073)
- Daras and coauthors. A survey on diffusion models for inverse problems. arXiv:[2410.00083](https://arxiv.org/abs/2410.00083), 2024.
- Lipman, Chen, Ben-Hamu, Nickel, Le. Flow matching for generative modeling. *ICLR*, 2023.
- Giles. Multilevel Monte Carlo methods. *Acta Numerica*, 2015. doi:[10.1017/S096249291500001X](https://doi.org/10.1017/S096249291500001X)
- Crockett, Fessler. Bilevel methods for image reconstruction. *Foundations and Trends in Signal Processing*, 2022. doi:[10.1561/2000000111](https://doi.org/10.1561/2000000111)
- Salehi, Mukherjee, Roberts, Ehrhardt. An adaptively inexact first-order method for bilevel optimization with application to hyperparameter learning. *SIAM Journal on Mathematics of Data Science*, 2025. doi:[10.1137/24M1653513](https://doi.org/10.1137/24M1653513)
- Hertrich and coauthors. Learning regularization functionals for inverse problems: a comparative study. *Handbook of Numerical Analysis*, 2026. doi:[10.1016/bs.hna.2026.04.001](https://doi.org/10.1016/bs.hna.2026.04.001)
- Tachella and coauthors. DeepInverse: a Python package for solving imaging inverse problems with deep learning. *Journal of Open Source Software*, 2025. doi:[10.21105/joss.08923](https://doi.org/10.21105/joss.08923)

The skills are agent instructions distilled from these sources, written from inside the variational and bilevel imaging lineage they describe. They are not a survey, and they are not endorsed by the authors cited.

## Contributing

Corrections and additions are welcome, particularly from the communities cited. Open an issue or a pull request. A change that names a DeepInverse class or states how the library behaves should come with the test that pins it. Names from other libraries go in `tests/external_names.txt`, and a code block that cannot run on its own starts with `# fragment:`.

## Citing

If the skills help your work, cite the repository with the "Cite this repository" button on GitHub, or [`CITATION.cff`](CITATION.cff), and cite the papers behind the method you used.

## Author

Mohammad Sadegh Salehi ([Google Scholar](https://scholar.google.com/citations?user=bunZmJsAAAAJ)), who works on bilevel learning and inexact optimisation for imaging. MIT licence.

```
skills/
  computational-imaging/    SKILL.md, references/forward-models.md
  inverse-problems/         SKILL.md, references/{deepinv,libraries,communities,lineage}.md
  imaging-optimisation/     SKILL.md, references/algorithms.md
tests/                      symbol, snippet, link, and behaviour checks
evals/                      behavioural prompts
```
