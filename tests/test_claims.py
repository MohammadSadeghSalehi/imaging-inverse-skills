"""Library behaviours the skills state as fact. If a DeepInverse release changes
one of them, the corresponding sentence in the skills has to change too."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
dinv = pytest.importorskip("deepinv")


def _deblur_problem(n: int = 16):
    torch.manual_seed(0)
    x = torch.rand(1, 1, n, n)
    physics = dinv.physics.BlurFFT(
        img_size=(1, n, n),
        filter=dinv.physics.blur.gaussian_blur(sigma=1.5),
        noise_model=dinv.physics.GaussianNoise(sigma=0.02),
    )
    return x, physics, physics(x)


def test_pdcp_wiring_identity_physics_matches_pgd_and_real_physics_does_not() -> None:
    """algorithms.md: `PDCP(K=physics.A)` must be called with an identity physics."""
    x, physics, y = _deblur_problem()
    L = float(physics.compute_sqnorm(x, tol=1e-6, verbose=False))
    f, R, lam = dinv.optim.L2(), dinv.optim.L1Prior(), 0.02

    def objective(z):
        return float(f(z, y, physics).sum() + lam * R(z).sum())

    ref = objective(dinv.optim.PGD(prior=R, data_fidelity=f, stepsize=1 / L, lambda_reg=lam, max_iter=3000)(y, physics))
    t = 0.99 / L**0.5
    kw = dict(K=physics.A, K_adjoint=physics.A_adjoint, data_fidelity=f, prior=R,
              stepsize=t, stepsize_dual=t, lambda_reg=lam, beta=1.0, max_iter=3000)
    right = objective(dinv.optim.PDCP(**kw)(y, dinv.physics.Denoising()))
    wrong = objective(dinv.optim.PDCP(**kw)(y, physics))

    assert right == pytest.approx(ref, rel=1e-3)
    assert wrong > ref * 1.005


def test_compute_sqnorm_needs_image_shaped_input_on_tomography() -> None:
    """algorithms.md and deepinv.md: pass a tensor shaped like x, not y."""
    physics = dinv.physics.Tomography(angles=30, img_width=32)
    x = torch.rand(1, 1, 32, 32)
    assert float(physics.compute_sqnorm(x, verbose=False)) > 0
    with pytest.raises(Exception):
        physics.compute_sqnorm(physics.A(x), verbose=False)


def test_pdcp_does_not_check_the_step_product() -> None:
    """algorithms.md: PDCP states tau*sigma*||K||^2 <= 1 but does not enforce it."""
    x, physics, y = _deblur_problem()
    model = dinv.optim.PDCP(data_fidelity=dinv.optim.L2(), prior=dinv.optim.L1Prior(),
                            stepsize=10.0, stepsize_dual=10.0, max_iter=2)
    model(y, dinv.physics.Denoising())  # no error raised for a product of 100


def test_mlem_with_prior_runs_as_one_step_late() -> None:
    """forward-models.md: MLEM accepts a prior (Green's one-step-late)."""
    x, physics, y = _deblur_problem()
    y = torch.poisson(physics.A(x).clamp(min=0) * 50) / 50
    out = dinv.optim.MLEM(data_fidelity=dinv.optim.PoissonLikelihood(), prior=dinv.optim.Tikhonov(),
                          lambda_reg=0.01, max_iter=5)(y, physics)
    assert out.min() >= 0


def test_certified_hypergradient_matches_finite_difference() -> None:
    """algorithms.md: the MAID example's hypergradient converges to a finite difference."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "skills/imaging-optimisation/references/algorithms.md").read_text()
    code = next(c for c in re.findall(r"```python\n(.*?)```", text, re.S) if "def hypergradient" in c)
    ns: dict = {}
    exec(code.split("theta = torch.tensor")[0], ns)

    def upper(th: float) -> float:
        th = torch.tensor(th, dtype=torch.float64)
        x = ns["solve_lower"](th, 1e-9, ns["physics"].A_adjoint(ns["y"]))
        return float(0.5 * (x - ns["x_true"]).pow(2).sum())

    d = 1e-4
    fd = (upper(-4.0 + d) - upper(-4.0 - d)) / (2 * d)
    theta = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    hg, _, _ = ns["hypergradient"](theta, 1e-3, ns["physics"].A_adjoint(ns["y"]))
    assert hg == pytest.approx(fd, rel=1e-3)
