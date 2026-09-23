"""Checks that keep the skills honest against the library they cite.

- every SKILL.md has valid frontmatter;
- every relative `.md` path mentioned in a skill exists;
- every backticked CamelCase name is a real DeepInverse symbol, unless listed
  in tests/external_names.txt;
- every ```python block runs, except blocks whose first line starts with
  "# fragment:".
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
MARKDOWN = sorted(SKILLS.rglob("*.md")) + [ROOT / "README.md"]
SKILL_DIRS = sorted(p.parent for p in SKILLS.glob("*/SKILL.md"))


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


# --------------------------------------------------------------------------- frontmatter


def _frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "missing YAML frontmatter"
    out, key = {}, None
    for line in m.group(1).splitlines():
        if re.match(r"^[a-z_]+:", line):
            key, _, val = line.partition(":")
            out[key] = val.strip().lstrip(">").strip()
        elif key:
            out[key] = (out[key] + " " + line.strip()).strip()
    return out


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda p: p.name)
def test_frontmatter(skill_dir: Path) -> None:
    fm = _frontmatter((skill_dir / "SKILL.md").read_text())
    assert fm.get("name") == skill_dir.name, "name must match the folder"
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fm["name"])
    assert 50 < len(fm.get("description", "")) <= 1024


# --------------------------------------------------------------------------- links


@pytest.mark.parametrize("md", MARKDOWN, ids=_rel)
def test_relative_links_exist(md: Path) -> None:
    missing = [
        ref
        for ref in re.findall(r"`((?:\.\./|references/)[^`\s]+\.md)`", md.read_text())
        if not (md.parent / ref).resolve().exists()
    ]
    assert not missing, f"broken references: {missing}"


# --------------------------------------------------------------------------- symbols

dinv = pytest.importorskip("deepinv")


def _deepinv_symbols() -> set[str]:
    names: set[str] = set()
    for mod in pkgutil.walk_packages(dinv.__path__, "deepinv."):
        try:
            module = importlib.import_module(mod.name)
        except Exception:  # optional dependencies
            continue
        names.update(n for n in vars(module) if not n.startswith("_"))
        for obj in vars(module).values():
            if isinstance(obj, type):
                names.update(n for n in vars(obj) if not n.startswith("_"))
    return names


EXTERNAL = {
    l.strip()
    for l in (ROOT / "tests" / "external_names.txt").read_text().splitlines()
    if l.strip() and not l.startswith("#")
}


@pytest.fixture(scope="session")
def deepinv_symbols() -> set[str]:
    return _deepinv_symbols()


@pytest.mark.parametrize("md", MARKDOWN, ids=_rel)
def test_cited_names_exist(md: Path, deepinv_symbols: set[str]) -> None:
    cited = set(re.findall(r"`(?:dinv\.|deepinv\.)?(?:[a-z_]+\.)*([A-Z][A-Za-z0-9_]+)`", md.read_text()))
    unknown = sorted(n for n in cited - EXTERNAL if n not in deepinv_symbols)
    assert not unknown, f"not in deepinv {dinv.__version__}: {unknown}"


# --------------------------------------------------------------------------- snippets


def _snippets() -> list[tuple[str, str]]:
    out = []
    for md in MARKDOWN:
        for i, code in enumerate(re.findall(r"```python\n(.*?)```", md.read_text(), re.S)):
            if not code.lstrip().startswith("# fragment:"):
                out.append((f"{_rel(md)}#{i}", code))
    return out


@pytest.mark.parametrize("name,code", _snippets(), ids=[n for n, _ in _snippets()])
def test_snippet_runs(name: str, code: str) -> None:
    exec(compile(code, name, "exec"), {"__name__": "__snippet__"})
