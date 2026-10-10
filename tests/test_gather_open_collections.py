"""Dating and filtering rules in scripts/gather_open_collections.py — no network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "gather_open_collections", REPO_ROOT / "scripts" / "gather_open_collections.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gather_open_collections"] = module
    spec.loader.exec_module(module)
    return module


g = _load()


@pytest.mark.parametrize("text, expected", [
    ("1910-1914 (circa)", 1914),
    ("late 19th century", 1900),
    ("19th–20th century", 2000),
    ("1890s", 1899),
    ("ca. 1540–70", 1570),
    ("1905-15", 1915),
    ("ca. 1961–1917 B.C.", -1917),
    ("undated", None),
    (None, None),
])
def test_latest_year_takes_the_end_of_the_range(text, expected):
    assert g.latest_year(text) == expected


@pytest.mark.parametrize("collector, expected", [
    ("Leo Frobenius (1873-1938)", 1912),     # expedition window beats lifespan
    ("Hans Glauning (1868-1908)", 1908),     # died 1908
    ("Heinrich Bey", None),
    ("Karl Adametz (1877-)", None),          # open lifespan constrains nothing
    (None, None),
])
def test_collector_taq(collector, expected):
    assert g.collector_taq(collector) == expected


def test_taq_is_the_tightest_bound():
    # A label saying "19th–20th century" doesn't outweigh a 1904 accession.
    assert g.terminus_ante_quem(date_display="19th–20th century", acquired_year=1904) == 1904
    assert g.terminus_ante_quem(date_display="1890s", acquired_year=1922) == 1899
    assert g.terminus_ante_quem() is None


@pytest.mark.parametrize("taq, expected", [(1908, "yes"), (1910, "yes"), (1912, "no"), (None, "unknown")])
def test_classify(taq, expected):
    assert g.classify(taq, 1910) == expected


@pytest.mark.parametrize("title, expected", [
    ("Türpfosten", True),
    ("Yoruba Tür", True),
    ("Ifa-Brett", True),
    ("Plaque with mudfish", True),
    ("Houten deurpaneel", True),
    ("Twin figure (ibeji)", False),
])
def test_panel_filter(title, expected):
    assert g.is_panel(g.record(title=title)) is expected
