"""Console-script entry points delegating to scripts/ modules."""
from __future__ import annotations

def _run(modname: str) -> None:
    import runpy, sys
    # Prefer installed package sibling scripts under repo scripts/
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    script = root / "scripts" / f"{modname}.py"
    if script.exists():
        sys.argv[0] = str(script)
        runpy.run_path(str(script), run_name="__main__")
    else:
        raise SystemExit(f"script not found: {script}")

def build_partitions():
    _run("build_partitions")

def run_condition_s():
    _run("run_condition_s")

def run_condition_g():
    _run("run_condition_g")

def make_spatial_blocks():
    _run("make_spatial_blocks")

def aggregate_tessellation():
    _run("aggregate_tessellation")


def audit_data_readiness():
    _run("audit_data_readiness")


def run_hk_smoke():
    _run("run_hk_smoke")


def run_orchestrator():
    _run("run_orchestrator")


def build_metric_distances():
    _run("build_metric_distances")


def run_mitma_condition_s():
    _run("run_mitma_condition_s")


def run_mitma_condition_g():
    _run("run_mitma_condition_g")


def run_lombardy_condition_s():
    _run("run_lombardy_condition_s")


def run_lombardy_condition_g():
    _run("run_lombardy_condition_g")


def run_neurogravity_pretrain():
    _run("run_neurogravity_pretrain")


def run_neurogravity_fewshot():
    _run("run_neurogravity_fewshot")
