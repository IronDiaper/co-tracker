import json
import os
import shutil
import time

import pytest

import helpers
from helpers import run_demo

collect_ignore = ["test_bilinear_sample.py"]

_REPORT = {}
_OUTCOMES = {}
_START = time.time()


@pytest.fixture(scope="session", autouse=True)
def results_dir():
    shutil.rmtree(helpers.RESULTS_DIR, ignore_errors=True)
    os.makedirs(helpers.RESULTS_DIR, exist_ok=True)
    return helpers.RESULTS_DIR


@pytest.fixture(scope="session")
def device():
    return run_demo.pick_device(os.environ.get("COTRACKER_DEVICE", "auto"))


@pytest.fixture(scope="session")
def model(device):
    run_demo.setup_threads()
    return run_demo.load_model(run_demo.DEFAULT_CHECKPOINT, device)


@pytest.fixture(scope="session")
def apple_demo(model, device):
    out_dir = os.path.join(helpers.RESULTS_DIR, "e2e_apple")
    return run_demo.run(os.path.join(helpers.DATA_DIR, "apple.mp4"), out_dir=out_dir,
                        grid_size=10, device=device, model=model)


@pytest.fixture
def report(request):
    name = request.node.nodeid.split("::", 1)[-1]

    def _report(summary, artifact=None, **metrics):
        _REPORT[name] = {
            "summary": summary,
            "artifact": helpers.rel(artifact) if artifact else None,
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        }

    return _report


def pytest_runtest_logreport(report):
    name = report.nodeid.split("::", 1)[-1]
    if report.when == "call" or report.outcome != "passed":
        _OUTCOMES[name] = report.outcome.upper()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _OUTCOMES:
        return
    tr = terminalreporter
    tr.write_sep("=", "CoTracker3: results")
    width = max(len(n) for n in _OUTCOMES)
    for name, outcome in _OUTCOMES.items():
        entry = _REPORT.get(name, {})
        markup = {"green": outcome == "PASSED", "red": outcome == "FAILED",
                  "yellow": outcome == "SKIPPED"}
        tr.write(f"{outcome:<8}", **markup)
        tr.write_line(f" {name:<{width}}  {entry.get('summary', '')}")
        if entry.get("artifact"):
            tr.write_line(f"{'':<9} {'':<{width}}  -> {entry['artifact']}")
    tr.write_line(f"Visual results: {helpers.rel(helpers.RESULTS_DIR)}/")

    os.makedirs(helpers.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(helpers.RESULTS_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"exit_status": int(exitstatus),
                   "duration_seconds": round(time.time() - _START, 1),
                   "tests": {n: {"outcome": o, **_REPORT.get(n, {})} for n, o in _OUTCOMES.items()}},
                  f, indent=2, ensure_ascii=False)
