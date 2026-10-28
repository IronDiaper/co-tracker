import os
import subprocess
import sys

from helpers import RESULTS_DIR, TESTS_DIR


def test_author_bilinear_sampler(report):
    script = os.path.join(TESTS_DIR, "test_bilinear_sample.py")
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=300)
    log = os.path.join(RESULTS_DIR, "author_test_bilinear_sample.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(proc.stdout + proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    last = [line for line in proc.stderr.splitlines() if line.strip()][-1]
    assert last.startswith("OK"), proc.stderr
    report(f"unittest of the authors: {last}", artifact=log)
