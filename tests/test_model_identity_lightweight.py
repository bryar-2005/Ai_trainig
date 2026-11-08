"""Model registry hashing must not import the ML runtime on the HTTP event loop."""

import os
import subprocess
import sys
from pathlib import Path


def test_hashing_model_assets_needs_no_ml_runtime(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"architecture":"synthetic"}', encoding="utf-8")
    code = """
import sys
from ml.train.model_identity import local_model_digest
assert local_model_digest(sys.argv[1])
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'peft' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        check=True,
        timeout=10,
    )
