import hashlib
import subprocess
import sys
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_data.py"


def _generate(output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--image-size",
            "8",
            "--train-per-class",
            "2",
            "--val-per-class",
            "1",
            "--test-per-class",
            "1",
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_demo_is_reproducible_and_refuses_overwrite(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    result = _generate(first)
    assert result.returncode == 0, result.stderr
    result = _generate(second)
    assert result.returncode == 0, result.stderr

    files = sorted(first.rglob("*.png"))
    assert len(files) == 8
    assert len({hashlib.sha256(path.read_bytes()).digest() for path in files}) == 8
    for path in files:
        assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert image.size == (8, 8)

    original = {path: path.read_bytes() for path in files}
    result = _generate(first)
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr
    assert {path: path.read_bytes() for path in files} == original
