from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_prepare_windows_cli_split_policy_sequence(tmp_path):
    out_dir = tmp_path / "processed"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_windows.py",
            "tests/fixtures/dummy.fa",
            "--out-dir", str(out_dir),
            "--block-size", "32",
            "--stride", "16",
            "--max-n-rate", "1.0",
            "--max-windows", "8",
            "--shard",
            "--shard-size", "5",
            "--split-policy", "sequence",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    manifest = json.loads(result.stdout)
    assert manifest["split_policy"] == "sequence"


def test_prepare_windows_cli_default_split_policy_is_window(tmp_path):
    out_dir = tmp_path / "processed_default"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_windows.py",
            "tests/fixtures/dummy.fa",
            "--out-dir", str(out_dir),
            "--block-size", "32",
            "--stride", "16",
            "--max-n-rate", "1.0",
            "--max-windows", "8",
            "--shard",
            "--shard-size", "10",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    manifest = json.loads(result.stdout)
    assert manifest["split_policy"] == "window"


def test_prepare_windows_cli_invalid_split_policy_rejected():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_windows.py",
            "tests/fixtures/dummy.fa",
            "--out-dir", "/tmp/nonexistent",
            "--block-size", "32",
            "--stride", "16",
            "--split-policy", "invalid",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


class TestHFModelCard:
    hf_readme = Path("repo/moss-dna-gpt-20m-patens/README.md")

    def test_hf_card_mentions_bits_per_base(self):
        text = self.hf_readme.read_text(encoding="utf-8")
        assert "1.41665" in text

    def test_hf_card_mentions_markov_baseline(self):
        text = self.hf_readme.read_text(encoding="utf-8")
        assert "1.88614" in text
        assert "Markov" in text

    def test_hf_card_has_limitations_or_not_intended_section(self):
        text = self.hf_readme.read_text(encoding="utf-8")
        assert any(kw in text for kw in ("Limitations", "limitations", "Not intended for"))


def test_publish_dry_run():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/publish_to_hf.py",
            "--repo-id", "dummy/dummy",
            "--local-dir", "repo/moss-dna-gpt-20m-patens",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "DRY RUN" in result.stdout
