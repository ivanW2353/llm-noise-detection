"""Smoke tests for refactored analysis scripts.

Verifies that all scripts can import their dependencies and parse arguments
after the Phase 1 + Phase 2 refactoring.
"""
import subprocess
import sys


def test_script_help(script_name):
    """Test that a script can show help without errors."""
    result = subprocess.run(
        [sys.executable, script_name, "--help"],
        cwd="/root/noisedetect",
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0, f"{script_name} failed: {result.stderr}"
    assert "usage:" in result.stdout.lower(), f"No usage in {script_name} output"
    return True


def main():
    scripts = [
        "scripts/3_analysis/analyze_detection.py",
        "scripts/3_analysis/analyze_unsupervised.py",
        "scripts/3_analysis/analyze_memorization.py",
        "scripts/3_analysis/analyze_early_detection.py",
        "scripts/3_analysis/analyze_transfer.py",
        "scripts/3_analysis/analyze_token_concentration.py",
        "scripts/3_analysis/analyze_all_features.py",
        "scripts/3_analysis/analyze_token_level.py",
        "scripts/3_analysis/compute_ifd.py",
        "scripts/2_train/train.py",
        "scripts/2_train/evaluate.py",
        "scripts/1_data/make_noise.py",
    ]

    print("Testing refactored scripts...")
    passed = 0
    for script in scripts:
        try:
            test_script_help(script)
            print(f"  ✓ {script}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {script}: {e}")

    print(f"\n{passed}/{len(scripts)} scripts passed")
    return passed == len(scripts)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
