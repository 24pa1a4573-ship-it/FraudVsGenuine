"""
run_pipeline.py
────────────────────────────────────────────────────────────────
Convenience script to run the full pipeline end-to-end:
  1. simulate_returns.py  → generates synthetic dataset
  2. train.py             → trains all models
  3. Prints a sample prediction from each preset scenario

Usage:
    python run_pipeline.py
"""

import io
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output on Windows (default console is cp1252)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def run(cmd: list[str], desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print("="*60)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, capture_output=False, env=env)
    if result.returncode != 0:
        print(f"\n❌ Step failed: {desc}")
        sys.exit(result.returncode)
    print(f"\n✓ {desc} complete.")


def main():
    print("\n" + "="*60)
    print("  Returns Fraud Sorter — Full Pipeline Runner")
    print("="*60)

    # Step 1: Simulate
    run(
        [sys.executable, "src/simulate_returns.py"],
        "Step 1/3: Generating synthetic returns dataset"
    )

    # Step 2: Train
    run(
        [sys.executable, "src/train.py"],
        "Step 2/3: Training models"
    )

    # Step 3: Sample predictions
    print(f"\n{'='*60}")
    print("  Step 3/3: Running sample predictions")
    print("="*60)

    try:
        from src.pipeline import ReturnSorterPipeline, PRESET_SCENARIOS

        pipeline = ReturnSorterPipeline()

        for name, scenario in PRESET_SCENARIOS.items():
            result = pipeline.predict(scenario)
            print(f"\n── {name} ──────────────────────────────────────")
            print(f"  Risk Score : {result.risk_score:.1f} / 100")
            print(f"  Category   : {result.category}")
            print(f"  Action     : {result.recommended_action}")
            print(f"  Top reason : {result.top_reasons[0] if result.top_reasons else 'N/A'}")

    except Exception as e:
        print(f"  Warning: Could not run sample predictions: {e}")

    print(f"\n{'='*60}")
    print("  ✓ All steps complete!")
    print("  Launch the dashboard with:")
    print("      streamlit run app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
