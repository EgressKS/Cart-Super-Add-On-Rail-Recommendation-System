"""
Zomato CSAO Recommendation System
Executes the complete pipeline end-to-end.
"""
import subprocess, sys, os, time

PYTHON = sys.executable
BASE   = os.path.dirname(os.path.abspath(__file__))

def run(script: str, label: str):
    path = os.path.join(BASE, script)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([PYTHON, path], cwd=BASE)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAILED] {label} exited with code {result.returncode}")
        sys.exit(1)
    print(f"  Done in {elapsed:.1f}s")

if __name__ == "__main__":
    run("data/generation/generate_data.py",   "Step 1: Synthetic Data Generation")
    run("features/feature_engineering.py",    "Step 2: Feature Engineering")
    run("models/train_and_evaluate.py",       "Step 3: Model Training & Evaluation")
    run("models/business_analysis.py",        "Step 4: Business Impact Analysis")
    run("api/main.py",                        "Step 5: API Testing")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE!")
    print("  Artifacts in:")
    print(f"    Data    : {os.path.join(BASE,'data','raw')}")
    print(f"    Features: {os.path.join(BASE,'data','processed')}")
    print(f"    Model   : {os.path.join(BASE,'models','saved')}")
    print(f"    Reports : {os.path.join(BASE,'models','reports')}")
    print("\n  To start the API server:")
    print(f"    uvicorn api.main:app --reload --port 8000")
    print("="*60)
