"""Run steps 3 → 12 end to end:  python3 run_all.py [--max-words 300] [--skip-extract]"""
import subprocess, sys
args = [a for a in sys.argv[1:] if a != "--skip-extract"]
steps = ["extract.py", "clean.py", "structure.py", "chunk.py", "embed_and_store.py", "test_retrieval.py"]
if "--skip-extract" in sys.argv:          # PDFs not needed: start from output/01_extracted
    steps = steps[1:]
for step in steps:
    print(f"\n=== {step}")
    subprocess.run([sys.executable, step, *(args if step == "chunk.py" else [])], check=True)
