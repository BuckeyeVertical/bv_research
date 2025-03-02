import shutil
import argparse
from pathlib import Path

def duplicate_config(n, source_file="config.yaml", dest_dir="configs"):
    source_path = Path(source_file)
    dest_dir = Path(dest_dir)
    
    if not source_path.exists():
        print(f"Error: {source_file} not found.")
        return
    
    dest_dir.mkdir(parents=True, exist_ok=True)  # Ensure destination directory exists
    
    for i in range(1, n + 1):
        dest_file = dest_dir / f"config{i}.yaml"
        shutil.copy(source_path, dest_file)
        print(f"Copied {source_file} -> {dest_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duplicate config.yaml N times")
    parser.add_argument("n", type=int, help="Number of copies to create")
    args = parser.parse_args()
    
    duplicate_config(args.n)
