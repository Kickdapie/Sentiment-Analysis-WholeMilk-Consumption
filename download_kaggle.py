"""
Download Kaggle Sentiment140 dataset into data/kaggle_sentiment140/.
Requires: pip install kaggle, and Kaggle API key in ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys

import config

def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    target_dir = os.path.join(config.DATA_DIR, "kaggle_sentiment140")
    os.makedirs(target_dir, exist_ok=True)
    try:
        subprocess.run([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "kazanova/sentiment140",
            "-p", target_dir,
            "--unzip",
        ], check=True)
        print(f"Downloaded to {target_dir}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("Kaggle CLI failed. Download manually from:")
        print("  https://www.kaggle.com/datasets/kazanova/sentiment140")
        print(f"  Place the CSV in: {target_dir}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
