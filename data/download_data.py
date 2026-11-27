"""Download the Kermany retinal OCT dataset from Kaggle."""

import argparse
import os
import subprocess
from pathlib import Path

import yaml


def load_config(config_path):
    """Read a YAML config file and return it as a dictionary."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_dataset(kaggle_id, raw_dir):
    """Download and unzip the dataset, unless it was already downloaded."""
    done_file = raw_dir / ".download_complete"
    if done_file.exists():
        print("Dataset already downloaded, skipping.")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", kaggle_id,
               "-p", str(raw_dir), "--unzip"]
    subprocess.run(command, check=True)
    done_file.touch()


def print_folders(raw_dir):
    """Print every folder that holds files, with its number of files."""
    for folder, _, files in sorted(os.walk(raw_dir)):
        if files:
            print(f"{folder!r}: {len(files)} files")


def main():
    """Read the config, download the data and show the folders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    args = parser.parse_args()

    config = load_config(args.config)["dataset"]
    raw_dir = Path(config["raw_dir"])
    download_dataset(config["kaggle_id"], raw_dir)
    print_folders(raw_dir)


if __name__ == "__main__":
    main()
