# Load the .npz file
import argparse
from medmnist import OCTMNIST
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=int,
        default=28,          # fallback if you don't pass --size
        help="Image size to use for OCTMNIST. Choices are: 244,128,64,28 (Default)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    data_folder = Path.home() / "MAM09A2" / "data" / "raw"

    # Use the CLI parameter here:
    OCTMNIST(
        split="train",
        download=True,
        root=data_folder,
        size=args.size
    )

if __name__ == "__main__":
    main()




