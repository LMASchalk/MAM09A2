#import medmnist
#print(medmnist.__version__)

# Load the .npz file
from medmnist import OCTMNIST
from pathlib import Path
data_folder = Path.home() / "MAM09A2" / "data" / "raw"
OCTMNIST(split="train", download=True,root=data_folder,size=224)



