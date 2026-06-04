#import medmnist
#print(medmnist.__version__)

# Load the .npz file
from medmnist import OCTMNIST
OCTMNIST(split="train", download=True,root=r"data\raw",size=224)



