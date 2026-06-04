import numpy as np
import matplotlib.pyplot as plt

data = np.load(r"data\raw\octmnist_224.npz")

print(np.shape(data["train_labels"]))
images = data["train_images"]
labels = data["train_labels"].squeeze() # removes the dimension

# Discover all the unique labels in the dataset
diagnosis = {
    0:"choroidal neovascularization (CNV)",
    1:"Diabetic Macular Edema (DME)",
    2:"Drusen",
    3:"Normal"
}
# One figure that shows all labels and an example image
n_classes = len(diagnosis.keys())
fig, axes = plt.subplots(1, n_classes)

for ax, cls in zip(axes, diagnosis.keys()):
    # The first image with the label
    idx = np.where(labels == cls)[0][1]
    img = images[idx]
    
    # Since it is grayscale
    ax.imshow(img, cmap="gray")
    ax.set_title(f"{diagnosis[cls]}\nClass:{cls}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# Independent image of a condition
# plt.imshow(images[np.where(labels == 2)[0][1]],cmap="gray")
# plt.axis("off")
# plt.tight_layout()
# plt.show()
