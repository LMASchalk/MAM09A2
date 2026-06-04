import numpy as np
import matplotlib.pyplot as plt

data = np.load(r"data\raw\pathmnist_64.npz")
#print(np.shape(data["train_labels"]))
X_train = data["train_images"]
img = X_train[2]

# 5. Display it
plt.imshow(img)          # for RGB images
plt.axis("off")
plt.show()
