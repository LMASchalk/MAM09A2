# MAM09A2 - OctMNIST classification experiment. **Conventional machine learning vs a deep learning approach**

## About the methods
*This section gives a short explanation and reasoning for the selected methods for both the machine learning and deep learning approaches. The goal is to be able to evaluate the effectiveness of the deep learning approach by comparing it to the conventional machine learning method.*

### Conventional machine learning

For the conventional machine learning approach a Linear Support Vector Classifier (LinearSVC) was used. Each 28x28 grayscale image is flattened into a single vector of 784 pixel values, which forms the input to the model.

Before classification the pixel values are rescaled with a `MinMaxScaler`, which linearly maps each feature to the range [0, 1] using the minimum and maximum seen in the training data. ~~Scaling matters because a support vector machine positions its decision boundary using distances between points, so the default of feeding raw pixels straight in would let the features with the largest values dominate the margin and distort the fit. MinMaxScaler was chosen over the common alternative (StandardScaler) because pixel intensities are already bounded and non-negative (0-255): mapping them to [0, 1] preserves that structure and keeps zero pixels at zero, whereas standardising would introduce negative values and assume a roughly Gaussian spread that pixel data does not have.~~ The scaler sits inside a pipeline together with the classifier, so it is fitted on the training data only and the same transformation is reused on the validation and test data.

The LinearSVC then learns a linear decision boundary for each of the four classes in a one-vs-rest fashion, separating each class from the rest with the largest possible margin. A few settings were changed from their defaults. The parameter `class_weight` is set to `"balanced"` rather than the default, which treats every class equally. This scales each class's penalty inversely to how often it appears, so mistakes on the rare classes such as drusen count for more and the model is not pulled towards the common classes. The parameter `dual` is set to `False` instead of letting it be chosen automatically, which solves the primal form of the optimisation rather than the dual. This is the recommended and faster option when there are many more samples than features, as is the case here with far more images than the 784 pixel features. ~~The parameter `random_state` is fixed to a set value (42) rather than left unseeded, so the solver's internal randomness is deterministic and the same model is produced on every run.~~ Finally, `max_iter` is raised to 5000 from its default of 1000. This represents the cap on the number of solver iterations before it stops, and the higher cap gives the solver enough room to converge on this data, avoiding the convergence warning that the default can produce on the larger train+val refit.

The regularisation strength `C`, which controls the trade-off between a wide margin and misclassifying training points, is the only hyperparameter that is tuned. Five candidate values spanning several orders of magnitude (0.001, 0.01, 0.1, 1.0 and 10.0) are each fitted on the training split and scored on the official validation split, and the value giving the best balanced accuracy is kept~~, rather than tuning with k-fold cross-validation, since the dataset already provides a dedicated validation set~~. The model with the best C is then refitted on the combined training and validation data, and the held-out test set is scored exactly once using the same shared metrics module as the CNN (accuracy, macro one-vs-rest AUC, macro-F1 and per-class recall), so the two models are directly comparable.

### Deep learning
For the classification task two different Deep Learning archetypes were used, MultiLayer Perceptron and Convolutional Neural Network. The specifics of how the two archetypes are implemented will be discussed in the next sections. The optimizer, loss function and regularization are the same for both experiments. **Optimizer: Adam** (Often used since it is computationally efficient and able to deal with pathological curvatures in the gradient. However, Adam does often tend to find minima that are more extreme and other optimizers such as stochastic gradient descent with momentum often find more flatter minima which is leads to better generalisation), **Loss function: CrossEntropy** (Good for classification due to the shape of the loss function exploding at 0) and **Regularization: Early stopping and L2**. 

#### MultiLayer Perceptron
A MultiLayer Perceptron (MLP) is the simplest for of a neural net. Ours consist of two fully connected ReLu layers. With the first layer containing 512 hidden units and the second layer containing 256 hidden units. These numbers were chosen in order to roughly match the amount of weights in the CNN running on 28x28 images. 

#### Convolutional Neural Network
For the deep learning approach a Convolutional Neural Net (CNN) was used. It consists of a simple architecture of two feature extraction blocks (nn.Conv2d + nn.ReLU + nn.MaxPool2d) and a classifier block (nn.Flatten + nn.Linear + nn.ReLU + nn.Linear). See below for an overview of the architecture. 

The feature extraction blocks finds features within the image, in this case set to 32 at the first block and 64 at the second. These are patterns in the images that are learnt, for instance lines, shapes etc. For a single image you can extract these features and see them in a feature map, see image below. 


## About the dataset
The dataset consists of 109,309 images of retinal Optical Coherence Tomography (OCT) from the OctMNIST dataset. These are 2D grayscale images and are available in multiple resolutions, however in this project 28x28 will be used. The dataset consists of a Training,Validation,Test split of (97,477 / 10,832 / 1,000). To see the original paper on this dataset see: Kermany D, Goldbaum M, Cai W et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018; 172(5):1122-1131. doi:10.1016/j.cell.2018.02.010. https://www.cell.com/cms/10.1016/j.cell.2018.02.010/asset/17bdc187-16b7-4a49-acea-f982b88d3b89/main.assets/gr2_lrg.jpg


## About the data splits and preprocessing
We use the official MedMNIST train/validation/test splits rather than resampling our own. This preserves comparability with published OCTMNIST benchmarks and, because MedMNIST exposes no patient identifiers, avoids the patient-level leakage that a random image-level re-split would risk (multiple scans from one patient landing in different splits).

The training and validation sets are class-imbalanced (47.2% normal, 34.4% CNV, 10.5% DME, and 8.0% drusen) while the test set is balanced at 250 images per class (25% each). We address the training imbalance through class weighting in the models rather than altering the splits, and we report per-class metrics because the train/test distribution mismatch means overall accuracy alone can hide poor performance on the minority classes.**!!!**

MedMNIST provides the images already centre-cropped and resized to a fixed square resolution (we use 28×28) with intensities as 8-bit grayscale; "preprocessed" here refers to that standardisation from the original heterogeneous OCT scans. Our pipeline adds only the model-specific steps applied per split: flattening and feature scaling for the classical baseline, and tensor conversion with normalisation to [0, 1] for the network.**!!!**

## Final Evaluation on the Test Set

Both models are evaluated on the official MedMNIST test set (1,000 images,
250 per class), used exactly once after all model selection and tuning is
complete. The same evaluation code and the same metrics are applied to both
models, so the comparison is like-for-like.

We report four metrics:

- **Accuracy (ACC)** - the fraction of test images classified correctly. A
  MedMNIST standard metric, so it is directly comparable to published OCTMNIST
  benchmarks; because the test set is balanced, it is not skewed toward any
  single class.
- **AUC** - area under the ROC curve (one-vs-rest). It is
  threshold-independent, measuring how well the model ranks each class against
  the rest rather than judging a single hard decision. Also a MedMNIST standard
  metric, included for comparability.
- **Macro-F1** - the harmonic mean of precision and recall, averaged equally
  across the four classes. Unlike accuracy, it penalises a model that
  over-predicts the majority classes, so it captures minority-class failure in
  a single number. (Micro-F1 is not reported separately, as it is equal to
  accuracy for single-label classification.)
- **Per-class recall** - for each disease, the fraction of its true cases the
  model identified. This is our diagnostic metric: it shows precisely which
  classes are handled well and which are missed, which the aggregate scores
  cannot reveal.

The first two provide comparability with the MedMNIST benchmark; the last two
expose the per-class behaviour that matters given the class imbalance in the
training data.

### Results

| Metric | LinearSVC baseline | CNN |
|---|---|---|
| Accuracy (ACC) | 0.3760 | 0.6780 |
| AUC (micro, OvR) | 0.6393 | 0.9229 |
| Macro-F1 | 0.2588 | 0.6434 |

**The state of the art model for this classification task is currently ResNet-18 and has an ACC of 0.743. Our initial simple CNN already reaches 91.3% of the performance compared to the state of the art**

Per-class recall:

| Class | LinearSVC baseline | CNN |
|---|---|---|
| CNV | 0.7160 | 0.9640 |
| DME | 0.0160 | 0.6200 |
| drusen | 0.0000 | 0.2120 |
| normal | 0.7720 | 0.9160 |

~~Confusion matrices for both models are saved under `reports/figures/`.~~

### The different classifications
The dataset contains four classifications: choroidal neovascularization (CNV), diabetic macular edema (DME), drusen and normal. An example from the dataset of each class can be seen below. An explanation and a showcase presentation on OCT of each condition can also be seen below. 

<img src="reports\figures\DifferentClassesShowcase.png" alt="DifferentClassesShowcase.png">

####  choroidal neovascularization (CNV)
This condition occurs when new abnormal vessels grow from the choroid into the retinal pigment epithelium (RPE), see the figure below. On OCT this shows as a localized elevation of the RPE by an hyperreflective mass.

<p float="left">
  <img src="reports/figures/neovascularization_hero-2940749225.jpg"
       alt="Choroidal neovascularization photo" width="45%" />
  <img src="reports/figures/choroidalNeovascularization.png"
       alt="Choroidal neovascularization on OCT" width="45%" />
</p>

*source:  https://www.allaboutvision.com/conditions/choroidal-neovascularization-cnv/* 

#### diabetic macular edema (DME)
This condition occurs due to leakage from existing retinal capillaries in the macula due to diabetes, see figure below. The leakage causes fluid to build up between the retinal layers, which can be seen on OCT as a spongy structure. 

<p float="left">
  <img src="reports\figures\Diabetic_Macular_Edema.jpg"
       alt="Choroidal neovascularization photo" width="45%" />
  <img src="reports\figures\diabeticMacularEdema.png"
       alt="Choroidal neovascularization on OCT" width="45%" />
</p>

*source: https://eyewiki.org/Diabetic_Macular_Edema/*

#### drusen
Drusen are extracellular deposits of lipids, proteins and other debris and often presents between bruch's membreme and the RPE, see the figure below. Drusen are a marker and mediator of age-related macular degeneration, since they can cause dysregulation of the RPE. On OCT drusen present as small bumps along the RPE. 

<p float="left">
  <img src="reports\figures\drusen.jpg"
       alt="drusen photo" width="45%" />
  <img src="reports\figures\drusenOCT.png"
       alt="drusen on OCT" width="45%" />
</p>

*source: https://www.scienceofamd.org/learn/*

## How to run
These instructions apply specifically to the UvA Snellius server
### Environment setup. 
1. Pull the following repository: https://github.com/uvadlc/uvadlc_practicals_20252
2. Then run sbatch src/bashrunscripts/install_environment.job

### Downloading the dataset
Run sbatch src/bashrunscripts/make_dataset.job

### Training for machine learning
Run sbatch src/bashrunscripts/fit_linearSVC.job

### Training for Deep learning 
Run sbatch src/bashrunscripts/fit_deeplearning.job
