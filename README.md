# MAM09A2 - OctMNIST classification experiment. **Conventional machine learning vs a deeplearning approach**

## About the methods
*This section gives a short explanation and reasoning for the selected methods for both the machine learning and deep learning approaches. The goal is to be able to evaluate the effectiveness of the deep learning approach by comparing it to the conventional machine learning method.*

### Conventional machine learning

### Deep learning 
For the deep learning approach a Convolutional Neural Net (CNN) was used. It consists of a simple architecture of two feature extraction blocks (nn.Conv2d + nn.ReLU + nn.MaxPool2d) and a classifier block (nn.Flatten + nn.Linear + nn.ReLU + nn.Linear). See below for an overview of the architecture. 



The feature extraction blocks finds features within the image, in this case set to 32 at the first block and 64 at the second. These are patterns in the images that are learnt, for instance lines, shapes etc. For a single image you can extract these features and see them in a feature map, see image below. 


## About the dataset
The dataset consists of 109,309 images of retinal Optical Coherence Tomography (OCT) from the OctMNIST dataset. These are 2D grayscale images and are available in multiple resolutions, however in this project 244x244 will be used. The dataset consists of a Training,Validation,Test split of (97,477 / 10,832 / 1,000). To see the original paper on this dataset see: Kermany D, Goldbaum M, Cai W et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018; 172(5):1122-1131. doi:10.1016/j.cell.2018.02.010. https://www.cell.com/cms/10.1016/j.cell.2018.02.010/asset/17bdc187-16b7-4a49-acea-f982b88d3b89/main.assets/gr2_lrg.jpg

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
