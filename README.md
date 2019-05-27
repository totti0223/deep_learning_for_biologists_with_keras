# Deep Learning for Biologists with Keras



## By whom?

Yosuke Toda (tyosuke-at-aquaseerser.com)

JST PRESTO researcher at Nagoya Univeristy /  Agri-Heir Co., Ltd.



## What is it?

- Tutorials for deep learning based analysis  (mainly) on biological relavent themes. Should give you a better implementation of DL much more than general tutorial tasks like MNIST and CIFAR-10. 
- Google Colaboratory based notebooks. All you need is an internet connection, google chrome browser, and google account. **GPU learning environment at a click!**
- To open the notebook, click the ![image](https://colab.research.google.com/assets/colab-badge.svg)in each section. Logging into Google Account and copying the ipynb to your local google doc folder is preferred for code customization and saving.

## Note

- Mathmatical calculations and/or theoretical backgrounds will not be thoroughly explained in this tutorial. The object of this notebook is to get a overview of how we can perform DL in the field of biology (especially in plant science and agriculture) for non informatitians.
- **Keras with Tensorflow background** is the main DL framework used in the notebook. I do not intend to mix different frameworks for clarity in the current situation.
- Feedbacks and requests, complements including typos and misusage of codes in the notebooks are highly welcomed in the issues of github repo or at twitter (@totti0223).
- A lot of stuff in this notebook is still in alpha ver. (code readability, comments). But to gain early feedbacks, opening them cowardly.

## To do

- Add more examples (described in the *"Notebooks to be opened"* section)



## Notebooks Open

### Keras Code Collection: How to use Keras

<img src = "assets/image-20190117153946252.png" width="150" ALIGN="left" /> [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/totti0223/deep_learning_for_biologists_with_keras/blob/master/notebooks/basic_usage_of_keras.ipynb) <br>

![badge](https://img.shields.io/badge/tag-preferred_for_Keras_beginners-green.svg) 

Basic code collection of Keras. Loading images, build, train, evaluate and using the CNN. As Keras is quite flexible and have multiple ways of writing code even for a simplest CNN, I have written down all the patterns in one Colab notebook.  *Images of rice seeds were provided from Dr. S. Nishiuchi at Nagoya Univ. in 2016 (personal communication).* However, the data itself is not going to be discussed. Preferred to go through the notebook of Rice Seed Integrity below. 



![badge](https://img.shields.io/badge/todo-orange.svg) Finalize with feedbacks

<br>

### Rice Seed Integrity: What is deep learning.

<img src = "assets/image-20190115201428173.png" width="150" ALIGN="left" /> [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/totti0223/deep_learning_for_biologists_with_keras/blob/master/notebooks/rice_seed_classification.ipynb) <br>

![badge](https://img.shields.io/badge/type-classification-blue.svg)![badge](https://img.shields.io/badge/tag-preferred_for_beginners-green.svg) ![badge](https://img.shields.io/badge/tag-CNN-green.svg) ![badge](https://img.shields.io/badge/tag-comparison_with_classical_ML-green.svg)

An introductory notebook to deep learning based image analysis as well as comparing it with classical machine learning algorithms and manual image classification. The object of this notebook is to give the readers an implementation of; What does "Representative Learning" actually mean? What is Feature Selection? *Images of rice seeds were provided from Dr. S. Nishiuchi at Nagoya Univ. in 2016 (personal communication).*



![badge](https://img.shields.io/badge/todo-orange.svg) Refurnish Codes and Comments

<br>

### 17 Flowers Dataset 

<img src = "assets/image-20190115201017711.png" width="150" ALIGN="left" />   [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/totti0223/deep_learning_for_biologists_with_keras/blob/master/notebooks/flower_image_classification.ipynb) <br>

![badge](https://img.shields.io/badge/type-classification-blue.svg) ![badge](https://img.shields.io/badge/tag-CNN-green.svg)  ![badge](https://img.shields.io/badge/tag-Transfer_Learning-green.svg) ![badge](https://img.shields.io/badge/tag-Fine_Tuning-green.svg)

Will build a convolutional neural network (CNN) based classification model using a 17 category flower dataset provided by the team at University of Oxford (http://www.robots.ox.ac.uk/~vgg/data/flowers/17/). The dataset provides of 80 images per category. We will compare the training process starting from scratch (*de novo*), transfer-learning and fine-tuning which the later two are pretrained with ImageNet Dataset. We will see that upon training with not so much data (for CNN), pretraining has a great effect upon speed and (ocasionally) accuracy of the model.



![badge](https://img.shields.io/badge/todo-orange.svg) Refurnish Codes and Comments

<br>

### Crop/Weed Segmentation

<img src = "assets/image-20190115201227438.png" width="150" ALIGN="left" />  [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/totti0223/deep_learning_for_biologists_with_keras/blob/master/notebooks/crop_weed_semantic_segmentation.ipynb) <br>

![badge](https://img.shields.io/badge/type-segmentation-blue.svg) ![badge](https://img.shields.io/badge/tag-UNet-green.svg)

In this notebook, we will perform a segmentation of crop and weed region from images taken by an autonomous field robot, which the dataset are from Haug et al., "A Crop/Weed Field Image Dataset for the Evaluation of Computer Vision Based Precision Agriculture Tasks" (2015). With conventional approaches, we possibly can isolate the weed and crop resions from the soil area using a color threshold in the green domain. However, how can we further classify the weed (red) from the crop (green) region? Such feature selection is a master of a master craftsmanship. Instead, we will use DL, in specific, semantic segmentation methods to 1) Isolate the grass regions from the soil, 2) Isolating and classifying weeds and crops regions. A neural network architecture named U-Net will be used here.



![badge](https://img.shields.io/badge/todo-orange.svg) Need to add commentary throughout the notebook.

<br>

### Bamboo forest detection from Google Earth Images

<img src = "assets/image-20190120202734357.png" width="150" ALIGN="left" />  [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/totti0223/deep_learning_for_biologists_with_keras/blob/master/notebooks/bamboo_forest_detection.ipynb) <br>

Original paper from Watanabe et al., "Using deep learning for bamboo forest detection from Google Earth images" doi: https://doi.org/10.1101/351643.

 The data was provided by Dr. Ise (personal communcation). However the image originates from Google Earth, so please confirm the copyright if you intend to use this in non research purpose.

![badge](https://img.shields.io/badge/type-segmentation-blue.svg) ![badge](https://img.shields.io/badge/tag-CNN-green.svg)



![badge](https://img.shields.io/badge/todo-orange.svg) Need to add commentary throughout the notebook and above.

<br>

### Yeast GFP Protein Localization

<img src = "assets/image-20190115201711326.png" height="150px" ALIGN="left" /> [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/totti0223/deep_learning_for_biologists_with_keras/blob/master/notebooks/yeast_GFP_localization_classification.ipynb) <br>

![badge](https://img.shields.io/badge/type-classification-blue.svg) ![badge](https://img.shields.io/badge/tag-CNN-green.svg) ![badge](https://img.shields.io/badge/tag-Pandas_Dataframe_yielding-green.svg)

Yet another classification task. Dataset from the paper *Accurate Classification of Protein Subcellular Localization from High-Throughput Microscopy Images Using Deep Learning Tanel Pärnamaa and Leopold Parts G3: GENES, GENOMES, GENETICS May 1, 2017 vol. 7 no. 5 1385-1392*. This dataset has a csv file with file name information associated with class label instead of allocating the file to folders corresponding to its class. Moreover, dataset comprises of 90,000 images, possibly too large to load everything in label. Here we use the ImageDataGenerator class and its flow_from_dataframe function to feed batches of image upon training. Note: flow_from_dataframe in keras repository has a bug. we instead install keras_preprocessing and use the uptodate ImageDataGenerator (19, Jan. 2019 ).



![badge](https://img.shields.io/badge/todo-orange.svg) Need to add commentary throughout the notebook.

<br>





## Others

### Simulated ChIP-seq motif extraction

<img src = "assets/image-20190115202731524.png" height="150px" ALIGN="left" /> [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/totti0223/deep_learning_for_biologists_with_keras/blob/master/notebooks/dna_simulated_chip.ipynb) <br>

![badge](https://img.shields.io/badge/type-classification-blue.svg) ![badge](https://img.shields.io/badge/tag-CNN-green.svg) ![badge](https://img.shields.io/badge/tag-basics_of_handling_DNA_in_DL-green.svg) ![badge](https://img.shields.io/badge/tag-simple_visualization_of_DL_decision-green.svg)



![badge](https://img.shields.io/badge/todo-orange.svg)Need to add commentary throughout the notebook.

<br>

<br>

<br>

## Updates
- Will have a major update including code sanitizing, reviewing, new sections etc.


## Notebooks To be Opened

- Crop disease diagnosis interpretability (currently under revision in peer reviewed journal)

- Arabidopsis Leaf Counting

- Stomatal Aperture Quantification pipeline

- GAN of somekind

- Pix2pix for microscope image alternation









  <!---
​    <img src = "assets/image-20190115144920126.png" height="120px" ALIGN="left" />
  -->



