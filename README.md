<p align="center">
    <img width="550" src="https://github.com/user-attachments/assets/e31fc937-934a-44ae-9797-1d4c2a3fd15d" alt="NeuRoNeLab logo">
</p>

<h1 align="center">
    Advancing Label-Free Cell Classification with Connectome-Inspired Explainable Models and a novel LIVECell-CLS Dataset
</h1>

<p align="center">
  P. Fiore, A. Terlizzi, F. Bardozzo, P. Liò, R. Tagliaferri
</p>

<p align="center">
 <a href="#"><img src="https://img.shields.io/github/contributors/NeuRoNeLab/connectome-livecell-cls?style=for-the-badge" alt="Contributors"/></a>
 <img src="https://img.shields.io/github/last-commit/NeuRoNeLab/connectome-livecell-cls?style=for-the-badge" alt="last commit">
</p>
<p align="center">
 <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs Welcome"/></a>
 <a href="#"><img src="https://img.shields.io/github/languages/top/NeuRoNeLab/connectome-livecell-cls?style=for-the-badge" alt="Languages"/></a>
</p>



# Table of Contents 
1. [Introduction](#introduction)
2. [LIVECell-CLS Dataset](#livecell-cls-dataset)
3. [Methods](#methods)
4. [Results](#results)
5. [Models' Weights](#models-weights)
6. [Installation Guide](#installation-guide)
   - [Installing Python](#installing-python)
   - [Cloning the Repository](#cloning-the-repository)
   - [Creating the Virtual Environment](#creating-the-virtual-environment)
   - [Installing Requirements](#installing-requirements)
7. [Getting to know the Model](#getting-to-know-the-model)
   - [Training](#training)
   - [Evaluation](#evaluation)
8. [Acknowledgements and references](#acknowledgements-and-references)  

# Introduction

Deep learning for label-free cell imaging has become a cornerstone of modern medical and biological research; however, creating robust and generalizable models for these tasks demands large-scale datasets.

To address this, we present **LIVECell-CLS**, the largest benchmark dataset for label-free cell classification, featuring over **1.6 million images across 8 distinct cell lines**. This dataset establishes a new standard for evaluating and advancing deep learning models in the field. Alongside LIVECell-CLS, we provide a comprehensive analysis of 14 state-of-the-art architectures, including CNNs, Vision Transformers (ViTs), and MLP-Mixers, revealing that CNN-based models consistently outperform other approaches in balanced accuracy and F1-score. 

To further enhance performance, we introduce Tensor Network variants inspired by the *C. elegans* connectome, which improve latent feature representations and achieve up to 4% gains in test accuracy with minimal parameter overhead. Our top-performing model, Elegans-EfficientNetV2-M, achieves 90.35% test accuracy and 94.82% F1-score, setting a new benchmark for label-free cell classification. 

Additionally, using Explainable AI techniques and UMAP visualizations, we provide insights into how these models process cell image data, highlighting improved feature separability and decision-making precision, especially for morphologically similar cell lines. This repository includes the LIVECell-CLS dataset, pre-trained models, and tools to facilitate reproducibility and further research in this domain.

# LIVECell-CLS Dataset

# Methods 

# Results 

# Model's Weights

# Installation Guide 

To install the necessary requirements for the project, please follow the steps below.

## Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.9` or higher.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).
## Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

_You may skip this step, but please keep in mind that doing so **could potentially lead to conflicts** if you have other projects on your machine_. 

## Cloning the Repository 
To clone this repository, download and extract the `.zip` project files using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/NeuRoNeLab/remote-sensing-captioning-transformer.git
```

## Installing Requirements
To install the requirements, please: 
1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Install the project requirements using `pip`:
```shell 
pip install -r requirements.txt
```

# Getting to know the Model

## Training 

## Evaluation



# Acknowledgements and references
