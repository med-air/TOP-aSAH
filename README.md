# Introduction

Pytorch implementation for paper **[Causal Effect Estimation on Imaging and Clinical
Data for Treatment Decision Support of
Aneurysmal Subarachnoid Hemorrhage](https://github.com/med-air/TOP-aSAH)**

<p align="center">
<img src="./assets/intro.png" alt="intro" width="85%"/>
</p>

# Abstract
> Aneurysmal subarachnoid hemorrhage is a serious medical emergency of brain that has high mortality and poor prognosis. Treatment effect estimation is of high clinical significance to support the treatment decision-making for aneurysmal subarachnoid hemorrhage. However, most existing studies on treatment decision support of this disease are unable to simultaneously compare the potential outcomes of different treatments for a patient. Furthermore, these studies fail to harmoniously integrate the imaging data with non-imaging clinical data, both of which are significant in clinical scenarios. In this paper, the key challenges we address are: how to effectively estimate the treatment effect for aneurysmal subarachnoid hemorrhage; and how to utilize multi-modality data to perform this estimation. Specifically, we first propose a novel scheme that uses multi-modality confounders distillation architecture to predict the treatment outcome and treatment assignment simultaneously. Notably, with these distilled confounder features, we design an imaging and non-imaging interaction representation learning strategy to use the complementary information extracted from different modalities to balance the feature distribution of different treatment groups. 
We have conducted extensive experiments using a clinical dataset of 656 subarachnoid hemorrhage cases, which was collected from the Hospital Authority of Hong Kong. Our method shows consistent improvements on the evaluation metrics of treatment effect estimation, achieving state-of-the-art results over strong competitors.

# Setup

This model has been tested on the following systems:

+ Linux: Ubuntu 18.04

```bash
Package                Version
---------------------- -------------------
torch                  1.4.0
torchvision            0.5.0
h5py                   3.1.0
opencv-python          4.5.2.52
SimpleITK              2.0.2
scikit-image.          0.17.2
ml-collections         0.1.1
tensorboardx           2.2.0
medpy                  0.4.0
scikit-learn           0.24.2
pandas                 1.1.5
```

# Training & Testing
+ This article uses a private dataset. In order to successfully run the code, you need to prepare your own dataset.
+ Specifically, you need to prepare a .xls file, which saves the patients' non-imaging clinical data and the path of imaging data. We have provided an example for you to run the data, which is saved in "./data/SAH/example.xls".
+ We run main.py to train and evaluate the model:
```bash 
python main.py
```
+ Our proposed model is saved in models.py, named "MultiRL".

# License
This project is covered under the **Apache 2.0 License**.

# Contact
For any questions, please contact 'wama@cse.cuhk.edu.hk'
