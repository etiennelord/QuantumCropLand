# QuantumCropLand (v1.0 June 2025) 
Small dataset for advancing research on quantum image encoding and classification for agriculture application. The dataset is provided as a numpy file (npz) with two vectors named '_data_' and '_labels_'. The data were extracted form the Sentinel-2 (ESA) satellite imagery using [Google Earth Engine](https://earthengine.google.com/  "Google Earth Engine"), using the [AAFC Annual Crop Inventory](https://www.agr.gc.ca/atlas/aci) layers.

The dataset goal is to facilitate work on quantum image encoding and classification by providing a standard dataset, similar to the Iris dataset or EuroSat dataset, but for agricultural data. 

Authors: Ryan Godin, Amanda Boatswain Jacques, Etienne Lord (Agriculture and Agri-Food Canada)

# Dataset information

![dataset overview](https://github.com/etiennelord/QuantumCropLand/blob/main/20250605_16x16_12channel_ndvi_psri.png)

The dataset was created using Google Earth Engine, similarly to the work of :
Boatswain Jacques, A. A., Diallo, A. B., & Lord, E. (2023). The Canadian Cropland Dataset: A New Land Cover Dataset for Multitemporal Deep Learning Classification in Agriculture. arXiv [Cs.CV]. Retrieved from http://arxiv.org/abs/2306.00114

The dataset is used to test different convolutional neural network (CNN) and quantum CNN (QCNN) architecture using only small number of images. Sample scripts for classification are provided. The selected images were manually selected to express some crop diversity from different Canadian provinces and seasons. The resulting dataset selection clustering using t-distributed Stochastic Neighbor Embedding (t-SNE) shows the diversity in the dataset images.

![dataset T-distributed Stochastic Neighbor Embedding overview](https://github.com/etiennelord/QuantumCropLand/blob/main/20250605_tSNE.svg)

# Dataset statistics
_Filename_: 20250605_16x16_12channel_ndvi_psri.npz

_Creation Date_: 2025-06-05

_Number of Datapoints_: 500 (100 for each class)

_Number of Channels_: 14

_Channel Size_: 16 x 16 pixels

_Channel Labels and order_: ['B4' 'B3' 'B2' 'B8' 'B1' 'B5' 'B6' 'B7' 'B8A' 'B9' 'B11' 'B12' 'NDVI'
 'PSRI']
 
_Classes_: ['BARLEY' 'CORN' 'OAT' 'ORCHARD' 'SOYBEAN']

_Month(s)_: July, August, September

_Year(s)_: 2019

_Points for each province_:

AB: 124
BC: 102
MB: 83
NB: 3
ON: 18
PE: 19
QC: 96
SK: 55

Data Vector Name: ‘data’

Label Vector Name: ‘labels’

Shape of data: (500, 14, 16, 16)

Shape of labels: (500,)

File Size: 7006 kB

# Sample results for the provided 2D-CNN

_Classification Report (Epochs 50, Seed 42, data division: 70-15-15%) :_

              precision    recall  f1-score   support

      Barley       0.78      0.75      0.77        24
        Corn       0.69      1.00      0.82         9
         Oat       0.71      0.71      0.71        14
     Orchard       0.82      0.88      0.85        16
     Soybean       0.75      0.50      0.60        12

    accuracy                           0.76        75
    macro avg      0.75      0.77      0.75        75
    weighted avg   0.76      0.76      0.75        75

Note that those classification results present low accuracy, in contrast to actual '_production_' machine learning models principaly due to the low data volume during the model training, no use of data augmentation, and inherent data diversity. 

We encourage researchers and practitioners to improve those classification results.

# Future 

This dataset and associated data is evolving. 

Future updates will include :
- Sentinel-1 data as well as elevation data.
- Some quantum image encoding information and benchmark.
- Some quantum classification algorithm and benchmark.


