# A-DEEP-LEARNING-BASED-APPROACH-FOR-MAPPING-SHRUBS-IN-ARCTIC-TUNDRA-FROM-VERY-HIGH-RESOLUTION-IMAGERY



A DEEP LEARNING-BASED APPROACH FOR MAPPING SHRUBS IN ARCTIC TUNDRA FROM VERY-HIGH RESOLUTION IMAGERY
Darko Radakovic, Mark Chopping, Bruce Cook, Aparna Varde and Stefan Robila


This repository contains preprocessing scripts, trained model weights (Advanced machine learning models, including Convolutional Neural Network (CNNs), U-Net, ResNet50, VGG19, and Vision Transformer (ViT)) for reproducibility of this manuscript. Representative datasets, Maxar imagery is subject to licensing restrictions, but can be requested from the authors.

For any questions please contact:
D. Radakovic with the Department of Earth and Environmental Studies, Montclair State University, Montclair, NJ 07043 USA (e-mail: radakovicd1@montclair.edu). 

This work was supported by National Aeronautics and Space Administration awards NNX15AU08A to MC and 80NSSC23K1559 to DR. Corresponding author: Darko Radakovic. 

Abstract:
Machine Learning (ML) provides great opportunities to analyze data at scale and generate new insights into complex phenomena. To build confidence their use, ML techniques require careful validation through extensive benchmarking. This study evaluated deep-learning models including the fundamental Convolutional Neural Network (CNN), and other computer vision advanced models (ResNet50, VGG19, U-Net, and Vision Transformer (ViT)) for detecting changes in shrub cover across Alaska’s North Slope using very-high-resolution QuickBird (QB) and WorldView (WV) satellite imagery spanning 2002 to 2020. Models were trained on geospatial data acquired by remote sensing entailing 4,100 4-band pansharpened multispectral and panchromatic image tiles (200-by-200 m) at top-of-atmosphere (TOA) radiance. ResNet50 and VGG19 achieved the highest accuracies (~86%) and balanced F1 scores (57 - 59%). Temporal trend analyses demonstrated an increase in shrub cover over time for all models, with the ResNet50 models showing consistently highest positive slopes (0.86% per year, p<0.001) while the CNN, VGG19, U-Net and ViT models showed more modest growth rates (0.08%, 0.29%, 0.25% and 0.04% per year respectively, p<0.001). Calibration against field-reference indicated strong predictive capability for ResNet50 and VGG19 models (R² = 0.78 - 0.81), with relatively low errors and biases. Comparisons with NASA’s Goddard's LiDAR, Hyperspectral, and Thermal (G-LiHT)-derived Canopy Height Model (CHM) had a weak relationship (R² = 0.18 - 0.27). The deep learning segmentation of shrub cover across heterogeneous Arctic landscapes highlights that the expansion aligns with documented Arctic warming trends, emphasizing the potential for deep learning-based approaches to advance the understanding of permafrost dynamics and vegetation-climate feedback in rapidly changing northern ecosystems. It highlights the adequate deployment of state-of-the-art computer vision models for image mining in geoscience. 
