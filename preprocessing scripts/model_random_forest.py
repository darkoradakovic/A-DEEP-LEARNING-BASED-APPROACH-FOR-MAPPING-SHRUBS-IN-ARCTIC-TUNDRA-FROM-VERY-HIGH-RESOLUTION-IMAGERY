
# --- --- 09/04/24 UPDATED---RANDOM FOREST


##### ------ ------- --------  CNN MODEL ALL YEARS MS6 with clustering ------ ------- -------- ------  #####

# Slicing larger 2000x2000m images
import os
import rasterio
from rasterio.windows import Window
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
import numpy as np
from rasterio.transform import from_bounds
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


from functionsdarko import (slice_image, load_sliced_images_and_metadata,
    process_and_store_predictions_with_validation4,
    process_and_store_predictions_with_validation4_smooth,
    process_and_store_predictions_with_validation5,
    process_and_store_predictions_with_validation6,
    process_images_evi5,
    process_and_store_predictions_with_validation44_4,
    process_and_store_predictions_with_validation44_6,
    cnn_segmentation_model, upgraded_cnn_comb_model, f1_score,
    apply_clustering, process_and_cluster_images,
    normalize_ndvi_dict
                                )


data_paths = {
    'REFLECTANCE': [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/x_train2_remove_brightness/REFLECTANCE',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train2_remove_brightness/REFLECTANCE',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area4_100m_clip_1950m_georef/x_train2_remove_brightness/REFLECTANCE',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/x_train2_remove_brightness/REFLECTANCE'
    ],
    'PCA': [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/x_train2_remove_brightness/PCA',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train2_remove_brightness/PCA',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area4_100m_clip_1950m_georef/x_train2_remove_brightness/PCA',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/x_train2_remove_brightness/PCA'
    ],
    'NDVI_REFLECTANCE': [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/x_train2_remove_brightness/NDVI',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train2_remove_brightness/NDVI',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area4_100m_clip_1950m_georef/x_train2_remove_brightness/NDVI',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/x_train2_remove_brightness/NDVI'
    ],
    'VALIDATION': [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/y_train_toolik',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/y_train_toolik',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area4_100m_clip_1950m_georef/y_train_toolik',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/y_train_toolik'
    ],
    # NDVI normal
    'NDDI': [
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/x_train_NDVI',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train_NDVI',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area4_100m_clip_1950m_georef/x_train_NDVI',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/x_train_NDVI'
    ]
}


# Load and process images from all paths
slices_REF = {}
metadata_dicts = {}
slices_NDVI = {}
slices_VAL = {}


for data_type, paths in data_paths.items():
    for path in paths:
        print(f"Loading {data_type} from {path}")
        slices, metadata_dict = load_sliced_images_and_metadata(path, 6 if 'VALIDATION' in data_type else 9, exclude_years=['WV02_2015', 'WV02_2018'])
        # slices, metadata_dict = load_sliced_images_and_metadata_resize2(path, 6 if 'VALIDATION' in data_type else 9, exclude_years=['WV02_2015', 'WV02_2018'])
        print(f"Loaded {len(slices)} entries for {data_type} from {path}")

        for key, meta in zip(slices, metadata_dict):
            print(f"Processing {len(slices[key])} slices for key: {key}")
            if data_type == 'NDDI':
                slices_NDVI.setdefault(key, []).extend(slices[key])
            elif data_type == 'REFLECTANCE':
                slices_REF.setdefault(key, []).extend(slices[key])
                metadata_dicts.setdefault(meta, []).extend(metadata_dict[meta])
            elif data_type == 'VALIDATION':
                slices_VAL.setdefault(key, []).extend(slices[key])
                # metadata_dicts.setdefault(meta, []).extend(metadata_dict[meta])


# gives equal amount per year
slices_REF_filtered, slices_VAL_filtered, slices_NDVI_filtered, metadata_dict_filtered = process_and_store_predictions_with_validation4(
    slices_REF, slices_VAL, slices_NDVI, metadata_dicts, classes=[5,11,12], exclude_years=['WV02_2015', 'WV02_2018'])
    # sliced_images_multiyear_REF, sliced_val_images_multiyear, sliced_ndvi_images_multiyear, classes=[5,8,10,11,12])
    # sliced_images_multiyear_REF, sliced_val_images_multiyear, sliced_ndvi_images_multiyear, classes=[13,14])  # WET TUNDRA



### Identify and remove empty slices

# Dictionary to hold non-empty images after removal
slices_REF_fil_noempty = {}
slices_VAL_fil_noempty = {}
slices_NDVI_fil_noempty = {}

# Set to track empty image indices across all years
empty_indices_global = set()

# Define a function to check if an image is empty (all zeros or NaNs)
def is_empty_image(image):
    return np.all(image == 0) or np.all(np.isnan(image))

# Step 1: Identify empty image indices for each year
for year in slices_REF_filtered.keys():
    ref_slices = slices_REF_filtered[year]  # Get REF images for that year

    # Check each image for emptiness
    for idx, image in enumerate(ref_slices):
        if is_empty_image(image):
            empty_indices_global.add(idx)  # Track this index as globally empty

# Step 2: Remove empty images across all years based on global empty indices
for year in slices_REF_filtered.keys():
    ref_slices = slices_REF_filtered[year]
    val_slices = slices_VAL_filtered[year]
    ndvi_slices = slices_NDVI_filtered[year]
    # ref_slices = slices_REF[year]
    # val_slices = slices_VAL['toolik']
    # ndvi_slices = slices_NDVI[year]

    # Filter out the images that are in empty_indices_global
    filtered_ref_slices = [image for idx, image in enumerate(ref_slices) if idx not in empty_indices_global]
    filtered_val_slices = [image for idx, image in enumerate(val_slices) if idx not in empty_indices_global]
    filtered_ndvi_slices = [image for idx, image in enumerate(ndvi_slices) if idx not in empty_indices_global]

    # Store the filtered images back into the new dictionaries
    slices_REF_fil_noempty[year] = filtered_ref_slices
    slices_VAL_fil_noempty[year] = filtered_val_slices
    slices_NDVI_fil_noempty[year] = filtered_ndvi_slices

# The new dictionaries now contain only the non-empty images and can be used for further processing
print(f"Images filtered, global empty indices: {empty_indices_global}")




## MANUALLY CHANGE HOW MANY CHANNELS
import numpy as np

# Placeholder lists for features and labels
features = []
labels = []

# Loop through each year (key) in the dictionary
for year in slices_REF_fil_noempty.keys():
    ref_slices = slices_REF_fil_noempty[year]  # Get list of REF images for that year (200, 200, 4)
    val_slices = slices_VAL_fil_noempty[year]  # Get list of VAL binary images (200, 200)
    ndvi_slices = slices_NDVI_fil_noempty[year]  # Get list of NDVI images (200, 200)

    # Ensure that the number of images matches across REF, VAL, and NDVI
    assert len(ref_slices) == len(val_slices) == len(ndvi_slices), "Mismatched number of images"

    # Iterate through each image in the year
    for i in range(len(ref_slices)):
        ref_image = ref_slices[i]  # Shape (200, 200, 4)
        val_image = val_slices[i]  # Shape (200, 200) - binary shrub labels
        ndvi_image = ndvi_slices[i]  # Shape (200, 200) - NDVI values

        # Flatten the image arrays into a list of features for each pixel
        # Each pixel will have 4 band values + 1 NDVI value = 5 features in total
        # Reshape the (200, 200, 4) image to (40000, 4) and the NDVI image to (40000,)
        ref_flattened = ref_image.reshape(-1, 4)  # (40000, 4)
        ndvi_flattened = ndvi_image.flatten()  # (40000,)
        val_flattened = val_image.flatten()  # (40000,) - binary labels

        ## MANUALLY
        # Stack the bands with the NDVI to create features (40000, 5)
        combined_features = ref_flattened  # (40000, 4)
        # combined_features = np.column_stack((ref_flattened, ndvi_flattened))  # (40000, 5)

        # Append the features and labels to the overall lists
        features.append(combined_features)
        labels.append(val_flattened)

# Convert the lists to numpy arrays
features = np.vstack(features)  # Shape will be (n_samples, 5) where n_samples = number of pixels across all images
labels = np.hstack(labels)  # Shape will be (n_samples,)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")



del slices_NDVI, slices_REF, slices_VAL
del slices_REF_filtered, slices_VAL_filtered, slices_NDVI_filtered, metadata_dict_filtered
del filtered_ref_slices, filtered_val_slices, filtered_ndvi_slices
del slices_REF_fil_noempty, slices_NDVI_fil_noempty, slices_VAL_fil_noempty





# RANDOM FOREST MODEL

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Assume you have the features and labels loaded as numpy arrays or pandas DataFrame
# features: an array of shape (n_samples, 5) where 4 bands + 1 NDVI per pixel
# labels: an array of shape (n_samples,) with binary values (0 or 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# del features,labels
# del X_train,y_train


# # Initialize the Random Forest classifier
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf = RandomForestClassifier(n_estimators=50, random_state=42)


# Define the function to train in chunks and save after each chunk
def train_rf_in_chunks(X_train, y_train, n_estimators=100, chunk_size=10, save_path="rf_model.pkl"):
    """
    Train Random Forest incrementally and save the model in chunks.

    Parameters:
    - X_train: Training data
    - y_train: Labels
    - n_estimators: Total number of trees to train
    - chunk_size: Number of trees to train in each chunk
    - save_path: File path to save the model
    """
    rf = None

    # Train in chunks
    for i in range(0, n_estimators, chunk_size):
        # Calculate the number of trees for this chunk
        n_estimators_chunk = min(chunk_size, n_estimators - i)

        # Train the model with this chunk of trees
        print(f"Training {n_estimators_chunk} trees (from {i} to {i + n_estimators_chunk})...")

        # Initialize the RandomForestClassifier with the current chunk size
        rf_chunk = RandomForestClassifier(n_estimators=n_estimators_chunk, warm_start=True, random_state=42)

        if rf is None:
            # First time, just fit the chunk
            rf = rf_chunk.fit(X_train, y_train)
        else:
            # Use the existing model and add more trees (warm start)
            rf.n_estimators += n_estimators_chunk
            rf.fit(X_train, y_train)

        # Save the model after each chunk
        joblib.dump(rf, save_path)
        print(f"Saved intermediate model after {i + n_estimators_chunk} trees to {save_path}.")

    print("Training completed!")
    return rf

# rf_model = train_rf_in_chunks(X_train, y_train, n_estimators=100, chunk_size=10, save_path="rf_model_intermediate.pkl")
# rf_model = train_rf_in_chunks(X_train, y_train, n_estimators=50, chunk_size=10, save_path="rf_model_intermediate2.pkl")
rf_model = train_rf_in_chunks(X_train, y_train, n_estimators=50, chunk_size=10, save_path="rf_model_intermediate3_REF_only.pkl")


# Load the existing model
# rf = joblib.load("rf_model_intermediate.pkl")  # 66GB keeps chrashing when loading
rf = joblib.load("rf_model_intermediate2.pkl")  #
rf = joblib.load("rf_model_intermediate3_REF_only.pkl")  #


# No checkpoint saving
# Train the model
# rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
# Calculate RMSE and R^2
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error RF model: {rmse}')
print(f'R^2 RF model: {r2_score(y_test, y_pred)}')

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)





# Feature importance
import matplotlib.pyplot as plt

# Get feature importance
importances = rf_model.feature_importances_
feature_names = ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'NDVI']

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='b')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest Model")
plt.show()


#
# MODEL: rf_model_intermediate2_REF_NDVI.pkl  (50 trees) REF+NDVI
# Accuracy: 0.8753
# Root Mean Squared Error RF model: 0.3531731161909128
# R^2 RF model: 0.050222785832781836
#
# Classification Report:
#               precision    recall  f1-score   support
#          0.0       0.90      0.96      0.93  17700504
#          1.0       0.66      0.41      0.51   3259496
#     accuracy                           0.88  20960000
#    macro avg       0.78      0.69      0.72  20960000
# weighted avg       0.86      0.88      0.86  20960000


# feature_names
# ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'NDVI']
# importances
# [0.12578326, 0.10575899, 0.19170499, 0.15429158, 0.42246118]


# MODEL: rf_model_intermediate3_REF_only.pkl (50 trees)
# Accuracy: 0.8711
# Root Mean Squared Error RF model: 0.3589948993875255
# R^2 RF model: 0.01865203105139135
# Confusion Matrix:
# [[17157618   542886]
#  [ 2158383  1101113]]
# Classification Report:
#               precision    recall  f1-score   support
#          0.0       0.89      0.97      0.93  17700504
#          1.0       0.67      0.34      0.45   3259496
#     accuracy                           0.87  20960000
#    macro avg       0.78      0.65      0.69  20960000
# weighted avg       0.85      0.87      0.85  20960000

# importances
# Out[13]: array([0.18771183, 0.16998492, 0.34641801, 0.29588524])
# feature_names
# Out[14]: ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'NDVI']






### PREDICTION INTERFERENCE TEST

import os
import rasterio
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Function to load raster images
def load_image(image_path):
    with rasterio.open(image_path) as src:
        crs = src.crs
        image = src.read()  # Read all bands
        transform = src.transform  # Get the transform for georeferencing later
    return image, transform, crs

# Paths to the images
#ms6a2 2009
pansharpened_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train2_remove_brightness/REFLECTANCE/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m_reflectance.tif'
ndvi_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train_NDVI/QB02_20090718220421_1010010009F45F00_NDVI_P001_NT_2000m_area2_GEOREF_clip1950m.tif'


#ms6a2 2017
pansharpened_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train2_remove_brightness/REFLECTANCE/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m_reflectance.tif'
ndvi_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train_NDVI/WV03_20170811223418_1040010031305A00_NDVI_P001_NT_2000m_area2_GEOREF_clip1950m.tif'


# Load pansharpened image (4 bands) and NDVI image
pansharpened_image, transform_ps, crs = load_image(pansharpened_image_path)  # Shape: (4, height, width)
# ndvi_image, transform_ndvi = load_image(ndvi_image_path)  # Shape: (1, height, width)


import rasterio
from rasterio.enums import Resampling
def resample_image(source_image_path, target_image_path):
    # Open the source (NDVI) image
    with rasterio.open(source_image_path) as src:
        # Get the profile of the source image
        src_profile = src.profile

        # Open the target (pansharpened) image to get its dimensions
        with rasterio.open(target_image_path) as target:
            target_profile = target.profile
            target_shape = (target.height, target.width)  # Target dimensions

            # Resample the source image (NDVI) to match the target (pansharpened)
            data_resampled = src.read(
                out_shape=(
                    src.count,  # Number of bands
                    target_shape[0],  # Target height
                    target_shape[1]   # Target width
                ),
                resampling=Resampling.bilinear  # Choose appropriate resampling method
            )

            # Update profile to match the target's resolution
            src_profile.update(
                width=target_shape[1],
                height=target_shape[0],
                transform=target.transform  # Transform for target image
            )

            # # Write the resampled image to disk
            # with rasterio.open(output_image_path, 'w', **src_profile) as dst:
            #     dst.write(data_resampled)

    # print(f"Resampled image saved to {output_image_path}")
    return data_resampled

# Resample NDVI to match the pansharpened image
ndvi_image = resample_image(ndvi_image_path, pansharpened_image_path)



# Check if image dimensions match
if pansharpened_image.shape[1:] != ndvi_image.shape[1:]:
    raise ValueError("Pansharpened image and NDVI image dimensions do not match.")

# Reshape the images for prediction
height, width = pansharpened_image.shape[1], pansharpened_image.shape[2]

# Reshape pansharpened image from (4, height, width) to (height * width, 4)
pansharpened_reshaped = pansharpened_image.reshape(4, -1).T  # Shape: (height * width, 4)

# Reshape NDVI image from (1, height, width) to (height * width, 1)
ndvi_reshaped = ndvi_image.reshape(1, -1).T  # Shape: (height * width, 1)

# Concatenate pansharpened image bands with NDVI values -> (height * width, 5)
features = np.concatenate([pansharpened_reshaped, ndvi_reshaped], axis=1)  # Shape: (height * width, 5)
# features = pansharpened_reshaped  # Shape: (height * width, 5)

# Load the saved Random Forest model
# rf_model = joblib.load('rf_model_intermediate2_REF_NDVI.pkl')
# rf_model = joblib.load('rf_model_intermediate2_REF_only.pkl')

# Make predictions on the image
predictions = rf_model.predict(features)

# Reshape the predictions back to the original image shape (height, width)
predicted_image = predictions.reshape(height, width)


## SAVE PREDICTION
# Optionally: Save the prediction as a new raster file
y_train_savedir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_test/large_image_test/RANDOMFOREST'
os.makedirs(y_train_savedir, exist_ok=True)
# output_path = "ms6a2_qb2009_test.tif"
output_path = pansharpened_image_path.split('/')[-1][:-4] + '_RANDOMFOREST_50t_REF_NDVI.tif'
# output_path = pansharpened_image_path.split('/')[-1][:-4] + '_RANDOMFOREST_50t_REF.tif'

output_path3 = os.path.join(y_train_savedir, output_path)

with rasterio.open(
    output_path3, 'w',
    driver='GTiff',
    height=predicted_image.shape[0],
    width=predicted_image.shape[1],
    count=1,
    dtype=predicted_image.dtype,
    crs=crs,  # Use CRS from the pansharpened image
    transform=transform_ps,
    nodata=0  # Set NoData value if necessary
) as dst:
    dst.write(predicted_image, 1)






import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.imshow(predicted_image)







#### -------------- ------------ OLD MULTISPECTRAL + NDVI -> TOOLIK OUT -- ST CNN------------ ------------ ------------
import numpy as np
import os
import rasterio
from collections import defaultdict
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def is_image_file(filename):
    return filename.endswith('.tif')

def validate_label_image(validation_image_path):
    with rasterio.open(validation_image_path) as src:
        validation_array = src.read(1)
        if (np.sum(validation_array == 1) / validation_array.size) > 0.5 or (
                np.sum(validation_array < 0.01) / validation_array.size) > 0.20:
            return False
    return True

def get_sequence_patch(x_train_dir, ndvi_train_dir, y_train_dir):
    x_train_sequences = []
    y_train_paths = []
    ndvi_train_sequences = []  # New list to hold NDVI sequences

    for label_image in sorted(os.listdir(y_train_dir)):
        if not is_image_file(label_image):
            continue
        label_image_path = os.path.join(y_train_dir, label_image)
        identifier = label_image[-9:-4]
        sequence_paths = []
        ndvi_sequence_paths = []  # New list to hold paths for NDVI images

        for year_dir in sorted(os.listdir(x_train_dir)):
            year_path = os.path.join(x_train_dir, year_dir)
            ndvi_year_path = os.path.join(ndvi_train_dir, year_dir.replace('PSHP','NDVI'))  # Path to NDVI images

            if not os.path.isdir(year_path):
            # if not os.path.isdir(year_path) or '2013' not in year_dir:
                continue

            images_for_year = [f for f in os.listdir(year_path) if f.endswith(identifier + '.tif')]
            ndvi_images_for_year = [f for f in os.listdir(ndvi_year_path) if 'NDVI' in f and f.endswith(identifier + '.tif')]  # Assuming NDVI images end with '_NDVI.tif'

            sequence_paths.extend([os.path.join(year_path, img) for img in images_for_year])
            ndvi_sequence_paths.extend([os.path.join(ndvi_year_path, img) for img in ndvi_images_for_year])

        x_train_sequences.append(sequence_paths)
        ndvi_train_sequences.append(ndvi_sequence_paths)
        y_train_paths.append(label_image_path)

    return x_train_sequences, ndvi_train_sequences, y_train_paths


def load_images_in_order(images_folder, ndvi_folder, labels_folder, target_shape=(200, 200)):

    x_train_sequences, ndvi_train_sequences, y_train_paths = get_sequence_patch(images_folder, ndvi_folder,labels_folder)

    # Initialize containers
    images_with_ndvi = []  # This will hold both multispectral and NDVI data
    labels = []

    # Iterate over sequences and corresponding labels
    for seq_paths, ndvi_seq_paths, lbl_path in zip(x_train_sequences, ndvi_train_sequences, y_train_paths):
        if not validate_label_image(lbl_path):
            continue  # Skip sequences where the label doesn't meet criteria

        # Load and preprocess multispectral and NDVI images
        sequence_images_with_ndvi = []  # Temp list for each sequence
        for ms_img_path, ndvi_img_path in zip(seq_paths, ndvi_seq_paths):
            ms_img = load_and_preprocess_ms_image(ms_img_path, target_shape)
            ndvi_img = load_and_preprocess_ndvi_image(ndvi_img_path, target_shape)
            # ndvi_img = np.expand_dims(ndvi_img, axis=-1)

            combined_img = np.concatenate((ms_img, ndvi_img), axis=-1)

            # Combine NDVI with multispectral images along the last axis
            # combined_img = np.concatenate((ms_img, ndvi_img[..., np.newaxis]), axis=-1)
            sequence_images_with_ndvi.append(combined_img)

        # Load and preprocess label
        label = load_and_preprocess_label(lbl_path, target_shape)
        # repeated_label = np.repeat(label[np.newaxis, :, :, :], len(seq_paths),axis=0)  # Repeat label for each time step in the sequence

        # Append processed sequence and label
        images_with_ndvi.append(np.stack(sequence_images_with_ndvi, axis=0))
        labels.append(label)
        # labels.append(repeated_label)

    return np.array(images_with_ndvi), np.array(labels)


def load_and_preprocess_ms_image(image_path, target_shape=(200, 200)):
    """
    Load and preprocess a multispectral image.
    """
    with rasterio.open(image_path) as src:
        image = src.read()  # Assuming the image has multiple bands and they are stored in the first dimension.
        image = np.transpose(image, (1, 2, 0))  # Transpose to (H, W, C) for compatibility with Keras/TensorFlow.
        image = resize(image, target_shape, mode='constant', preserve_range=True) / 255.0  # Normalize pixel values.
    return image

def load_and_preprocess_ndvi_image(image_path, target_shape=(200, 200)):
    """
    Load and preprocess an NDVI image.
    """
    with rasterio.open(image_path) as src:
        ndvi = src.read(1)  # Assuming NDVI image has a single band.
        ndvi = np.expand_dims(ndvi, axis=-1)  # Add a channel dimension to make it (H, W, 1).
        ndvi = resize(ndvi, target_shape, mode='constant', preserve_range=True)  # Preserve NDVI values.
        ndvi = (ndvi + 1) / 2
    return ndvi

def load_and_preprocess_label(label_path, target_shape=(200, 200), class_values=[5, 8, 10, 11, 12]):
    """
    Load and preprocess a label image. Converts labels to binary based on specified class values.
    """
    with rasterio.open(label_path) as src:
        label = src.read(1)  # Assuming label is a single-channel image.
        binary_label = np.isin(label, class_values).astype(np.float32)  # Convert to binary format.
        binary_label = resize(binary_label, target_shape, mode='constant', preserve_range=True, anti_aliasing=True)
        binary_label = np.expand_dims(binary_label, axis=-1)  # Add a channel dimension to make it (H, W, 1).
    return binary_label


### REDUCED MEMORY ST_CNN
def create_seq2seq_model3(channels: int) -> Model:
    x_input = Input(shape=(None, None, None, channels), dtype=tf.float32)

    x = ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True)(x_input)
    x = BatchNormalization()(x)
    x = tf.keras.activations.swish(x)

    x = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.swish(x)

    x = Conv3D(1, (1, 1, 1), activation='sigmoid')(x)

    return Model(inputs=x_input, outputs=x)


# Example usage:
root_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/test6_stcnn/toolik_train_new_stcnn_ms6a2'
# root_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/test6_stcnn/toolik_train_new_stcnn_ms6a2_smalltest'
# root_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/toolik_train_new_stcnn_ms6a2'
images_folder_X_train = os.path.join(root_dir, 'x_train')
images_folder_X_train_NDVI = os.path.join(root_dir, 'x_train_ndvi')
labels_folder_y_train = os.path.join(root_dir, 'y_train_toolik', 'toolik_veg_classmap_area2_clip1950m')

images_folder = images_folder_X_train
ndvi_folder = images_folder_X_train_NDVI
labels_folder = labels_folder_y_train
target_shape = (200, 200)

x_train, y_train = load_images_in_order(images_folder_X_train, images_folder_X_train_NDVI, labels_folder_y_train, target_shape=target_shape)








### ---- ---- ---- ---- ---- RANDOM FOREST CONVERSION ---- ---- ---- ----

import numpy as np

channels = 5

# Example for flattening one time step of x_train and corresponding y_train
time_step = 0  # Example: Use the first time step

# Flatten x_train for one time step
x_train_flattened = x_train[:, time_step, :, :, :].reshape(-1, 200*200*channels)  # Reshape to (num_samples, features)

num_samples = x_train.shape[0]
num_timesteps = x_train.shape[1]
height = x_train.shape[2]
width = x_train.shape[3]
channels = x_train.shape[4]


# Reshape x_train to treat each pixel across all channels as a single sample
# New shape will be (154 * 7 * 200 * 200, 5)
x_train_reshaped = x_train.reshape(-1, channels)
y_train_reshaped = y_train.reshape(-1, 1)




### FEATURED ENGINEERING
temporal_mean = np.mean(x_train, axis=1)  # shape becomes (154, 200, 200, 5)
temporal_std = np.std(x_train, axis=1)    # shape becomes (154, 200, 200, 5)
temporal_ndvi_mean = np.mean(x_train[:,:,:,:,-1], axis=1)  # NDVI is assumed to be the last channel
temporal_ndvi_std = np.std(x_train[:,:,:,:,-1], axis=1)

temporal_ndvi_mean_expanded = np.expand_dims(temporal_ndvi_mean, axis=-1)  # Shape becomes (154, 200, 200, 1)
temporal_ndvi_std_expanded = np.expand_dims(temporal_ndvi_std, axis=-1)    # Shape becomes (154, 200, 200, 1)


x_train_example = x_train[:,0,:,:,:]  # Shape is (154, 200, 200, 5)

# Concatenate the original x_train data with the expanded mean and std arrays
enhanced_x_train = np.concatenate((x_train_example, temporal_ndvi_mean_expanded, temporal_ndvi_std_expanded), axis=-1)


num_samples, height, width, channels = enhanced_x_train.shape
flattened_data = enhanced_x_train.reshape(-1, channels)  # Shape: (num_samples*height*width, channels)
flattened_labels = y_train.reshape(-1)  # Shape: (num_samples*height*width,)




### ---- ---- ---- ---- TRAIN RANDOM FOREST ---- ---- ---- ----
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# X_train, X_test, y_train, y_test = train_test_split(x_train_reshaped, y_train_reshaped, test_size=0.2, random_state=42)
#
#
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#
# rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Calculate RMSE and R^2
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error RF model: {rmse}')
print(f'R^2 RF model: {r2_score(y_test, y_pred)}')




# CHOOSE 10% of data

sample_size = int(0.1 * x_train_reshaped.shape[0])  # 10% of your data
sample_size = int(0.1 * flattened_data.shape[0])  # 10% of your data

# Generate random indices
indices = np.random.choice(flattened_data.shape[0], sample_size, replace=False)

# Use indices to sample data
x_train_sampled = flattened_data[indices]
y_train_sampled = flattened_labels[indices]

# Now split the sampled data
x_train3, x_test, y_train3, y_test = train_test_split(x_train_sampled, y_train_sampled, test_size=0.2, random_state=42)

# x_train4 = x_train3[:,4].reshape(-1,1)  # only NDVI

# Continue with training as before
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train3, y_train3.ravel())



# Predict
y_pred = rf_model.predict(x_test)

# x_test2 = x_test[:,4].reshape(-1,1)  # only NDVI
# y_pred = rf_model.predict(x_test2)

# Calculate RMSE and R^2
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error RF model: {rmse}')
print(f'R^2 RF model: {r2_score(y_test, y_pred)}')



# Feature importance
# feature_importance = pd.Series(rf_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
feature_importance = pd.Series(rf_model.feature_importances_)
print('Feature Importances:\n', feature_importance)



