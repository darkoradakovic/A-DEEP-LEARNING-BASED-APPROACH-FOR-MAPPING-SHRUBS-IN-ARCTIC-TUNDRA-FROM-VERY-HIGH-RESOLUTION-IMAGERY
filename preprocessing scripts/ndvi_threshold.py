

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Define function to read raster data
def read_raster(path):
    with rasterio.open(path) as src:
        return src.read()

from functionsdarko import (slice_image,combine_tiles_to_large_image_predictionsoneyear,
                            rasterize_shapefile_to_binary
                            )

def load_images_from_folder(folder_path, id, exclude_years):
    images = {}
    metadata_dict = {}
    for filename in sorted(os.listdir(folder_path), key=lambda x: x.split('_')[1][:8]):
        if filename.endswith('.tif'):
            identifier = filename[:id]
            if identifier not in exclude_years:

                if identifier not in images:
                    images[identifier] = []
                    metadata_dict[identifier] = []
            # if filename.endswith('.tif') and 'WV03_2016' in filename:
            # if filename.endswith('.tif') and 'QB02_2009' in filename:
            # if filename.endswith('.tif') and 'QB02_2013' in filename:
                image_path = os.path.join(folder_path, filename)
                with rasterio.open(image_path) as src:
                    image = src.read()  # Read the first band
                    images[identifier].append(image)
                    metadata_dict[identifier].append(src.meta)
    return images, metadata_dict

# Define function to calculate NDVI
def compute_ndvi(nir, red):
    return (nir - red) / (nir + red)

def compute_evi2(nir, red):
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1)

def compute_kndvi(nir, red):
    ndvi = compute_ndvi(nir, red)
    return np.tanh(ndvi)

def compute_nirv(nir, ndvi):
    return ndvi * nir

# Define function to plot comparison
def plot_comparison(ndvi, evi2, kndvi, nirv, title):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(ndvi, cmap='RdYlGn')
    plt.title(f'{title} - NDVI')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(evi2, cmap='RdYlGn')
    plt.title(f'{title} - EVI2')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(kndvi, cmap='RdYlGn')
    plt.title(f'{title} - kNDVI')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(nirv, cmap='RdYlGn')
    plt.title(f'{title} - NIRv')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def normalize_image(image):
    scaler = MinMaxScaler()
    image_reshaped = image.reshape(-1, 1)
    normalized_image = scaler.fit_transform(image_reshaped)
    return normalized_image.reshape(image.shape)


data_paths = {
    'REFLECTANCE': [
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/x_train2_remove_brightness/REFLECTANCE',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train2_remove_brightness/REFLECTANCE',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area4_100m_clip_1950m_georef/x_train2_remove_brightness/REFLECTANCE',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/x_train2_remove_brightness/REFLECTANCE'
    ],
    'PCA': [
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/x_train2_remove_brightness/PCA',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train2_remove_brightness/PCA',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area4_100m_clip_1950m_georef/x_train2_remove_brightness/PCA',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/x_train2_remove_brightness/PCA'
    ],
    'NDVI_REFLECTANCE': [
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/x_train2_remove_brightness/NDVI',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train2_remove_brightness/NDVI',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area4_100m_clip_1950m_georef/x_train2_remove_brightness/NDVI',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/x_train2_remove_brightness/NDVI'
    ],
    'VALIDATION': [
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/y_train_toolik',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/y_train_toolik',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area4_100m_clip_1950m_georef/y_train_toolik',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/y_train_toolik'
    ],
    # NDVI normal
    'NDDI': [
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/x_train_NDVI',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train_NDVI',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area4_100m_clip_1950m_georef/x_train_NDVI',
    # '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/x_train_NDVI'
    ]
}


exclude_years = ['WV02_2015', 'WV02_2018']

# Iterate through each folder and load timeseries images
# for reflectance_folder, ndvi_folder in zip(data_paths['REFLECTANCE'], data_paths['NDVI_REFLECTANCE']):
#     reflectance_images = load_images_from_folder(reflectance_folder)
#     ndvi_images = load_images_from_folder(ndvi_folder)
#
#     # Assuming reflectance_images and ndvi_images are lists of arrays, each array being a timeseries image
#     for reflectance_image, ndvi_image in zip(reflectance_images, ndvi_images):
#         # Assuming the bands order: Blue, Green, Red, NIR
#         blue, green, red, nir = reflectance_image
#
#         # NDVI image
#         ndvi = ndvi_image[0]  # Assuming NDVI is single band
#
#         # Compute vegetation indices
#         computed_ndvi = compute_ndvi(nir, red)
#         evi2 = compute_evi2(nir, red)
#         kndvi = compute_kndvi(nir, red)
#         nirv = compute_nirv(nir, computed_ndvi)
#
#         # Display or save the computed indices for comparison
#         plt.figure(figsize=(12, 8))
#         plt.subplot(2, 2, 1)
#         plt.title("NDVI")
#         plt.imshow(computed_ndvi, cmap='RdYlGn')
#         plt.colorbar()
#
#         plt.subplot(2, 2, 2)
#         plt.title("EVI2")
#         plt.imshow(evi2, cmap='RdYlGn')
#         plt.colorbar()
#
#         plt.subplot(2, 2, 3)
#         plt.title("kNDVI")
#         plt.imshow(kndvi, cmap='RdYlGn')
#         plt.colorbar()
#
#         plt.subplot(2, 2, 4)
#         plt.title("NIRv")
#         plt.imshow(nirv, cmap='RdYlGn')
#         plt.colorbar()
#
#         plt.tight_layout()
#         plt.show()







for reflectance_folder, ndvi_folder in zip(data_paths['REFLECTANCE'], data_paths['NDVI_REFLECTANCE']):

    ndvi_values = []
    evi_values = []
    kndvi_values = []
    nirv_values = []

    evi_images = {}

    ndvi_water_values = []

    reflectance_images, _ = load_images_from_folder(reflectance_folder, 9 , exclude_years)
    ndvi_images, metadata_dict = load_images_from_folder(ndvi_folder, 9, exclude_years)

    for reflectance_image, ndvi_image in zip(reflectance_images.values(), ndvi_images.items()):
        # Assuming the bands order: Blue, Green, Red, NIR
        blue, green, red, nir = reflectance_image[0]

        # NDVI image
        ndvi = ndvi_image[0]  # Assuming NDVI is single band

        # Compute vegetation indices
        computed_ndvi = compute_ndvi(nir, red)
        evi2 = compute_evi2(nir, red)
        kndvi = compute_kndvi(nir, red)
        nirv = compute_nirv(nir, computed_ndvi)

        ndvi_values.append(np.nanmean(ndvi_image[1]))
        evi_values.append(np.nanmean(evi2))
        kndvi_values.append(np.nanmean(kndvi))
        nirv_values.append(np.nanmean(nirv))
        ndvi_image2 = ndvi_image[1][0]
        ndvi_water_values.append(np.nanmean(ndvi_image2[ndvi_image2 < 0]))

        if ndvi_image[0] not in evi_images:  # ndvi_image[0] is "YEAR"
            evi_images[ndvi_image[0]] = []
        evi_images[ndvi_image[0]] = evi2

        # ndvi_values.append(np.nanmedian(ndvi_image[1]))
        # evi_values.append(np.nanmedian(evi2))
        # kndvi_values.append(np.nanmedian(kndvi))
        # nirv_values.append(np.nanmedian(nirv))



# Plotting the mean values of each index in one figure
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(ndvi_images.keys(), ndvi_values, label='NDVI', marker='o')
plt.title('NDVI Values')
plt.xlabel('Time', fontsize=9)
plt.xticks(fontsize=7)
plt.ylabel('Mean NDVI')
plt.ylim(0, 1)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(ndvi_images.keys(), ndvi_water_values, label='EVI2', marker='o')
plt.title('mean NDVI water Values')
plt.xlabel('Time')
plt.ylabel('Mean EVI2')
plt.xticks(fontsize=7)
# plt.ylim(0, 1)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(ndvi_images.keys(), kndvi_values, label='kNDVI', marker='o')
plt.title('kNDVI Values')
plt.xlabel('Time')
plt.ylabel('Mean kNDVI')
plt.xticks(fontsize=7)
plt.ylim(0, 1)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(ndvi_images.keys(), nirv_values, label='NIRv', marker='o')
plt.title('NIRv Values')
plt.xlabel('Time')
plt.ylabel('Mean NIRv')
plt.xticks(fontsize=7)
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()






ndvi_values2 = []
normalized_ndvi_images = {}
for filename, image in ndvi_images.items():
    normalized_image = normalize_image(image[0])
    ndvi_values2.append(np.nanmean(normalized_image))
    normalized_ndvi_images[filename] = normalized_image

def adjust_image_mean(image, target_mean):
    current_mean = np.nanmean(image)
    adjustment_value = target_mean - current_mean
    adjusted_image = image + adjustment_value
    return adjusted_image

target_mean = np.nanmean(ndvi_values)

ndvi_values2 = []
ndvi_images_adjusted = {}
for filename, image in ndvi_images.items():
    adjusted_image = adjust_image_mean(image[0], target_mean)
    ndvi_values2.append(np.nanmean(adjusted_image))
    ndvi_images_adjusted[filename] = adjusted_image


plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
plt.plot(ndvi_images.keys(), ndvi_values2, label='NDVI', marker='o')
plt.title('NDVI Values')
plt.xlabel('Time', fontsize=9)
plt.xticks(fontsize=7)
plt.ylabel('Mean NDVI')
plt.ylim(0, 1)
plt.legend()








for reflectance_image, ndvi_image in zip(ndvi_images_adjusted):
        # Assuming the bands order: Blue, Green, Red, NIR
        blue, green, red, nir = reflectance_image

        # NDVI image
        ndvi = ndvi_image[0]  # Assuming NDVI is single band

        # Compute vegetation indices
        computed_ndvi = compute_ndvi(nir, red)
        evi2 = compute_evi2(nir, red)
        kndvi = compute_kndvi(nir, red)
        nirv = compute_nirv(nir, computed_ndvi)

        # Display or save the computed indices for comparison
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.title("NDVI")
        plt.imshow(computed_ndvi, cmap='RdYlGn')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.title("EVI2")
        plt.imshow(evi2, cmap='RdYlGn')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.title("kNDVI")
        plt.imshow(kndvi, cmap='RdYlGn')
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.title("NIRv")
        plt.imshow(nirv, cmap='RdYlGn')
        plt.colorbar()

        plt.tight_layout()
        plt.show()





output_folder2 = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_test/Standardized_NDVI_ms6a5'


def save_adjusted_images(images, metadata_dict, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for identifier, image_list in images.items():
        for i, image in enumerate(image_list):
            print(image.shape)
            output_path = os.path.join(output_folder, f"{identifier}_{i}.tif")
            meta = metadata_dict[identifier][i]

            # Ensure the image is a 3D array with one band
            # if len(image.shape) == 2:
            # image = image[0]
            print("newshape: ", image.shape)

            with rasterio.open(output_path, 'w', **meta) as dst:
                # dst.write(image, 1)
                dst.write(image)


# save_adjusted_images(ndvi_images_adjusted, metadata_dict, output_folder)



for ndvi_folder in data_paths['NDVI_REFLECTANCE']:

    ndvi_images, metadict = load_images_from_folder(ndvi_folder, 9, exclude_years)

    # Compute the mean NDVI value for each image
    means = []
    for image_list in ndvi_images.values():
        for image in image_list:
            means.append(np.nanmean(image))

    # Determine the target mean NDVI value (mean of means)
    target_mean = np.nanmean(means)

    # Adjust NDVI images to have the same mean
    ndvi_images_adjusted = {}
    for identifier, image_list in ndvi_images.items():
        ndvi_images_adjusted[identifier] = []
        for image in image_list:
            adjusted_image = adjust_image_mean(image, target_mean)
            # adjusted_image = adjusted_image.reshape(adjusted_image.shape[1], adjusted_image.shape[2], 1)
            adjusted_image = adjusted_image.reshape(1, adjusted_image.shape[1], adjusted_image.shape[2])

            ndvi_images_adjusted[identifier].append(adjusted_image)

    # Save the adjusted NDVI images
    output_folder = os.path.join(output_folder2, 'adjusted')
    save_adjusted_images(ndvi_images_adjusted, metadict, output_folder)




image2 = image.reshape(image.shape[1], image.shape[2], 1)





from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from functionsdarko import (slice_image, load_sliced_images_and_metadata, load_sliced_REF_PCA_images,
    process_and_store_predictions_with_validation4,
    process_validation4,
    process_and_store_predictions_with_validation5,
    process_images_evi5,
    process_and_store_predictions_with_validation44_4,
    process_and_store_predictions_with_validation44_6,
    process_and_store_predictions_with_validation44_4cloud,
    cnn_segmentation_model, upgraded_cnn_comb_model, f1_score,
    calculate_ndvi, apply_clustering, process_and_cluster_images,
    make_prediction,
    dynamic_threshold_adjustment,
    evaluate_thresholds_for_year2,
    remove_edge_lines,
    calculate_metrics44_2,
    collect_image_paths2,
    specify_cloud_areas,
    cloud_check, predict_clouds,
    filter_slices
    )


### TEST SHRUB COVER
# New UNET model New Shrub (class 5, 11, 12 ) REF
model = load_model(
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/shrub_cluster_cl5_11_12_UNET_model_ms6_allareas_myears_clim_f1_REF.tf',
    custom_objects={'f1_score': f1_score}
)





# Load and process images from all paths
sliced_images_multiyear_REF = {}
sliced_images_multiyear_PCA = {}
sliced_images_multiyear_REFNDVI = {}
metadata_dicts = {}
sliced_val_images_multiyear = {}
sliced_ndvi_images_multiyear = {}

for data_type, paths in data_paths.items():
    for path in paths:
        print(f"Loading {data_type} from {path}")
        slices, metadata_dicts = load_sliced_images_and_metadata(path, 6 if 'VALIDATION' in data_type else 9)
        print(f"Loaded {len(slices)} entries for {data_type} from {path}")

        for key in slices:
            print(f"Processing {len(slices[key])} slices for key: {key}")
            if data_type == 'VALIDATION':
                sliced_val_images_multiyear.setdefault(key, []).extend(slices[key])
            elif data_type == 'NDDI':
                sliced_ndvi_images_multiyear.setdefault(key, []).extend(slices[key])
            elif data_type == 'REFLECTANCE':
                sliced_images_multiyear_REF.setdefault(key, []).extend(slices[key])
            elif data_type == 'NDVI_REFLECTANCE':
                sliced_images_multiyear_REFNDVI.setdefault(key, []).extend(slices[key])
            elif data_type == 'PCA':
                sliced_images_multiyear_PCA.setdefault(key, []).extend(slices[key])


sliced_ms_images_REF, sliced_val_images2, sliced_ndvi_images2 = process_and_store_predictions_with_validation4(
    sliced_images_multiyear_REF, sliced_val_images_multiyear, sliced_ndvi_images_multiyear, classes=[5,11,12])


sliced_ms_images_REFNDVI, sliced_val_images2, _, = process_and_store_predictions_with_validation4(
    sliced_images_multiyear_REFNDVI, sliced_val_images_multiyear, sliced_ndvi_images_multiyear, classes=[5,11,12])


# REF
sliced_ms_images_REFval, sliced_val_images5, sliced_ndvi_images5, metadata_dict5 = process_validation4(
    sliced_images_multiyear_REF, sliced_val_images_multiyear, sliced_ndvi_images_multiyear, metadata_dicts, resize_to=(200, 200), classes=[5, 11, 12])


# NDVI
# sliced_val_images4, _, _, sliced_ms_images_REFNDVI, _, _, = process_and_store_predictions_with_validation4(sliced_ms_images4, sliced_val_images, sliced_ndvi_images, metadata_dict)
sliced_ms_images_REFNDVIval, _, _, _, = process_validation4(
    sliced_images_multiyear_REFNDVI, sliced_val_images_multiyear, sliced_ndvi_images_multiyear, metadata_dicts, resize_to=(200, 200), classes=[5, 11, 12])


#
# # PCA
# # sliced_val_images3, _, _, sliced_ms_images_PCA, _, _, = process_and_store_predictions_with_validation4(sliced_ms_images3, sliced_val_images, sliced_ndvi_images, metadata_dict)
# sliced_ms_images_PCAval, sliced_val_images5, _, _, = process_validation4(
#     sliced_ms_images3validation, sliced_val_images2validation, sliced_ndvi_images2validation, metadata_dictvalidation, resize_to=(200, 200), classes=[5, 11, 12])
#     # sliced_ms_images3, sliced_val_images, sliced_ndvi_images, metadata_dict, resize_to=(200, 200), classes=[5, 8,10, 11, 12])
#     # sliced_ms_images3, sliced_val_images, sliced_ndvi_images, metadata_dict, resize_to=(200, 200), classes=[13,14])  # WET TUNDRA



### --- PREDICTION ---
# COMBINATION REF new cloud  --- # 430 sec for PCA
# predictions_dict, metadata_dict2, water_dict2 = process_and_store_predictions_with_validation44_4(sliced_ms_images_REF, sliced_ms_images_PCA, sliced_ms_images_REFNDVI, sliced_ndvi_images2, model)
predictions_dict, water_dict2 = process_and_store_predictions_with_validation44_6(
    sliced_ms_images_REFval, sliced_ms_images_REFNDVIval, sliced_ms_images_REFNDVIval, sliced_ndvi_images5, model, n_clusters=3)



for year, water_predictions in water_dict2.items():
    if year in predictions_dict:
        predictions_for_year = predictions_dict[year]
        for i, (prediction, water_prediction) in enumerate(zip(predictions_for_year, water_predictions)):
            predictions_dict[year][i][water_prediction == 1] = 0

shrub_cover_per_year = {}
predictions_dict_binary = {}
for year in predictions_dict:
    shrub_cover_per_year[year], predictions_dict_binary[year] = evaluate_thresholds_for_year2 \
        (predictions_dict[year], 'MEAN_p70')

predictions_dict_binary2 = {year: [] for year in predictions_dict_binary.keys()}
for year, predictions in predictions_dict_binary.items():
    for pred in predictions:
        cleaned_image = remove_edge_lines(pred)
        predictions_dict_binary2[year].append(cleaned_image)
metrics_per_year, shrub_cover_per_year, shrub_cover_per_year2 = calculate_metrics44_2(predictions_dict_binary2)




pixel_area_m2=0.5 * 0.5
shrub_cover_per_year3 = {}
exclude_years = ['WV02_2015', 'WV02_2018']

ndvi_threshold_per_year = {}
ndvi_mask_per_year = {}
ndvi_binary_per_year = {}

for year, shrub_binary_maps in shrub_cover_per_year2.items():
    if year in exclude_years:  # year='QB02_2009'; shrub_binary_maps=shrub_cover_per_year2[year]  ; year='WV03_2017'
        continue
    # print(year)
    shrub_cover_year = 0
    ndvi_threshold_year = []
    ndvi_mask_year = []
    ndvi_binary_year = []

    for i in range(len(shrub_binary_maps)):  # i=33 ; i = 43 # 43 is good
        ndvi_image = sliced_ms_images_REFNDVIval[year][i]
        # ndvi_image = np.squeeze(ndvi_image)
        # print(ndvi_image.shape)

        shrub_binary_map = shrub_cover_per_year2[year][i].astype(bool)
        # shrub_binary_map = np.squeeze(shrub_binary_map)
        # print(shrub_binary_map.shape)
        # shrub_binary_map = shrub_binary_map.astype(bool)
        # shrub_binary_map = shrub_binary_maps[i].astype(bool)

        ndvi_image_mask = ndvi_image[shrub_binary_map]
        # print(ndvi_image_mask.shape)

        ndvi_image_shrub_mean = np.nanmean(ndvi_image_mask)   #  i=33  0.4358535 'QB02_2009'; i=33  0.5469089  'WV03_2017'
        # ndvi_image_shrub_mean = np.nanmean(ndvi_image)    # i=33  0.45210987 'QB02_2009';  i=33 0.5264116  'WV03_2017'
        lower_boundary_shrub_ndvi = np.nanpercentile(ndvi_image_mask, 20)  # i=33  0.40051956176  'QB02_2009'; i=33 0.50030363  'WV03_2017'
        # lower_boundary_shrub_ndvi = np.nanpercentile(ndvi_image, 20)  # i=33  0.4373421728610992 'QB02_2009';

        ndvi_mask_year.append(ndvi_image_mask)
        ndvi_threshold_year.append(ndvi_image_shrub_mean)
        # ndvi_threshold_year.append(lower_boundary_shrub_ndvi)

        #THRESHOLDING
        ndvi_shrub_binary_map = ndvi_image > ndvi_image_shrub_mean
        # ndvi_shrub_binary_map = ndvi_image > lower_boundary_shrub_ndvi

        ndvi_binary_year.append(ndvi_shrub_binary_map)

        # Calculate shrub cover percentage from NDVI binary maps
        ndvi_shrub_cover_percentage = np.sum(ndvi_shrub_binary_map) / ndvi_image.size

        # Compare with UNET-based shrub cover
        unet_shrub_cover_percentage = np.sum(shrub_binary_map) / shrub_binary_map.size

        # print(f"NDVI-based shrub cover: {ndvi_shrub_cover_percentage * 100:.2f}%")
        # print(f"UNET-based shrub cover: {unet_shrub_cover_percentage * 100:.2f}%")

        shrub_cover = np.sum(ndvi_shrub_binary_map) * pixel_area_m2
        shrub_cover_year += shrub_cover

        # print(f"{(cover / (len(predictions_dict[year])*(100 * 100))):.2f}")
    # print(f"{(shrub_cover_year / (len(shrub_cover_per_year2[year]) * (100 * 100))):.2f}")
    total_shrub_cover_percentage = shrub_cover_year / (len(shrub_binary_maps) * ndvi_image.size * pixel_area_m2)
    print(f"Year: {year}, Shrub cover percentage: {total_shrub_cover_percentage * 100:.2f}%")

    shrub_cover_per_year3[year] = shrub_cover_year
    ndvi_threshold_per_year[year] = ndvi_threshold_year
    ndvi_mask_per_year[year] = ndvi_mask_year
    ndvi_binary_per_year[year] = ndvi_binary_year



## TOO BAD
threshold_ndvi = []
for year, shrub_binary_maps in ndvi_threshold_per_year.items():
    threshold_ndvi.append(np.nanmean(shrub_binary_maps))


# Thresholding with NDVI mean of entire year
pixel_area_m2=0.5 * 0.5
shrub_cover_per_year = {}
ndvi_binary_per_year_entireimg = {}
exclude_years = ['WV02_2015', 'WV02_2018']
index = 0
for year, shrub_binary_maps in shrub_cover_per_year2.items():
    if year in exclude_years:  # year='QB02_2009'
        continue
    shrub_cover_year = 0
    ndvi_mask_year = []
    ndvi_binary_year = []

    for i in range(len(shrub_binary_maps)):
        ndvi_image = sliced_ms_images_REFNDVIval[year][i]

        #THRESHOLDING
        # ndvi_shrub_binary_map = ndvi_image > ndvi_image_shrub_mean
        ndvi_shrub_binary_map = ndvi_image > threshold_ndvi[index]

        ndvi_binary_year.append(ndvi_shrub_binary_map)

        shrub_cover = np.sum(ndvi_shrub_binary_map) * pixel_area_m2
        shrub_cover_year += shrub_cover

    index += 1

    total_shrub_cover_percentage = shrub_cover_year / (len(shrub_binary_maps) * ndvi_image.size * pixel_area_m2)
    print(f"Year: {year}, Shrub cover percentage: {total_shrub_cover_percentage * 100:.2f}%")

    shrub_cover_per_year[year] = shrub_cover_year
    ndvi_mask_per_year[year] = ndvi_mask_year
    ndvi_binary_per_year_entireimg[year] = ndvi_binary_year


# # Thresholding with NDVI mean of entire year
# pixel_area_m2=0.5 * 0.5
# shrub_cover_per_year = {}
# exclude_years = ['WV02_2015', 'WV02_2018']
# index = 0
# for year, ndvi_image in ndvi_images.items():
#     if year in exclude_years:  # year='QB02_2009'
#         continue
#     shrub_cover_year = 0
#     ndvi_mask_year = []
#
#     # ndvi_image = combined_images_per_year2[year]
#     # shrub_binary_map = combined_images_per_year[year].astype(bool)
#     # ndvi_image_mask = ndvi_image[shrub_binary_map]
#     # print(ndvi_image_mask.shape)
#
#     ndvi_image_shrub_mean = np.nanmean(ndvi_image_mask)
#
#
#
#     #THRESHOLDING
#     # ndvi_shrub_binary_map = ndvi_image > ndvi_image_shrub_mean
#     ndvi_shrub_binary_map = ndvi_image[0] > threshold_ndvi[index]
#
#
#     # print(f"NDVI-based shrub cover: {ndvi_shrub_cover_percentage * 100:.2f}%")
#     # print(f"UNET-based shrub cover: {unet_shrub_cover_percentage * 100:.2f}%")
#
#     shrub_cover = np.sum(ndvi_shrub_binary_map) * pixel_area_m2
#     shrub_cover_year += shrub_cover
#
#     index += 1
#
#         # print(f"{(cover / (len(predictions_dict[year])*(100 * 100))):.2f}")
#     # print(f"{(shrub_cover_year / (len(shrub_cover_per_year2[year]) * (100 * 100))):.2f}")
#     total_shrub_cover_percentage = shrub_cover_year / (ndvi_image[0].size * pixel_area_m2)
#     print(f"Year: {year}, Shrub cover percentage: {total_shrub_cover_percentage * 100:.2f}%")
#
#     shrub_cover_per_year[year] = shrub_cover_year
#     ndvi_mask_per_year[year] = ndvi_mask_year



## PSHP boundaries from prediction

pixel_area_m2=0.5 * 0.5
shrub_cover_per_year3 = {}
exclude_years = ['WV02_2015', 'WV02_2018']

REF_threshold_per_year = {}
REF_mask_per_year = {}
REF_binary_per_year = {}

for year, shrub_binary_maps in shrub_cover_per_year2.items():
    if year in exclude_years:  # year='QB02_2009'; shrub_binary_maps=shrub_cover_per_year2[year]  ; year='WV03_2017'
        continue
    # print(year)
    shrub_cover_year = 0
    REF_threshold_year = []
    REF_mask_year = []
    REF_binary_year = []

    for i in range(len(shrub_binary_maps)):  # i=33 ; i = 43 # 43 is good
        REF_image = sliced_ms_images_REFval[year][i]
        # ndvi_image = np.squeeze(ndvi_image)
        # print(ndvi_image.shape)

        shrub_binary_map = shrub_cover_per_year2[year][i].astype(bool)
        # shrub_binary_map = np.squeeze(shrub_binary_map)
        # print(shrub_binary_map.shape)
        # shrub_binary_map = shrub_binary_map.astype(bool)
        # shrub_binary_map = shrub_binary_maps[i].astype(bool)

        REF_image_mask = REF_image[shrub_binary_map]
        REF_image_mask = REF_image[:, :, :][shrub_binary_map]
        np.nanmean(REF_image)
        np.nanpercentile(REF_image, LOW)
        np.nanpercentile(REF_image, UP)
        REF_image_maskband0 = REF_image[:, :, 0].reshape(200,200,1)[shrub_binary_map]
        REF_image_maskband1 = REF_image[:, :, 1].reshape(200,200,1)[shrub_binary_map]
        REF_image_maskband2 = REF_image[:, :, 2].reshape(200,200,1)[shrub_binary_map]
        REF_image_maskband3 = REF_image[:, :, 3].reshape(200,200,1)[shrub_binary_map]
        # print(ndvi_image_mask.shape)

        # REF_image_shrub_mean = np.nanmean(REF_image_mask)   #  i=33  0.4358535 'QB02_2009'; i=33  0.5469089  'WV03_2017'
        # # ndvi_image_shrub_mean = np.nanmean(ndvi_image)    # i=33  0.45210987 'QB02_2009';  i=33 0.5264116  'WV03_2017'
        # lower_boundary_shrub = np.nanpercentile(REF_image_mask, 20)  # i=33  0.40051956176  'QB02_2009'; i=33 0.50030363  'WV03_2017'
        # lower_boundary_shrub_ndvi = np.nanpercentile(ndvi_image, 20)  # i=33  0.4373421728610992 'QB02_2009';

        LOW = 30
        UP = 70

        lower_boundary_shrubband0 = np.nanpercentile(REF_image_maskband0, LOW)
        upper_boundary_shrubband0 = np.nanpercentile(REF_image_maskband0, UP)
        lower_boundary_shrubband1 = np.nanpercentile(REF_image_maskband1, LOW)
        upper_boundary_shrubband1 = np.nanpercentile(REF_image_maskband1, UP)
        lower_boundary_shrubband2 = np.nanpercentile(REF_image_maskband2, LOW)
        upper_boundary_shrubband2 = np.nanpercentile(REF_image_maskband2, UP)
        lower_boundary_shrubband3 = np.nanpercentile(REF_image_maskband3, LOW)
        upper_boundary_shrubband3 = np.nanpercentile(REF_image_maskband3, UP)

        REF_image_shrub_mean0 = np.nanmean(REF_image_maskband0)
        REF_image_shrub_mean1 = np.nanmean(REF_image_maskband1)
        REF_image_shrub_mean2 = np.nanmean(REF_image_maskband2)
        REF_image_shrub_mean3 = np.nanmean(REF_image_maskband3)


        # REF_mask_year.append(REF_image_mask)
        # REF_threshold_year.append(REF_image_mask)
        # ndvi_threshold_year.append(lower_boundary_shrub_ndvi)

        #THRESHOLDING
        # REF_shrub_binary_map = (REF_image[:, :, 0] > lower_boundary_shrubband0) & (REF_image[:, :, 0] < upper_boundary_shrubband0) & \
        #     (REF_image[:, :, 1] > lower_boundary_shrubband1) & (REF_image[:, :, 1] < upper_boundary_shrubband1) & \
        #     (REF_image[:, :, 2] > lower_boundary_shrubband2) & (REF_image[:, :, 2] < upper_boundary_shrubband2) & \
        #     (REF_image[:, :, 3] > lower_boundary_shrubband3) & (REF_image[:, :, 3] < upper_boundary_shrubband3)

        REF_shrub_binary_map = (REF_image[:, :, 0] > REF_image_shrub_mean0) & \
            (REF_image[:, :, 1] > REF_image_shrub_mean1) & (REF_image[:, :, 2] > REF_image_shrub_mean2) & \
            (REF_image[:, :, 3] > REF_image_shrub_mean3)


        REF_binary_year.append(REF_shrub_binary_map)

        shrub_cover = np.sum(REF_shrub_binary_map) * pixel_area_m2
        shrub_cover_year += shrub_cover

        # print(f"{(cover / (len(predictions_dict[year])*(100 * 100))):.2f}")
    # print(f"{(shrub_cover_year / (len(shrub_cover_per_year2[year]) * (100 * 100))):.2f}")
    total_shrub_cover_percentage = shrub_cover_year / (len(shrub_binary_maps) * REF_image[:,:,0].size * pixel_area_m2)
    print(f"Year: {year}, Shrub cover percentage: {total_shrub_cover_percentage * 100:.2f}%")

    shrub_cover_per_year3[year] = shrub_cover_year
    REF_threshold_per_year[year] = REF_threshold_year
    REF_mask_per_year[year] = REF_mask_year
    REF_binary_per_year[year] = REF_binary_year














ndvi_image = ndvi_images['QB02_2009']

ndvi_values = ndvi_image[0].flatten()

# Optionally, remove NaN values if they exist
ndvi_values = ndvi_values[~np.isnan(ndvi_values)]

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(ndvi_values, bins=50, color='green', alpha=0.7)
plt.title('Histogram of NDVI Values')
plt.xlabel('NDVI Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


ndvi_image_mask



ndvi_mask_per_year2 = []
for year, data in ndvi_mask_year.items():
    # for data in shrub_binary_maps:
        ndvi_mask_per_year2.append(data)

ndvi_mask_per_year3 = np.concatenate(ndvi_mask_year)

ndvi_values2 = ndvi_mask_per_year3[~np.isnan(ndvi_mask_per_year3)]

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(ndvi_values2, bins=50, color='green', alpha=0.7)
plt.title('Histogram of NDVI Values')
plt.xlabel('NDVI Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

import rasterio







### SAVE TILES


import numpy as np
import rasterio
from rasterio.transform import Affine
from skimage.transform import resize



def combine_tiles_to_large_image(ndvi_binary_per_year, metadata_dict_per_year):
    combined_images_per_year = {}
    transformation_info_per_year = {}

    for year, ndvi_images in ndvi_binary_per_year.items():
        year_metadata = metadata_dict_per_year[year]

        # Initialize variables to determine the size of the combined image
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        # Calculate the extents of the combined image
        for meta in year_metadata:
            transform = meta['transform']
            width, height = meta['width'], meta['height']

            # Top-left corner (origin)
            x0, y0 = transform * (0, 0)
            # Bottom-right corner
            x1, y1 = transform * (width, height)

            min_x, min_y = min(min_x, x0), min(min_y, y1)
            max_x, max_y = max(max_x, x1), max(max_y, y0)

        # Calculate the dimensions of the combined image
        pixel_size_x = year_metadata[0]['transform'][0]  # Assumes uniform pixel size
        pixel_size_y = -year_metadata[0]['transform'][4]  # Assumes uniform pixel size and y-scale is negative

        combined_width = int(np.ceil((max_x - min_x) / pixel_size_x))
        combined_height = int(np.ceil((max_y - min_y) / pixel_size_y))

        # Create an empty array for the combined image
        combined_image = np.zeros((combined_height, combined_width), dtype=np.uint8)

        # Place each tile in the correct position in the combined image
        for ndvi_image, meta in zip(ndvi_images, year_metadata):
            transform = meta['transform']
            expected_width, expected_height = meta['width'], meta['height']

            # Resize the tile if necessary
            ndvi_image_resized = resize(np.squeeze(ndvi_image), (expected_height, expected_width), preserve_range=True, anti_aliasing=False).astype(np.uint8)

            # Calculate the top-left corner of where this tile will go
            x0, y0 = transform * (0, 0)

            # Calculate the pixel offsets in the combined image
            x_offset = int((x0 - min_x) / pixel_size_x)
            y_offset = int((max_y - y0) / pixel_size_y)

            # Insert the resized tile into the combined image
            combined_image[y_offset:y_offset+expected_height, x_offset:x_offset+expected_width] = ndvi_image_resized

        combined_images_per_year[year] = combined_image
        transformation_info_per_year[year] = {
            'min_x': min_x,
            'max_y': max_y,
            'pixel_size_x': pixel_size_x,
            'pixel_size_y': pixel_size_y
        }

    return combined_images_per_year, transformation_info_per_year



def save_combined_image(combined_image, output_path, metadata_dict, transformation_info):
    # Adjust the metadata for the combined image
    meta = metadata_dict[0].copy()  # Copy the metadata from the first tile
    min_x = transformation_info['min_x']
    max_y = transformation_info['max_y']
    pixel_size_x = transformation_info['pixel_size_x']
    pixel_size_y = transformation_info['pixel_size_y']

    meta.update({
        'height': combined_image.shape[0],
        'width': combined_image.shape[1],
        'transform': Affine.translation(min_x, max_y) * Affine.scale(pixel_size_x, -pixel_size_y)
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(combined_image, 1)

# Example usage:
# combined_images_per_year, transformation_info_per_year = combine_tiles_to_large_image(ndvi_binary_per_year, metadata_dict5)
# combined_images_per_year, transformation_info_per_year = combine_tiles_to_large_image(ndvi_binary_per_year_entireimg, metadata_dict5)
combined_images_per_year, transformation_info_per_year = combine_tiles_to_large_image(REF_binary_per_year, metadata_dict5)

output_folder = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_test/ndvi_test"
os.makedirs(output_folder, exist_ok=True)

for year, combined_image in combined_images_per_year.items():
    # output_path = os.path.join(output_folder, f"comb_ndvi_entireimg_{year}.tif")
    output_path = os.path.join(output_folder, f"REF_binary_{year}.tif")
    save_combined_image(combined_image, output_path, metadata_dict5[year], transformation_info_per_year[year])




shrub_cover_per_year3 = {}
for year, shrub_binary_maps in shrub_cover_per_year2.items():
    if year in exclude_years:  # year='QB02_2009'; shrub_binary_maps=shrub_cover_per_year2[year]
        continue
    shrub_cover_year = []
    shrub_cover_per_year3[year] = []
    for i in range(len(shrub_binary_maps)):
        # shrub_binary_maps
        shrub_cover_year.append(shrub_binary_maps[i])

    shrub_cover_per_year3[year] = shrub_cover_year


# Save predictions
combined_images_per_year, transformation_info_per_year = combine_tiles_to_large_image(shrub_cover_per_year3, metadata_dict5)
# combined_images_per_year2, _ = combine_tiles_to_large_image(sliced_ms_images_REFNDVI, metadata_dict5)

output_folder = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_test/ndvi_test"
os.makedirs(output_folder, exist_ok=True)

for year, combined_image in combined_images_per_year.items():
    output_path = os.path.join(output_folder, f"shrubcover_binary_{year}.tif")
    save_combined_image(combined_image, output_path, metadata_dict5[year], transformation_info_per_year[year])








# SAVE TILES WITH ANNOTATION

import numpy as np
import rasterio
from rasterio.transform import Affine
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont

def combine_tiles_to_large_image(ndvi_binary_per_year, metadata_dict_per_year):
    combined_images_per_year = {}
    transformation_info_per_year = {}

    for year, ndvi_images in ndvi_binary_per_year.items():
        year_metadata = metadata_dict_per_year[year]

        # Initialize variables to determine the size of the combined image
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        # Calculate the extents of the combined image
        for meta in year_metadata:
            transform = meta['transform']
            width, height = meta['width'], meta['height']

            # Top-left corner (origin)
            x0, y0 = transform * (0, 0)
            # Bottom-right corner
            x1, y1 = transform * (width, height)

            min_x, min_y = min(min_x, x0), min(min_y, y1)
            max_x, max_y = max(max_x, x1), max(max_y, y0)

        # Calculate the dimensions of the combined image
        pixel_size_x = year_metadata[0]['transform'][0]  # Assumes uniform pixel size
        pixel_size_y = -year_metadata[0]['transform'][4]  # Assumes uniform pixel size and y-scale is negative

        combined_width = int(np.ceil((max_x - min_x) / pixel_size_x))
        combined_height = int(np.ceil((max_y - min_y) / pixel_size_y))

        # Create an empty array for the combined image
        combined_image = np.zeros((combined_height, combined_width), dtype=np.uint8)

        # Place each tile in the correct position in the combined image
        count = 0
        for ndvi_image, meta in zip(ndvi_images, year_metadata):
            transform = meta['transform']
            expected_width, expected_height = meta['width'], meta['height']

            # Resize the tile if necessary
            ndvi_image_resized = resize(np.squeeze(ndvi_image), (expected_height, expected_width), preserve_range=True, anti_aliasing=False).astype(np.uint8)

            # Calculate the top-left corner of where this tile will go
            x0, y0 = transform * (0, 0)

            # Calculate the pixel offsets in the combined image
            x_offset = int((x0 - min_x) / pixel_size_x)
            y_offset = int((max_y - y0) / pixel_size_y)

            # Insert the resized tile into the combined image
            combined_image[y_offset:y_offset+expected_height, x_offset:x_offset+expected_width] = ndvi_image_resized

            # Add text to the combined image
            pil_image = Image.fromarray(combined_image)
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.load_default()
            # font = 11
            text = f"Tile {count}"
            draw.text((x_offset + 10, y_offset + 10), text, font=font, fill=255)  # Adjust position and color as needed
            combined_image = np.array(pil_image)
            count += 1

        combined_images_per_year[year] = combined_image
        transformation_info_per_year[year] = {
            'min_x': min_x,
            'max_y': max_y,
            'pixel_size_x': pixel_size_x,
            'pixel_size_y': pixel_size_y
        }

    return combined_images_per_year, transformation_info_per_year

def save_combined_image(combined_image, output_path, metadata_dict, transformation_info):
    # Adjust the metadata for the combined image
    meta = metadata_dict[0].copy()  # Copy the metadata from the first tile
    min_x = transformation_info['min_x']
    max_y = transformation_info['max_y']
    pixel_size_x = transformation_info['pixel_size_x']
    pixel_size_y = transformation_info['pixel_size_y']

    meta.update({
        'height': combined_image.shape[0],
        'width': combined_image.shape[1],
        'transform': Affine.translation(min_x, max_y) * Affine.scale(pixel_size_x, -pixel_size_y)
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(combined_image, 1)

# Example usage:
combined_images_per_year, transformation_info_per_year = combine_tiles_to_large_image(ndvi_binary_per_year, metadata_dict5)

# output_folder = "/path/to/output/folder"
os.makedirs(output_folder, exist_ok=True)

for year, combined_image in combined_images_per_year.items():
    output_path = os.path.join(output_folder, f"combined_text_{year}.tif")
    save_combined_image(combined_image, output_path, metadata_dict5[year], transformation_info_per_year[year])















#### --- --- --- NEW TRAINING UNET WITH OTHER MS sites, removing water to match predictions other years  --- --- --- ####
### ---  UNET_TRAINING_DATA_v2



import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image

# predictions binary
# binary_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_test/UNET_TRAINING_DATA_v2/y_train/QB02_20050724225630_1010010004654800_PSHP_P002_NT_2000m_area1.tif'
binary_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_test/large_image_test/shrub_cluster_cl5_11_12_UNET_model_ms6_allareas_myears_clim_f1_REF/QB02_2002_shrubmap_predictions_area1_ms2.tif'
# for replacing part of binary from better binary (like cloud)
binary_image_path2 = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_test/large_image_test/shrub_cluster_cl5_11_12_UNET_model_ms6_allareas_myears_clim_f1_REF_NDVI.tf/ms1/WV02_20130820230414_1030010025076500_PSHP_P003_NT_2000m_area1.tif'

pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_test/UNET_TRAINING_DATA_v2/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2.tif'

output_folder = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train2_tundra"
os.makedirs(output_folder, exist_ok=True)


# ms1a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20050724225626_1010010004654800_PSHP_P001_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20130820230414_1030010025076500_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140707224109_1030010033485600_PSHP_P007_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170813230621_1040010031CAA600_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20200716223511_10300100A97DD900_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20220619223632_10300100D5894700_PSHP_P007_NT_2000m_area1.tif'
# ms2a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020731213451_1010010000E6BE00_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140810214641_1030010034AC5200_PSHP_P006_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20200822213156_10300100AA78CA00_PSHP_P004_NT_2000m_area1.tif'
# ms2a13
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020731213451_1010010000E6BE00_PSHP_P003_NT_2000m_area13.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140810214641_1030010034AC5200_PSHP_P006_NT_2000m_area13.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20200822213156_10300100AA78CA00_PSHP_P004_NT_2000m_area13.tif'
# ms5a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20030610212157_1010010001F8FC00_PSHP_P001_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20100623215741_103001000594C600_PSHP_P007_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20220710215253_1040010079D00200_PSHP_P009_NT_2000m_area1.tif'
# ms6a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area1.tif'
# ms6a2
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'
# ms10a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20050724225630_1010010004654800_PSHP_P002_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140707224108_1030010033485600_PSHP_P006_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170813230622_1040010031CAA600_PSHP_P004_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20220619223631_10300100D5894700_PSHP_P006_NT_2000m_area1.tif'
# ms12a9
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020802215611_1010010000EC0A00_PSHP_P001_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20120722232542_1030010019B33400_PSHP_P002_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20200606222801_104001005D57A500_PSHP_P007_NT_2000m_area9.tif'
# ms13a2
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020802215615_1010010000EC0A00_PSHP_P002_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20080817223311_101001000872CD00_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20200606222804_104001005D57A500_PSHP_P009_NT_2000m_area2.tif'
# ms16a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20040804220250_1010010003255300_PSHP_P007_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20120620230434_1030010019D3EF00_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20160725220350_103001005A56ED00_PSHP_P003_NT_2000m_area1.tif'
# ms16a9
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20040804220250_1010010003255300_PSHP_P007_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20120620230434_1030010019D3EF00_PSHP_P003_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20160725220350_103001005A56ED00_PSHP_P003_NT_2000m_area9.tif'

# # test ms16a9 2020 low sun elevation
# pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/multisites/TIMESERIES/timeseries_ms16/timeseries_multisite16_areas/timeseries_multisite16_6_280323_165325/WV02_20200703070116_10300100A88C8A00_P007_area9_-1911376_4151947/WV02_20200703070116_10300100A88C8A00_PSHP_P007_NT_2000m_area9.tif'
# p1bs_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/multisites/TIMESERIES/timeseries_ms16/timeseries_multisite16_areas/timeseries_multisite16_6_280323_165325/WV02_20200703070116_10300100A88C8A00_P007_area9_-1911376_4151947/WV02_20200703070116_10300100A88C8A00_P1BS_P007_NT_2000m_area9.tif'



pshp_image = load_image(pshp_image_path)
p1bs_image_path = pshp_image_path.replace('x_train','P1BS_files').replace('PSHP','P1BS')


ndvi_image_path = pshp_image_path.replace('x_train','ndvi').replace('PSHP','NDVI')
ndvi_image = load_image(ndvi_image_path)

# water_mask = ndvi_image < 0.1
water_mask = ndvi_image < 0  # captures more shrubs

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(pshp_image[3,:,:]))



## STEP 0
target_slice_size_meters = 200
slices_pshp, metadata = slice_image(pshp_image_path, target_slice_size_meters)
slices_p1bs, _ = slice_image(p1bs_image_path, target_slice_size_meters)
# stacking PSHP + P1BS
slices_pshp = np.stack(slices_pshp)
slices_p1bs = np.stack(slices_p1bs)
slices_pshp = np.moveaxis(slices_pshp, 1, -1)
slices_p1bs = np.moveaxis(slices_p1bs, 1, -1)
x_train = np.concatenate((slices_pshp, slices_p1bs), axis=-1)  # x_train now has shape (num_samples, height, width, 5)
# y_train = y_train.astype('float32')
# Normalize
x_train = x_train / 255.0



## predictions on stacked PSHP P1BS
predictions = model.predict(x_train,  verbose=1)
combined_image, transformation_info = combine_tiles_to_large_image_predictionsoneyear(predictions, metadata)


plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(combined_image))



# predictions 512fil model (wettundra cnn_model3)
predictions = []
for batch in np.array_split(x_train, 10):  # Split into smaller chunks
    predictions_batch = model.predict(batch, verbose=1)
    predictions.append(predictions_batch)
predictions = np.concatenate(predictions, axis=0)

combined_image, transformation_info = combine_tiles_to_large_image_predictionsoneyear(predictions, metadata)


plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(combined_image))



combined_predictionimage_p70 = combined_image > np.percentile(combined_image,70)
combined_predictionimage_p80 = combined_image > np.percentile(combined_image,80)
combined_predictionimage_p90 = combined_image > np.percentile(combined_image,90)
combined_predictionimage_p95 = combined_image > np.percentile(combined_image,95)


# v7 shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS ep13
file_namep70 = pshp_image_path.split('/')[-1].replace('.tif','_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep8_p70_v13.tif')
file_namep80 = pshp_image_path.split('/')[-1].replace('.tif','_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep8_p80_v13.tif')
file_namep90 = pshp_image_path.split('/')[-1].replace('.tif','_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep8_p90_v13.tif')
file_namep95 = pshp_image_path.split('/')[-1].replace('.tif','_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep8_p95_v13.tif')


output_folder = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2"


# output_image_path = os.path.join(output_folder, file_name)
# print(file_name)
combined_image3 = [combined_predictionimage_p70, combined_predictionimage_p80, combined_predictionimage_p90, combined_predictionimage_p95]
for idx, savename in enumerate([file_namep70,file_namep80, file_namep90, file_namep95]):
    output_image_path = os.path.join(output_folder, savename)
    print(savename)
    with rasterio.open(pshp_image_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32,count=1)
        with rasterio.open(output_image_path, 'w', **profile) as dst:
            dst.write(combined_image3[idx].reshape(1, combined_image3[idx].shape[0], combined_image3[idx].shape[1]))
            # dst.write(combined_image2.reshape(1, combined_image2.shape[0], combined_image2.shape[1]))





# Load images
binary_image = load_image(binary_image_path)
# binary_image2 = load_image(binary_image_path2)
# section_size = 1000
# binary_image2_section = binary_image2[:, :section_size, :section_size]

plt.figure(figsize=(10, 10))
plt.imshow(np.clip(np.squeeze(ndvi_image),0,1))  # You can choose any colormap (e.g., 'viridis', 'plasma')

plt.figure(figsize=(10, 10))
plt.imshow(np.clip(np.squeeze(ndvi_image),-1,0))  # You can choose any colormap (e.g., 'viridis', 'plasma')


# section_size = 1000
# ndvi_image_section = ndvi_image[:, :section_size, 700:3600]
# ndvi_image_section2 = ndvi_image[:, :1260, 700:2300]
# ndvi_image_section2 = ndvi_image[:, :1400, 700:2300]
# ndvi_image_section3 = ndvi_image[:, :600, 2400:3200]
# ndvi_image_section4 = ndvi_image[:, :1400, 3200:4000]
# ndvi_image_section5 = ndvi_image[:, 600:1400, 2400:3200]

# Create a mask where NDVI values indicate water (customize this threshold as needed)
water_mask = ndvi_image < 0.1


### DETECTING LAKES with NDVI (NOT GREAT)
### See ndvi_lakes_rivers.py file for more details

modified_binary_image = (ndvi_image <= -0.1).astype(int)
modified_binary_image = (ndvi_image <= 0).astype(int)
modified_binary_image = (ndvi_image <= 0.2).astype(int)

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(modified_binary_image))

### DETECTING LAKES PSHP

## ms2a1 2002 south
modified_binary_image2 = (
    # (ndvi_image < 0.3) &
    (pshp_image[0, :, :] > 26) & (pshp_image[0, :, :] < 40) &
    (pshp_image[1, :, :] > 19) & (pshp_image[1, :, :] < 30)
    # (pshp_image[2, :, :] > 13) & (pshp_image[2, :, :] < 21) &
    # (pshp_image[3, :, :] > 15) & (pshp_image[3, :, :] < 43)
).astype(int)
modified_binary_image2 = (modified_binary_image2 > 0).astype(int)

## ms2a1 2002 north
modified_binary_image2 = (
    # (ndvi_image < 0.3) &
    (pshp_image[0, :, :] > 38) & (pshp_image[0, :, :] < 45) &
    (pshp_image[1, :, :] > 28) & (pshp_image[1, :, :] < 35) &
    # (pshp_image[2, :, :] > 18) & (pshp_image[2, :, :] < 21) &
    (pshp_image[3, :, :] > 40) & (pshp_image[3, :, :] < 48)
).astype(int)
modified_binary_image2 = (modified_binary_image2 > 0).astype(int)

print(np.sum(modified_binary_image2)/modified_binary_image2.size*100)

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(modified_binary_image2))



## SMOOTH
from scipy.ndimage import binary_closing
from skimage.morphology import remove_small_objects
smoothed_binary_image = binary_closing(modified_binary_image2, structure=np.ones((3, 3)))

# Remove small objects (connected components smaller than 5 pixels)
# Convert the binary image to boolean type for the function
smoothed_binary_image = smoothed_binary_image.astype(bool)
final_binary_image = remove_small_objects(smoothed_binary_image, min_size=80)

# Convert back to integer type if needed
final_binary_image = final_binary_image.astype(int)

# Print the percentage of detected area
print(np.sum(final_binary_image) / final_binary_image.size * 100)

# Plot the final binary image
plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(final_binary_image), cmap='gray')
plt.show()




## SAVE WATER BODIES LAKES / NDVI
output_folder = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train_lakes"
os.makedirs(output_folder, exist_ok=True)

# file_name = pshp_image_path.split('/')[-1].replace('.tif','_shape7CNNp95v2_shape8CNNp80v5_shape9CNNp90v2_shape2fill_remove.tif')  ###
# file_name = pshp_image_path.split('/')[-1]
# file_name = pshp_image_path.split('/')[-1].replace('.tif','_NDVI_t0.2.tif')  ###
# file_name = pshp_image_path.split('/')[-1].replace('.tif','_PSHPwaterpix.tif')  ###
# file_name = pshp_image_path.split('/')[-1].replace('.tif','_PSHPwaterpix_b3_b4_removed.tif')  ###
file_name = pshp_image_path.split('/')[-1].replace('.tif','_PSHPwaterpix_b3_b4_removed_smooth.tif')  ###
file_name = pshp_image_path.split('/')[-1].replace('.tif','_PSHPwaterpix_b3_b4_removed_smooth_north.tif')  ###

output_image_path = os.path.join(output_folder, file_name)
print(file_name)

# Save using rasterio
with rasterio.open(pshp_image_path) as src:
    profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_image_path, 'w', **profile) as dst:
        ## -------->>>>> MANUALLY CHANGE ------------------
        # dst.write(modified_binary_image3.reshape(1,4000,4000))   ## -------->>>>> MANUALLY CHANGE
        # dst.write(modified_binary_image2.reshape(1, 4000, 4000))  ## -------->>>>> MANUALLY CHANGE
        dst.write(final_binary_image.reshape(1, 4000, 4000))  ## -------->>>>> MANUALLY CHANGE SMOOTH



modified_binary_image = (ndvi_image > 0.45).astype(int)  # ms6a2 2009
modified_binary_image = (ndvi_image > 0.41).astype(int)  # ms6a2 2011
modified_binary_image = (ndvi_image > 0.53).astype(int)  # ms6a2 2013
modified_binary_image = (ndvi_image > 0.5).astype(int)  # ms6a2 2016
modified_binary_image = (ndvi_image >= 0.4).astype(int)  # ms6a2 2017 tundra
modified_binary_image = (ndvi_image >= 0.35).astype(int)  # ms6a2 2017 tundra




# modified_binary_image = (ndvi_image > 0.38).astype(int)  # ms2a1 2002
# modified_binary_image = (ndvi_image > 0.49).astype(int)  # ms2a1 2014
# modified_binary_image = (ndvi_image > 0.345).astype(int)  # ms2a1 2020
# modified_binary_image = (ndvi_image > 0.298).astype(int)  # ms2a13 2002
# modified_binary_image = (ndvi_image > 0.40).astype(int)  # ms2a13 2014
# modified_binary_image = (ndvi_image > 0.15).astype(int)  # ms2a13 2020
# modified_binary_image = (ndvi_image > 0.21).astype(int)  # ms5a1 2002
# modified_binary_image = (ndvi_image > 0.43).astype(int)  # ms5a1 2010
# modified_binary_image = (ndvi_image > 0.3).astype(int)  # ms5a1 2022
# modified_binary_image = (ndvi_image > 0.42).astype(int)  # ms12a9 2002
# modified_binary_image = (ndvi_image > 0.57).astype(int)  # ms12a9 2012
# modified_binary_image = (ndvi_image > 0.46).astype(int)  # ms13a2 2002
# modified_binary_image = (ndvi_image > 0.44).astype(int)  # ms13a2 2008
# modified_binary_image1 = (ndvi_image > 0.42).astype(int)  # ms13a2 2020
# modified_binary_image2 = ((ndvi_image < 0.18) & (ndvi_image > 0.05)).astype(int)  # ms13a2 2020
# modified_binary_image = modified_binary_image1 + modified_binary_image2  # ms13a2 2020
# modified_binary_image = (ndvi_image > 0.56).astype(int)  # ms16a1 2004
# modified_binary_image = (ndvi_image > 0.35).astype(int)  # ms16a1 2012
# modified_binary_image = (ndvi_image > 0.5).astype(int)  # ms16a9 2004
# modified_binary_image = (ndvi_image > 0.24).astype(int)  # ms16a9 2012
# modified_binary_image = (ndvi_image > 0.19).astype(int)  # ms16a9 2016
modified_binary_image = (ndvi_image > 0.45).astype(int)  # ms6a1 2009
modified_binary_image = (ndvi_image > 0.41).astype(int)  # ms6a1 2011
modified_binary_image = (ndvi_image > 0.53).astype(int)  # ms6a1 2013
modified_binary_image = (ndvi_image > 0.5).astype(int)  # ms6a1 2016
modified_binary_image = (ndvi_image > 0.52).astype(int)  # ms6a1 2017
modified_binary_image = (ndvi_image > 0.65).astype(int)  # ms10a1 2005

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(modified_binary_image))

modified_binary_image2 = ((ndvi_image < 0.35) & \
    (pshp_image[:,:,0] > 34) & (pshp_image[:,:,0] < 40) & (pshp_image[:,:,1] > 26) & (pshp_image[:,:,1] < 30) \
    (pshp_image[:,:,2] > 17) & (pshp_image[:,:,2] < 22) & (pshp_image[:,:,3] > 37) & (pshp_image[:,:,3] < 52)
    ).astype(int)

modified_binary_image2 = (
    (ndvi_image > 0.28) &
    (pshp_image[0, :, :] > 34) & (pshp_image[0, :, :] < 40) &
    (pshp_image[1, :, :] > 26) & (pshp_image[1, :, :] < 30) &
    (pshp_image[2, :, :] > 15) & (pshp_image[2, :, :] < 20) &
    (pshp_image[3, :, :] > 30) & (pshp_image[3, :, :] < 52)
).astype(int)

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(modified_binary_image2))



plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(modified_binary_image))

# modified_binary_image = (ndvi_image > 0.32).astype(int)
modified_binary_image2 = (ndvi_image[:, :1400, 700:2300] > 0.33).astype(int)
modified_binary_image3 = (ndvi_image[:, :600, 2400:3200] > 0.29).astype(int)
modified_binary_image4 = (ndvi_image[:, 600:1400, 3200:4000] > 0.35).astype(int)
modified_binary_image5 = (ndvi_image[:, 600:1500, 2400:3200] > 0.32).astype(int)
modified_binary_image6 = (ndvi_image[:, :600, 3200:4000] > 0.3).astype(int)
modified_binary_image2 = (ndvi_image[:, 2200:4000, 2200:4000] > 0.35).astype(int)    # ms16a9 2004
modified_binary_image2 = (ndvi_image[:, 2200:4000, 2200:4000] > 0.05).astype(int)    # ms16a9 2012
modified_binary_image2 = (ndvi_image[:, 2200:4000, 2200:4000] > 0.01).astype(int)    # ms16a9 2016

# with shapefile mask
boolean_mask = (rasterized_image > 0)
boolean_mask2 = (rasterized_image == 0)
modified_binary_image7 = np.where(boolean_mask, ndvi_image, 0)
modified_binary_image8 = (modified_binary_image7 > 0.34).astype(int)
modified_binary_image9 = np.where(boolean_mask, modified_binary_image8, 0)
modified_binary_image10 = np.where(boolean_mask2, modified_binary_image, 0)
modified_binary_image = modified_binary_image9 + modified_binary_image10

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(modified_binary_image))


# Apply the mask to the binary image
modified_binary_image = np.where(water_mask, 0, binary_image)
modified_binary_image1 = np.where(water_mask, 0, modified_binary_image)
# modified_binary_image[:, :section_size, :section_size] = binary_image2_section
# modified_binary_image[:, :section_size, 700:3600] = modified_binary_image2
modified_binary_image[:, :1400, 700:2300] = modified_binary_image2
modified_binary_image[:, :600, 2400:3200] = modified_binary_image3
modified_binary_image[:, 600:1400, 3200:4000] = modified_binary_image4
modified_binary_image[:, 600:1500, 2400:3200] = modified_binary_image5
modified_binary_image[:, :600, 3200:4000] = modified_binary_image6
modified_binary_image[:, 2200:4000, 2200:4000] = modified_binary_image2

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(modified_binary_image))

# Save the processed image
file_name = ndvi_image_path.split('/')[-1].replace('NDVI','PSHP')
file_name = 'WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2_t0.35.tif'
file_name = 'WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2.tif'
file_name = 'WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2_v2.tif'
file_name = 'QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'
file_name = 'WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif'
file_name = 'WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'

output_image_path = os.path.join(output_folder, file_name)
print(file_name)

# Save using rasterio
with rasterio.open(ndvi_image_path) as src:
    with rasterio.open(output_image_path, 'w', **src.meta) as dst:
        dst.write(modified_binary_image)






### Shapefile to binary

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import os

def rasterize_shapefile_to_binary(shapefile_path, tiff_path):
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    if gdf.empty:
        print(f"{shapefile_path} is empty.")
        return

    # Read the TIFF to use as a template for the raster dimensions, transform, and metadata
    with rasterio.open(tiff_path) as src:
        meta = src.meta.copy()
        transform = src.transform

    # Update metadata for the output file
    meta.update({
        'dtype': 'uint8',
        'count': 1,
        'compress': 'lzw'
    })

    # Rasterize the polygons
    rasterized_image = rasterize(
        [(shape, 1) for shape in gdf.geometry],
        out_shape=(meta['height'], meta['width']),
        transform=transform,
        fill=0,  # Background fill value
        all_touched=True,  # Rasterize all pixels touched by geometries
        dtype='uint8'
    )
    return rasterized_image

# ms1a1
shapefile_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_test/UNET_TRAINING_DATA_v2/shapefiles/QB02_20020731213451_1010010000E6BE00_PSHP_P003_NT_2000m_area13_shapefile.shp'
shapefile_path = pshp_image_path.replace('x_train','shapefiles_shrub').replace('.tif','.shp')


rasterized_image = rasterize_shapefile_to_binary(shapefile_path, ndvi_image_path)

# del rasterized_image, modified_binary_image, modified_binary_image1, modified_binary_image2, modified_binary_image3, modified_binary_image4, modified_binary_image5, modified_binary_image6, modified_binary_image7, modified_binary_image8, modified_binary_image9, modified_binary_image10


## Convert Community Veg Map to Binary
def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image

toolik_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/y_train_toolik/toolik_veg_classmap_area1_clip1950m.tif'
toolik_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/y_train_toolik/toolik_veg_classmap_area1_clip1950m.tif'
toolik_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/y_train_toolik/toolik_veg_classmap_area1_clip1950m.tif'
toolik_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area1_100m_clip_1950m_georef/y_train_toolik/toolik_veg_classmap_area1_clip1950m.tif'

toolik_im = load_image(toolik_path)

toolik_binary = np.isin(toolik_im, [5,11,12]).astype(int)  # SHRUB CLASSES
toolik_binary2 = resize(np.moveaxis(toolik_binary, 0, -1), (4000,4000), preserve_range=True,anti_aliasing=True)
toolik_binary2 = np.moveaxis(toolik_binary2, -1, 0).astype(int)


plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(toolik_im))

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(toolik_binary))



# Save the processed image
# file_name = ndvi_image_path.split('/')[-1].replace('NDVI','PSHP')
file_name = 'QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area1.tif'
file_name = 'WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area1.tif'
file_name = 'QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area1.tif'
file_name = 'WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area1.tif'
file_name = 'WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area1.tif'
output_folder = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/output_test/UNET_TRAINING_DATA_v2/y_train2_waterremoved'
output_image_path = os.path.join(output_folder, file_name)
print(file_name)

# Save using rasterio
with rasterio.open(toolik_path) as src:
    with rasterio.open(output_image_path, 'w', **src.meta) as dst:
        dst.write(toolik_binary2)







### RASTERIZE WET TUNDRA SHAPEFILE TO BINARY BATCH

wettundra3_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/shapefiles_wettundra3'
output_folder_wettundra = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train_wettundra3'
pshp_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train'

# lakes dir
wettundra3_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/shapefiles_lakes'
output_folder_wettundra = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train_lakes'


# rivers dir
wettundra3_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/shapefiles_rivers'
output_folder_wettundra = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train_rivers'


for file_name in os.listdir(pshp_dir):
    if file_name.endswith('.tif'):
        # Search for a matching shapefile
        image_path_shapefile = None
        for yfilename in os.listdir(wettundra3_dir):
            if file_name[:-4] in yfilename and yfilename.endswith('.shp'):
                image_path_shapefile = os.path.join(wettundra3_dir, yfilename)
                break  # Stop searching once the correct shapefile is found

        # If no corresponding shapefile is found, skip this iteration
        if image_path_shapefile is None:
            print(f"No matching shapefile found for {file_name}. Skipping.")
            continue

        # If shapefile is found, proceed with rasterizing
        pshp_image_path = os.path.join(pshp_dir, file_name)

        try:
            # Rasterize the shapefile
            rasterized_image = rasterize_shapefile_to_binary(image_path_shapefile, pshp_image_path)
        except Exception as e:
            print(f"Error rasterizing {file_name} with shapefile {image_path_shapefile}: {e}")
            continue

        # Save the rasterized image using rasterio
        output_image_path2 = os.path.join(output_folder_wettundra, file_name)
        print(f"Processing and saving {file_name}")

        try:
            with rasterio.open(pshp_image_path) as src:
                profile = src.profile
                profile.update(dtype=rasterio.uint8, count=1)
                with rasterio.open(output_image_path2, 'w', **profile) as dst:
                    dst.write(rasterized_image.reshape(1, 4000, 4000))  ## Adjust if different dimensions
        except Exception as e:
            print(f"Error saving rasterized image for {file_name}: {e}")












## ## ## --- --- --- CLUSTERING
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image

def cluster_masked_image(masked_image, n_clusters=5):
    # Reshape the masked image for clustering
    masked_image = np.moveaxis(masked_image, 0, -1)
    reshaped_image = masked_image.reshape(-1, masked_image.shape[-1])
    # reshaped_image = masked_image
    # Standardize the data
    scaler = StandardScaler()
    reshaped_image_scaled = scaler.fit_transform(reshaped_image)
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reshaped_image_scaled)
    # Reshape labels back to the original shape
    clustered_image = labels.reshape(masked_image.shape[:-1])
    # clustered_image = labels
    return clustered_image



# ms6a2
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'
# ms5a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20030610212157_1010010001F8FC00_PSHP_P001_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20100623215741_103001000594C600_PSHP_P007_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20220710215253_1040010079D00200_PSHP_P009_NT_2000m_area1.tif'
# ms12a9
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020802215611_1010010000EC0A00_PSHP_P001_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20120722232542_1030010019B33400_PSHP_P002_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20200606222801_104001005D57A500_PSHP_P007_NT_2000m_area9.tif'
# ms16a9
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20040804220250_1010010003255300_PSHP_P007_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20120620230434_1030010019D3EF00_PSHP_P003_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20160612215015_1030010057A6E600_PSHP_P012_NT_2000m_area9.tif'
# ms2a13
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020731213451_1010010000E6BE00_PSHP_P003_NT_2000m_area13.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140810214641_1030010034AC5200_PSHP_P006_NT_2000m_area13.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20200822213156_10300100AA78CA00_PSHP_P004_NT_2000m_area13.tif'
# ms13a2
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020802215615_1010010000EC0A00_PSHP_P002_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20080817223311_101001000872CD00_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20200606222804_104001005D57A500_PSHP_P009_NT_2000m_area2.tif'
# ms16a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20040804220250_1010010003255300_PSHP_P007_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20120620230434_1030010019D3EF00_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20160612215015_1030010057A6E600_PSHP_P012_NT_2000m_area1.tif'


pshp_image = load_image(pshp_image_path)
ndvi_image_path = pshp_image_path.replace('x_train', 'ndvi').replace('PSHP', 'NDVI')
ndvi_image = load_image(ndvi_image_path)
water_mask = ndvi_image < 0.1

# shapefile_basepath = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/shapefiles_wettundra/'
# # shapefile_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/shapefiles_wettundra/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2.shp'
# shapefile_path = os.path.join(shapefile_basepath, pshp_image_path.split('/')[-1].replace('.tif','.shp'))
# shapefile_binary = rasterize_shapefile_to_binary(shapefile_path, pshp_image_path)



# pshp_image_clustered2 = cluster_masked_image(np.moveaxis(pshp_image,0,-1), n_clusters=10)
pshp_image_clustered2 = cluster_masked_image(pshp_image, n_clusters=10)
# pshp_image_clustered2 = cluster_masked_image(pshp_image, n_clusters=20)

plt.figure(figsize=(10, 10))
plt.imshow(pshp_image_clustered2, cmap='viridis')  # You can choose any colormap (e.g., 'viridis', 'plasma')
plt.show()


# REF ms6a2
modified_binary_image2 = (pshp_image_clustered2 >= 2).astype(int) & (pshp_image_clustered2 <= 3).astype(int)
# RAD ms6a2 n_clusters=10
modified_binary_image2 = (pshp_image_clustered2 >= 4).astype(int) & (pshp_image_clustered2 <= 6).astype(int)
modified_binary_image2 = (pshp_image_clustered2 == 4).astype(int)

# RAD ms6a2 n_clusters=20
# modified_binary_image2 = (pshp_image_clustered2 >= 4).astype(int) & (pshp_image_clustered2 <= 6).astype(int)
modified_binary_image2 = (pshp_image_clustered2 == 17).astype(int) + (pshp_image_clustered2 == 8).astype(int) + (pshp_image_clustered2 == 15).astype(int) + (pshp_image_clustered2 == 0).astype(int) # 2009
modified_binary_image2 = (pshp_image_clustered2 == 5).astype(int) + (pshp_image_clustered2 == 13).astype(int) + (pshp_image_clustered2 == 19).astype(int) # 2011
modified_binary_image2 = (pshp_image_clustered2 == 12).astype(int) + (pshp_image_clustered2 == 17).astype(int) + (pshp_image_clustered2 == 19).astype(int) + (pshp_image_clustered2 == 4).astype(int)  # 2013
modified_binary_image2 = (pshp_image_clustered2 == 18).astype(int) + (pshp_image_clustered2 == 15).astype(int) + (pshp_image_clustered2 == 6).astype(int)    # 2016
modified_binary_image2 = (pshp_image_clustered2 == 18).astype(int) + (pshp_image_clustered2 == 4).astype(int) + (pshp_image_clustered2 == 0).astype(int)    # 2017
# RAD ms5a1 n_clusters=10
modified_binary_image2 = (pshp_image_clustered2 == 9).astype(int) + (pshp_image_clustered2 == 4).astype(int) + (pshp_image_clustered2 == 2).astype(int) + (pshp_image_clustered2 == 5).astype(int) + (pshp_image_clustered2 == 6).astype(int)    # 2003
modified_binary_image2 = (pshp_image_clustered2 == 8).astype(int) + (pshp_image_clustered2 == 7).astype(int) + (pshp_image_clustered2 == 6).astype(int) + (pshp_image_clustered2 == 0).astype(int) + (pshp_image_clustered2 == 1).astype(int)  + (pshp_image_clustered2 == 2).astype(int)    # 2010
modified_binary_image2 = (pshp_image_clustered2 == 5).astype(int) + (pshp_image_clustered2 == 0).astype(int) + (pshp_image_clustered2 == 2).astype(int) + (pshp_image_clustered2 == 7).astype(int) + (pshp_image_clustered2 == 6).astype(int)  + (pshp_image_clustered2 == 4).astype(int)    # 2003
# RAD ms12a9 n_clusters=10
modified_binary_image2 = (pshp_image_clustered2 == 9).astype(int) + (pshp_image_clustered2 == 4).astype(int) + (pshp_image_clustered2 == 3).astype(int) + (pshp_image_clustered2 == 5).astype(int) + (pshp_image_clustered2 == 0).astype(int)    # 2002
modified_binary_image2 = (pshp_image_clustered2 == 8).astype(int) + (pshp_image_clustered2 == 7).astype(int) + (pshp_image_clustered2 == 3).astype(int) + (pshp_image_clustered2 == 2).astype(int) + (pshp_image_clustered2 == 4).astype(int)    # 2012
modified_binary_image2 = (pshp_image_clustered2 == 8).astype(int) + (pshp_image_clustered2 == 0).astype(int) + (pshp_image_clustered2 == 4).astype(int)     # 2020
# RAD ms16a9 n_clusters=10
modified_binary_image2 = (pshp_image_clustered2 == 9).astype(int)  # 2004
modified_binary_image2 = (pshp_image_clustered2 == 8).astype(int)  # 2012
modified_binary_image2 = (pshp_image_clustered2 == 2).astype(int) + (pshp_image_clustered2 == 8).astype(int)  # 2016
# ms2a13 RAD  n_clusters=10
modified_binary_image2 = (pshp_image_clustered2 == 2).astype(int) + (pshp_image_clustered2 == 1).astype(int) + (pshp_image_clustered2 == 8).astype(int) + (pshp_image_clustered2 == 9).astype(int)  # 2002
modified_binary_image2 = (pshp_image_clustered2 == 2).astype(int) + (pshp_image_clustered2 == 6).astype(int) + (pshp_image_clustered2 == 5).astype(int) + (pshp_image_clustered2 == 7).astype(int) + (pshp_image_clustered2 == 1).astype(int)  # 2014
modified_binary_image2 = (pshp_image_clustered2 == 7).astype(int) + (pshp_image_clustered2 == 9).astype(int) + (pshp_image_clustered2 == 2).astype(int) + (pshp_image_clustered2 == 8).astype(int)  # 2020
# ms13a2 RAD  n_clusters=10
modified_binary_image2 = (pshp_image_clustered2 == 3).astype(int) + (pshp_image_clustered2 == 2).astype(int) + (pshp_image_clustered2 == 7).astype(int) # 2002
modified_binary_image2 = (pshp_image_clustered2 == 1).astype(int) + (pshp_image_clustered2 == 3).astype(int) + (pshp_image_clustered2 == 6).astype(int) # 2008
modified_binary_image2 = (pshp_image_clustered2 == 1).astype(int) + (pshp_image_clustered2 == 3).astype(int)  # 2020





plt.figure(figsize=(10, 10))
plt.imshow(modified_binary_image2, cmap='viridis')  # You can choose any colormap (e.g., 'viridis', 'plasma')


modified_binary_image3 = np.where(shapefile_binary == 1, modified_binary_image2, 0)

# plt.figure(figsize=(10, 10))
# plt.imshow(modified_binary_image3, cmap='viridis')  # You can choose any colormap (e.g., 'viridis', 'plasma')

modified_binary_image4 = np.where(water_mask[0], 0, modified_binary_image3)

plt.figure(figsize=(10, 10))
plt.imshow(modified_binary_image4, cmap='viridis')  # You can choose any colormap (e.g., 'viridis', 'plasma')




# Save cluster image
output_folder = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train_wettundra'
output_folder = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train_wettundra2_smi'

file_name = pshp_image_path.split('/')[-1]
# file_name = 'QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2_v2.tif'
# file_name = 'WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2_v2.tif'
# file_name = 'QB02_20050724225630_1010010004654800_PSHP_P002_NT_2000m_area1.tif'

output_image_path = os.path.join(output_folder, file_name)
print(file_name)


# Save using rasterio
with rasterio.open(pshp_image_path) as src:
    # src.update({'count': 1, 'dtype': 'int32'})  # Update metadata for saving
    meta = src.meta.copy()
    meta.update({
        'count': 1,  # Update for single band
        'dtype': 'uint8'  # Change as appropriate (e.g., 'uint8' or 'int32')
    })
    # with rasterio.open(output_image_path, 'w', **src.meta) as dst:
    with rasterio.open(output_image_path, 'w', **meta) as dst:
        # dst.write(modified_binary_image)
        dst.write(modified_binary_image4, 1)








### Soil Moisture Index (SMI)


def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image

def calculate_smi(image):
    # Assuming bands are ordered as: [Red, Green, NIR, ...]
    red = image[2, :, :]  # Adjust index based on your band order
    green = image[1, :, :]  # Adjust index based on your band order
    nir = image[3, :, :]  # Adjust index based on your band order

    # Calculate SMI
    smi = (nir - (red + green)) / (nir + (red + green))

    # Clip values to ensure they're in a reasonable range
    # smi = np.clip(smi, -1, 10)

    return smi

# ms6a2
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'
# ms5a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20030610212157_1010010001F8FC00_PSHP_P001_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20100623215741_103001000594C600_PSHP_P007_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20220710215253_1040010079D00200_PSHP_P009_NT_2000m_area1.tif'
# ms12a9
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020802215611_1010010000EC0A00_PSHP_P001_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20120722232542_1030010019B33400_PSHP_P002_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20200606222801_104001005D57A500_PSHP_P007_NT_2000m_area9.tif'
# ms16a9
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20040804220250_1010010003255300_PSHP_P007_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20120620230434_1030010019D3EF00_PSHP_P003_NT_2000m_area9.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20160612215015_1030010057A6E600_PSHP_P012_NT_2000m_area9.tif'
# ms2a13
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020731213451_1010010000E6BE00_PSHP_P003_NT_2000m_area13.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140810214641_1030010034AC5200_PSHP_P006_NT_2000m_area13.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20200822213156_10300100AA78CA00_PSHP_P004_NT_2000m_area13.tif'
# ms13a2
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020802215615_1010010000EC0A00_PSHP_P002_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20080817223311_101001000872CD00_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20200606222804_104001005D57A500_PSHP_P009_NT_2000m_area2.tif'
# ms16a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20040804220250_1010010003255300_PSHP_P007_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20120620230434_1030010019D3EF00_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20160612215015_1030010057A6E600_PSHP_P012_NT_2000m_area1.tif'
# ms2a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20020731213451_1010010000E6BE00_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140810214641_1030010034AC5200_PSHP_P006_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20200822213156_10300100AA78CA00_PSHP_P004_NT_2000m_area1.tif'
# ms6a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area1.tif'
# ms1a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20050724225626_1010010004654800_PSHP_P001_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20130820230414_1030010025076500_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140707224109_1030010033485600_PSHP_P007_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170813230621_1040010031CAA600_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20200716223511_10300100A97DD900_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20220619223632_10300100D5894700_PSHP_P007_NT_2000m_area1.tif'
# ms10a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20050724225630_1010010004654800_PSHP_P002_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140707224108_1030010033485600_PSHP_P006_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170813230622_1040010031CAA600_PSHP_P004_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20220619223631_10300100D5894700_PSHP_P006_NT_2000m_area1.tif'



pshp_image = load_image(pshp_image_path)

ndvi_image_path = pshp_image_path.replace('x_train', 'ndvi').replace('PSHP', 'NDVI')
ndvi_image = load_image(ndvi_image_path)

smi_image = calculate_smi(pshp_image)  # where pshp_image is your input image
water_mask = ndvi_image < 0.1

shapefile_basepath = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/shapefiles_wettundra/'
shapefile_path = os.path.join(shapefile_basepath, pshp_image_path.split('/')[-1].replace('.tif','.shp'))
shapefile_binary = rasterize_shapefile_to_binary(shapefile_path, pshp_image_path)


# del pshp_image, ndvi_image


plt.figure(figsize=(10, 10))
plt.imshow(smi_image, cmap='Blues')  # You can choose any colormap (e.g., 'viridis', 'plasma')

# ms6a2
smi_image_masked = (smi_image >= 650) & (smi_image <= 1000) # 2009
smi_image_masked = (smi_image >= 740) & (smi_image <= 1000) # 2011
smi_image_masked = (smi_image >= 650) & (smi_image <= 1000) # 2013
smi_image_masked = (smi_image >= 670) & (smi_image <= 1000) # 2016
smi_image_masked = (smi_image >= 750) & (smi_image <= 1000) # 2017
# ms5a1
smi_image_masked = (smi_image >= 670) & (smi_image <= 1000) # 2003
smi_image_masked = (smi_image >= 780) & (smi_image <= 1000) # 2010
smi_image_masked = (smi_image >= 548) & (smi_image <= 1000) # 2022
# ms12a9
smi_image_masked = (smi_image >= 630) & (smi_image <= 1000) # 2002
# smi_image_masked = (smi_image >= 1) & (smi_image <= 1000) # 2012  used 2020 modified_binary_image3 copy
smi_image_masked = (smi_image >= 800) & (smi_image <= 1000) # 2020
# ms16a9
smi_image_masked = (smi_image >= 830) & (smi_image <= 1000) # 2004
smi_image_masked = (smi_image >= 770) & (smi_image <= 1000) # 2012
smi_image_masked = (smi_image >= 7000) & (smi_image <= 1000) # 2016
# ms2a13
# smi_image_masked = (smi_image >= 620) & (smi_image <= 800) # 2002  used 2014 modified_binary_image3 copy
smi_image_masked = (smi_image >= 550) & (smi_image <= 1000) # 2014
smi_image_masked = (smi_image >= 0.03) & (smi_image <= 500) | (smi_image >= 600) # 2014
smi_image_masked = (smi_image >= 950) & (smi_image <= 1500) # 2020
# ms13a2
smi_image_masked = (smi_image >= 550) & (smi_image <= 1000) # 2002
smi_image_masked = (smi_image >= 660) & (smi_image <= 1000) # 2008
smi_image_masked = (smi_image >= 610) & (smi_image <= 1000) # 2020
# ms16a1
smi_image_masked = (smi_image >= 750) & (smi_image <= 1000) # 2004
smi_image_masked = (smi_image >= 730) & (smi_image <= 1000) # 2012
smi_image_masked = (smi_image >= 715) & (smi_image <= 1000) # 2016
# ms2a1
smi_image_masked = (smi_image >= 490) & (smi_image <= 500) # 2004
# smi_image_masked =  (smi_image <= 550) # 2004
# smi_image_masked = (smi_image <= .01) # 2014  used NDVI 0.45 Threshold
smi_image_masked = (smi_image >= 950) # 2020
# ms6a1
smi_image_masked = (smi_image >= 650) & (smi_image <= 1000) # 2009
smi_image_masked = (smi_image >= 740) & (smi_image <= 1000) # 2011
smi_image_masked = (smi_image >= 650) & (smi_image <= 1000) # 2013
smi_image_masked = (smi_image >= 770) & (smi_image <= 1000) # 2016
smi_image_masked = (smi_image >= 750) & (smi_image <= 1000) # 2017
# ms1a1
smi_image_masked = (smi_image >= 500) & (smi_image <= 1000) # 2005
smi_image_masked = (smi_image >= 450) & (smi_image <= 1000) # 2013
# smi_image_masked = (smi_image >= 400) & (smi_image <= 470) # 2014  using NDVI lower than 0.2
smi_image_masked = (smi_image >= 650) & (smi_image <= 1000) # 2017
# smi_image_masked = (smi_image >= 450) & (smi_image <= 1000) # 2020  using NDVI lower than 0.45
# smi_image_masked = (smi_image >= 650) & (smi_image <= 1000) # 2022   using NDVI lower than 0.26
# ms10a1
# smi_image_masked = (smi_image >= 0.1) & (smi_image <= 1000) # 2005  using NDVI
smi_image_masked = (smi_image >= 0.05) & (smi_image <= 1000) # 2014
smi_image_masked = (smi_image >= 550) & (smi_image <= 1000) # 2017

plt.figure(figsize=(10, 10))
plt.imshow(smi_image_masked, cmap='Blues')  # You can choose any colormap (e.g., 'viridis', 'plasma')


modified_binary_image3 = np.where(water_mask[0], 0, smi_image_masked)

plt.figure(figsize=(10, 10))
plt.imshow(modified_binary_image3, cmap='Blues')  # You can choose any colormap (e.g., 'viridis', 'plasma')

# shapefile
modified_binary_image3 = np.where(shapefile_binary == 1, modified_binary_image3, 0)




# If ROCKS outcrop visible
plt.figure(figsize=(10, 10))
plt.imshow(np.clip(np.squeeze(ndvi_image),0,1), cmap='Blues')  # You can choose any colormap (e.g., 'viridis', 'plasma')

ndvi_image_mask = ndvi_image > 0.2
# ndvi_image_mask = ndvi_image > 0.1
ndvi_image_mask = ndvi_image < 0.47
ndvi_image_mask = ndvi_image > 0.3
ndvi_image_mask = ndvi_image > 0.4
ndvi_image_mask = ndvi_image > 0.45
ndvi_image_mask = ndvi_image < 0.3 # ms2a1 2020
ndvi_image_mask = ndvi_image < 0.45 # ms1a1 2020
ndvi_image_mask = ndvi_image < 0.26 # ms1a1 2022
ndvi_image_mask = (ndvi_image >= 0.3) & (ndvi_image <= 0.55) # ms10a1 2005

plt.figure(figsize=(10, 10))
plt.imshow(np.clip(np.squeeze(ndvi_image_mask),0,1), cmap='Blues')  # You can choose any colormap (e.g., 'viridis', 'plasma')

# modified_binary_image4 = np.where(water_mask[0], 0, ndvi_image_mask)
# modified_binary_image4 = np.squeeze(modified_binary_image4)
modified_binary_image4 = np.where(ndvi_image_mask[0] == 1, modified_binary_image3, 0)
# modified_binary_image4 = np.where(ndvi_image_mask[0] == 0, modified_binary_image3, 0)

plt.figure(figsize=(10, 10))
plt.imshow(modified_binary_image4, cmap='Blues')  # You can choose any colormap (e.g., 'viridis', 'plasma')
# plt.imshow(np.squeeze(modified_binary_image4), cmap='Blues')  # You can choose any colormap (e.g., 'viridis', 'plasma')

# del modified_binary_image4


# SAVE SMI binary
output_folder = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train_wettundra2_smi'


file_name = pshp_image_path.split('/')[-1]
# file_name = 'QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2_v2.tif'

output_image_path = os.path.join(output_folder, file_name)
print(file_name)


# Save using rasterio
with rasterio.open(pshp_image_path) as src:
    # src.update({'count': 1, 'dtype': 'int32'})  # Update metadata for saving
    meta = src.meta.copy()
    meta.update({
        'count': 1,  # Update for single band
        'dtype': 'uint8'  # Change as appropriate (e.g., 'uint8' or 'int32')
    })
    # with rasterio.open(output_image_path, 'w', **src.meta) as dst:
    with rasterio.open(output_image_path, 'w', **meta) as dst:
        # dst.write(modified_binary_image4, 1)
        dst.write(modified_binary_image3, 1)    # when no rock outcrop visible

del dst, src











# SWIR smi
## SMVI  soil moisture from WV02 and WV03 M1BS images  (not better)

pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/multisites/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area2_-1948141_4012041/WV03_20160714215120_104001001F9EF500_M1BS_P001_NT_2000m_area2.tif'
m1bs_image = load_image(pshp_image_path)

def calculate_smi(nir, swir):
    smi = (nir - swir) / (nir + swir)
    # smi = swir / nir
    return smi

def convert_to_smi_and_save(image):
    # Assuming bands order: [Blue, Green, Red, NIR, SWIR]
    nir = image[6, :, :]  # NIR band
    swir = image[7, :, :]  # SWIR band (if available)

    # Calculate SMI
    smi = calculate_smi(nir, swir)

    # Clip values to ensure they're in a reasonable range
    # smi = np.clip(smi, -1, 1)
    return smi


# Example usage
smi_image = convert_to_smi_and_save(m1bs_image)

plt.figure(figsize=(10, 10))
plt.imshow(np.clip(smi_image,0, 0.2), cmap='Blues')  # You can choose any colormap (e.g., 'viridis', 'plasma')


def calculate_sndvi(nir, swir):
    return (nir - swir) / (nir + swir)

calculate_sndvi


smi_image_mask = (smi_image > 0.12) & (smi_image < 0.155)

smi_image_mask = (smi_image >= 0.15)

plt.figure(figsize=(10, 10))
plt.imshow(smi_image_mask, cmap='Blues')  # You can choose any colormap (e.g., 'viridis', 'plasma')




