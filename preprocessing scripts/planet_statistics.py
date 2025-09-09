

##### --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
##### --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
##### --- --- --- --- --- --- --- --- --- INFERENCE --- --- --- --- --- --- --- --- --- --- --- ---

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer
from affine import Affine

from skimage.filters import threshold_yen
# from sklearn.metrics import confusion_matrix
from skimage.filters import threshold_otsu
from skimage.filters import threshold_li

from functionsdarko import (
                            block_bootstrap_shrub_cover_continuous, block_bootstrap_shrub_cover_binary
                            )

def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image



# v1 Planet VGG ep17

# v1 Planet VGG ep17
modelname = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/PLANET_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_672px_PSHP_b0_b3_v1_ADAPT_ep17.tf'
model = load_model(modelname
)  # CHANGE MANUALLY in inference_on_single_planet_tif-function  → arr_batch = arr_batch[..., :3]

# v2 Planet ResNet ep40
modelname = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/PLANET_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_SR_672px_b0_b3_v2_ADAPT_ep40.tf'
model = load_model(modelname
)  # CHANGE MANUALLY in inference_on_single_planet_tif-function  → arr_batch = arr_batch[..., :3]


# # v2 Planet ResNet ep66
# model = load_model(
#     '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/PLANET_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_SR_672px_b0_b3_v2_ADAPT_ep66.tf'
# )




### RAW IMAGES


def transform_point(x, y, src_crs, dst_crs="ESRI:102001"):
    """Transform a single point (x,y) from src_crs to dst_crs using pyproj."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x2, y2 = transformer.transform(x, y)
    return x2, y2





def warp_planet_in_memory(in_path, out_crs="ESRI:102001", out_width=667, out_height=667, half_size=1000.0):
    """
    Warp the input Planet image (using all bands) to a fixed 2km bounding box
    (centered on the image) and resample to (out_width, out_height).
    Returns: out_arr with shape (bands, out_height, out_width), transform.
    """
    with rasterio.open(in_path) as src:
        src_crs = src.crs
        left, bottom, right, top = src.bounds
        cx = 0.5 * (left + right)
        cy = 0.5 * (bottom + top)
        # Transform the center to the target CRS:
        from pyproj import Transformer
        transformer = Transformer.from_crs(src_crs, out_crs, always_xy=True)
        cx_alb, cy_alb = transformer.transform(cx, cy)
        # Define the output bounding box (2km square)
        xMin = cx_alb - half_size
        xMax = cx_alb + half_size
        yMin = cy_alb - half_size
        yMax = cy_alb + half_size

        # Define the affine transform for the output
        pixel_size_x = (xMax - xMin) / out_width
        pixel_size_y = (yMin - yMax) / out_height  # negative since Y decreases downwards
        transform = Affine(pixel_size_x, 0, xMin,
                           0, pixel_size_y, yMax)

        band_count = src.count
        src_dtype = src.dtypes[0]
        out_arr = np.zeros((band_count, out_height, out_width), dtype=src_dtype)
        # Reproject each band
        for i in range(band_count):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=out_arr[i],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=out_crs,
                resampling=Resampling.bilinear,
            )
    return out_arr, transform


def inference_planet_image(tif_path, model,
                           warp_width=667, warp_height=667,
                           pad_to=672, half_size=1000.0,
                           scale_factor=5000.0):
    """
    1. Warps the Planet image to a fixed extent (warp_width x warp_height).
    2. Adjusts band count to 4 channels:
         - If the image has >4 bands, keep the first 4.
         - If it has <4 bands, add a dummy channel.
    3. Pads the image to pad_to x pad_to.
    4. Normalizes the image.
    5. Runs inference.
    """
    # Warp and resample image
    out_arr, _ = warp_planet_in_memory(tif_path, out_width=warp_width, out_height=warp_height, half_size=half_size)

    # Compute NDVI and water mask
    red = out_arr[2, ...].astype(np.float32) / scale_factor  # Red band (adjust index if needed)
    nir = out_arr[3, ...].astype(np.float32) / scale_factor  # NIR band (adjust index if needed)
    ndvi = (nir - red) / (nir + red + 1e-8)  # Add epsilon to avoid division by zero
    water_mask = (ndvi < 0.0).astype(int)  # Water where NDVI < 0
    print(water_mask.shape)

    # Adjust band count:
    num_bands, H, W = out_arr.shape
    if num_bands >= 4:
        # Select first 3 bands
        # out_arr = out_arr[:3]   ## RESNET
        out_arr = out_arr[:4]  ## VGG
    elif num_bands < 4:
        # Create dummy channel(s). For example, use zeros.
        extra = np.zeros((4 - num_bands, H, W), dtype=out_arr.dtype)
        out_arr = np.concatenate([out_arr, extra], axis=0)

    # Rearrange to (H, W, channels)
    img = np.moveaxis(out_arr, 0, -1)  # shape becomes (H, W, 4)



    # Pad image to pad_to x pad_to (e.g. 667 -> 672)
    pad_h = pad_to - H
    pad_w = pad_to - W
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    # Normalize (same as training)
    img_norm = img_padded / scale_factor
    print(img_padded.shape)


    # Expand dims to create a batch dimension
    input_tensor = np.expand_dims(img_norm, axis=0)  # shape: (1, pad_to, pad_to, 4)

    input_tensor = input_tensor[..., :3]  # RESNET
    # input_tensor = input_tensor[..., :4]  # VGG19

    # Run inference
    preds = model.predict(input_tensor)
    preds = preds[0, ..., 0]  # Remove batch and channel dimensions

    # Remove the padding to revert to original warped size
    preds = preds[:H, :W]

    # Apply water mask to predictions
    preds[water_mask == 1] = 0.0  # Set water areas to 0.0

    return preds



# def inference_planet_image(tif_path, model, pad_to=672):
#     # 1) read
#     with rasterio.open(tif_path) as src:
#         arr = src.read()  # shape (bands, H, W)
#     arr = np.moveaxis(arr, 0, -1)  # => (H, W, bands), e.g. 4 bands
#
#     # 2) if shape is not multiple-of-32, pad
#     H, W, C = arr.shape
#     padH = (32 - (H % 32)) % 32
#     padW = (32 - (W % 32)) % 32
#     arr_padded = np.pad(arr, ((0,padH),(0,padW),(0,0)), mode='constant')
#
#     # 3) normalize
#     arr_norm = arr_padded / 5000.0  # or your scale factor
#
#     # 4) expand dims => (1, newH, newW, 4)
#     arr_batch = np.expand_dims(arr_norm, axis=0)
#
#     # arr_batch = arr_batch[..., :3]  # RESNET
#     arr_batch = arr_batch[..., :4]  # VGG19
#
#     # 5) run inference
#     preds = model.predict(arr_batch)  # => shape (1, newH, newW, 1)
#
#     # 6) squeeze => (newH, newW)
#     preds = preds[0,...,0]
#
#     # 7) remove padding => back to original H, W
#     preds = preds[:H, :W]
#
#     return preds  # e.g. a float mask



### APPLY LOCAL Mean_p70 by slicing prediction

def apply_local_mean_p70(preds, tile_size=100, percentile_value=70):
    """
    1) Slices the preds array (H×W) into patches of size tile_size×tile_size.
    2) For each patch, compute the local percentile (e.g. 70%).
    3) Average those local percentiles -> "Mean_p70".
    4) Return the final threshold and the binary mask (preds > threshold).

    Note: if preds is 667×667, tile_size might be 100 or 50 or 64, etc.
    """
    H, W = preds.shape
    local_thresholds = []

    # 1) loop over patches
    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            # slice each tile
            patch = preds[i:i+tile_size, j:j+tile_size]
            # flatten, remove NaNs
            patch_vals = patch.flatten()
            patch_vals = patch_vals[~np.isnan(patch_vals)]
            if len(patch_vals) < 10:
                # skip if not enough data or do something else
                continue

            # 2) local percentile
            local_pX = np.percentile(patch_vals, percentile_value)
            local_thresholds.append(local_pX)

    # 3) average local thresholds => "Mean_p70"
    if len(local_thresholds) == 0:
        # fallback if no patches
        global_thr = np.percentile(preds[~np.isnan(preds)], percentile_value)
    else:
        global_thr = np.mean(local_thresholds)

    # 4) apply threshold to entire preds
    binary_mask = (preds > global_thr).astype(int)
    return global_thr, binary_mask


# ms1a1
tif_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms1a1_370417_2013-08-20_RE5_3A_Analytic_SR_clip_joined4.tif'

tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms1a1_370417_2013-08-20_RE5_3A_Analytic_SR_clip_joined4.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms1a1_20170703_214611_1007_psscene_analytic_sr_udm2_composite.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms1a1_20200720_203034_83_1067_3B_AnalyticMS_SR_clip.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms1a1_20220803_221626_79_2254_3B_AnalyticMS_SR_clip.tif'

#ms6a2
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms6a2_570820_2011-07-09_RE2_3A_Analytic_SR_clip.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms6a2_670810_2013-07-21_RE1_3A_Analytic_SR_clip.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms6a2_670810_2016-08-05_RE3_3A_Analytic_SR_clip.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms6a2_20170807_205445_1036_3B_AnalyticMS_SR_clip.tif'


#ms6a5
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2010_670810_2010-06-25_RE4_3A_Analytic_SR_clip_1950m.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2011_570820_2011-07-09_RE2_3A_Analytic_SR_clip_1950m.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2013_570820_2013-08-26_RE4_3A_Analytic_SR_clip_1950m.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2016_670810_2016-07-25_RE2_3A_Analytic_SR_clip_1950m.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2017_20170728_205413_1025_3B_AnalyticMS_SR_clip_1950m.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2024_20240719_205848_70_24a8_3B_AnalyticMS_SR_clip_1950m.tif'

# Now you can do:
# preds, ndvi = inference_planet_tif(tif_path, model)
preds = inference_planet_image(tif_path, model)
# preds shape => (667,667)

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(preds))
plt.show(block=True)


# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(ndvi))
# plt.show(block=True)


# Calculate thresholds for each image
thresholds_accumulated = []
method = 'MEAN_p80'

# pred_flat = preds.flatten()
# pred_flat = pred_flat[~np.isnan(pred_flat)]  # Remove NaN values
# thresholds_accumulated.append(threshold_yen(pred_flat))
# threshold_mean = np.mean(thresholds_accumulated)
# binary_preds = preds > threshold_mean


preds = inference_planet_image(tif_path, model)

_, binary_preds = apply_local_mean_p70(preds, tile_size=100, percentile_value=70)
_, binary_preds2 = apply_local_mean_p70(preds, tile_size=100, percentile_value=80)

# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(binary_preds2))
# plt.show(block=True)


# blocksize QB/WV = 200 --- which is 100m --- Planet 100m is ~30px
mean_cov_cont, std_cov_cont, ci_low_cont, ci_up_cont = block_bootstrap_shrub_cover_continuous(preds, block_size=(30, 30), n_boot=10000)
mean_cov_b, std_cov_b, ci_low_b, ci_up_b = block_bootstrap_shrub_cover_binary(binary_preds2, block_size=(30, 30), n_boot=10000)


all_statistics = []

metrics_by_year = {
                    'cover_frac_binary': mean_cov_b,  # from binary predictions
                    'cover_frac_cont': mean_cov_cont,  # from continuous predictions
                    'std_cov_cont': std_cov_cont,
                    'ci_low_cont': ci_low_cont,
                    'ci_up_cont': ci_up_cont,
                    'std_cov_b': std_cov_b,
                    'ci_low_b': ci_low_b,
                    'ci_up_b': ci_up_b,
                    # 'cover_m2': shrub_cover_m2,
                    # 'total_area_shrubmap_m2': total_area_shrubmap_m2,
                }

df = pd.DataFrame([metrics_by_year])

print(df)


df.to_clipboard(index=False)  # Copies DataFrame without the index






### Iterate over multiple images in one directory:
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Directory containing your .tif images
# ms1a1
tif_directory = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/raw/ms1a1/good'
# tif_directory = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/raw downloads/ms1a1/good'

# ms4a1
tif_directory = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/raw/ms4a1'

# ms6a2
tif_directory = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/raw/ms6a2/good'

# ms6a5
tif_directory = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet'



all_statistics = []
def extract_metadata_from_filename(filename):
    """Extract Year, Month, Day, and scene_id from the filename."""
    parts = filename.split('_')
    year = int(parts[1])  # Assuming second part is the year
    date_str = parts[-1].split('.')[0]  # Last part before .tif is the date
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        month, day = date.month, date.day
    except ValueError:
        month, day = None, None  # Handle case where date is not formatted correctly
    scene_id = filename
    return year, month, day, scene_id


# Iterate over all .tif files in the directory
for tif_file in os.listdir(tif_directory):
    if tif_file.endswith(".tif"):
        print(f"Processing {tif_file}...")

        # Extract metadata
        year, month, day, scene_id = extract_metadata_from_filename(tif_file)

        # Perform inference
        preds = inference_planet_image(os.path.join(tif_directory, tif_file), model)
        _, binary_preds = apply_local_mean_p70(preds, tile_size=100, percentile_value=80)

        # Calculate shrub cover metrics
        mean_cov_cont, std_cov_cont, ci_low_cont, ci_up_cont = block_bootstrap_shrub_cover_continuous(preds,
                                                                                                      block_size=(
                                                                                                      30, 30),
                                                                                                      n_boot=1000)
        mean_cov_b, std_cov_b, ci_low_b, ci_up_b = block_bootstrap_shrub_cover_binary(binary_preds, block_size=(30, 30),
                                                                                      n_boot=1000)

        # Store results in a dictionary
        metrics_by_year = {
            'File': tif_file,
            'Area': "Unknown",  # Replace with actual area if available
            'Year': year,
            'Month': month,
            'Day': day,
            'Method': "MEAN_p80",
            'cover_frac_binary': mean_cov_b,
            'cover_frac_cont': mean_cov_cont,
            'std_cov_cont': std_cov_cont,
            'ci_low_cont': ci_low_cont,
            'ci_up_cont': ci_up_cont,
            'std_cov_b': std_cov_b,
            'ci_low_b': ci_low_b,
            'ci_up_b': ci_up_b,
            'cover_m2': None,  # Placeholder
            'total_area_m2': None,  # Placeholder
            'Sensor': None,  # Placeholder
            'scene_id': scene_id,
            'model': "Resnet"
        }

        # Add the dictionary to the list
        all_statistics.append(metrics_by_year)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(all_statistics)

df.to_clipboard(index=False)  # Copies DataFrame without the index



#
# # Save the DataFrame to a CSV file
# output_file = "shrub_cover_statistics.csv"
# df.to_csv(output_file, index=False)
# print(f"Results saved to {output_file}")










# PLOT INDIVIDUAL DATA with scatterplot with errorbars

### PLANET BLOCK BOOTSTRAP SIZE 30 x 30

import matplotlib.pyplot as plt
import numpy as np

# ms1a1 data
data = {
    "Year": [2013, 2017, 2020, 2022],
    "Month": [8, 7, 7, 8],
    "Day": [20, 3, 20, 3],
    "Method": ["MEAN_p80", "MEAN_p80", "MEAN_p80", "MEAN_p80"],
    "cover_frac_binary": [0.232573966, 0.210810719, 0.262152411, 0.222293773],
    "cover_frac_cont": [0.28945574, 0.17880134, 0.40836892, 0.2674009],
    "ci_low_cont": [0.270251925, 0.165234346, 0.38930474, 0.247101735],
    "ci_up_cont": [0.308659558, 0.19236834, 0.427433091, 0.287700046],
    "ci_low_b": [0.203320953, 0.183665832, 0.23429361, 0.193685127],
    "ci_up_b": [0.261826979, 0.237955605, 0.290011211, 0.250902419]
}

# Convert to lists
years = data["Year"]
cover_binary = data["cover_frac_binary"]
ci_low_b = data["ci_low_b"]
ci_up_b = data["ci_up_b"]
cover_cont = data["cover_frac_cont"]
ci_low_cont = data["ci_low_cont"]
ci_up_cont = data["ci_up_cont"]

# Calculate errors (distance from mean to CI bounds)
error_binary = [
    np.array(cover_binary) - np.array(ci_low_b),
    np.array(ci_up_b) - np.array(cover_binary)]
error_cont = [
    np.array(cover_cont) - np.array(ci_low_cont),
    np.array(ci_up_cont) - np.array(cover_cont)]

plt.figure(figsize=(10, 6))
plt.errorbar(
    years, cover_binary, yerr=error_binary,
    fmt='o-', color='#1f77b4', label='Cover Fraction (Binary)',
    capsize=5, markersize=8)
plt.errorbar(
    years, cover_cont, yerr=error_cont,
    fmt='s--', color='#ff7f0e', label='Cover Fraction (Continuous)',
    capsize=5, markersize=8)
plt.xticks(years, fontsize=12)
plt.yticks(np.linspace(0, 0.4, 9), fontsize=12)
plt.ylim(0, 0.4)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Cover Fraction', fontsize=14)
plt.title('Shrub Cover Fraction Over Time (with 95% CI)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show(block=True)



## ms6a2
data = {
    "Year": [2011, 2013, 2016, 2017],
    "Month": [7, 7, 8, 8],
    "Day": [9, 21, 5, 7],
    "Method": ["MEAN_p80", "MEAN_p80", "MEAN_p80", "MEAN_p80"],
    "cover_frac_binary": [0.266443536, 0.242213884, 0.229705909, 0.250117602],
    "cover_frac_cont": [0.3157226, 0.34058985, 0.2749871, 0.2383615],
    "ci_low_cont": [0.299594232, 0.325427368, 0.261536174, 0.223898712],
    "ci_up_cont": [0.331850997, 0.355752334, 0.288438029, 0.252824274],
    "ci_low_b": [0.2389355, 0.217698536, 0.206674149, 0.222012491],
    "ci_up_b": [0.293951572, 0.266729232, 0.252737669, 0.278222713]
}

# Convert to lists
years = data["Year"]
cover_binary = data["cover_frac_binary"]
ci_low_b = data["ci_low_b"]
ci_up_b = data["ci_up_b"]
cover_cont = data["cover_frac_cont"]
ci_low_cont = data["ci_low_cont"]
ci_up_cont = data["ci_up_cont"]


# Calculate errors (distance from mean to CI bounds)
error_binary = [
    np.array(cover_binary) - np.array(ci_low_b),
    np.array(ci_up_b) - np.array(cover_binary)]
error_cont = [
    np.array(cover_cont) - np.array(ci_low_cont),
    np.array(ci_up_cont) - np.array(cover_cont)]

plt.figure(figsize=(10, 6))
plt.errorbar(
    years, cover_binary, yerr=error_binary,
    fmt='o-', color='#1f77b4', label='Cover Fraction (Binary)',
    capsize=5, markersize=8)
plt.errorbar(
    years, cover_cont, yerr=error_cont,
    fmt='s--', color='#ff7f0e', label='Cover Fraction (Continuous)',
    capsize=5, markersize=8)
plt.xticks(years, fontsize=12)
plt.yticks(np.linspace(0, 0.4, 9), fontsize=12)
plt.ylim(0, 0.4)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Cover Fraction', fontsize=14)
plt.title('Shrub Cover Fraction Over Time (with 95% CI)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show(block=True)






## ms6a5
data = {
    "Year": [2010, 2011, 2013, 2016, 2017, 2024],
    "Month": [6, 7, 8, 7, 7, 7],
    "Day": [25, 9, 26, 25, 28, 19],
    "Method": ["MEAN_p80", "MEAN_p80", "MEAN_p80", "MEAN_p80", "MEAN_p80", "MEAN_p80"],
    "cover_frac_binary": [0.229041434, 0.226833511, 0.224149539, 0.237237165, 0.204995447, 0.234840664],
    "cover_frac_cont": [0.27219877, 0.2686711, 0.20672591, 0.26237714, 0.19497123, 0.23558721],
    "ci_low_cont": [0.255355516, 0.252297429, 0.191150309, 0.244570933, 0.180358182, 0.217764049],
    "ci_up_cont": [0.289042017, 0.285044762, 0.222301511, 0.280183353, 0.209584285, 0.25341037],
    "ci_low_b": [0.202662308, 0.19988976, 0.19505043, 0.209853744, 0.178049898, 0.206527767],
    "ci_up_b": [0.25542056, 0.253777262, 0.253248649, 0.264620586, 0.231940995, 0.26315356]
}

# Convert to lists
years = data["Year"]
cover_binary = data["cover_frac_binary"]
ci_low_b = data["ci_low_b"]
ci_up_b = data["ci_up_b"]
cover_cont = data["cover_frac_cont"]
ci_low_cont = data["ci_low_cont"]
ci_up_cont = data["ci_up_cont"]


# Calculate errors (distance from mean to CI bounds)
error_binary = [
    np.array(cover_binary) - np.array(ci_low_b),
    np.array(ci_up_b) - np.array(cover_binary)]
error_cont = [
    np.array(cover_cont) - np.array(ci_low_cont),
    np.array(ci_up_cont) - np.array(cover_cont)]

plt.figure(figsize=(10, 6))
plt.ion()
# Scatter plot for cover_frac_binary
plt.scatter(years, cover_binary, color='#1f77b4', label='Cover Fraction (Binary)', s=80)
plt.errorbar(years, cover_binary, yerr=error_binary, fmt='none', ecolor='#1f77b4', capsize=5)
# Scatter plot for cover_frac_cont
plt.scatter(years, cover_cont, color='#ff7f0e', label='Cover Fraction (Continuous)', s=80, marker='s')
plt.errorbar(years, cover_cont, yerr=error_cont, fmt='none', ecolor='#ff7f0e', capsize=5)
# Customize plot
plt.xticks(years, fontsize=12)
plt.yticks(np.linspace(0, 0.4, 9), fontsize=12)
plt.ylim(0, 0.4)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Cover Fraction', fontsize=14)
plt.title('Shrub Cover Fraction Over Time (Scatter Plot with 95% CI)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show(block=True)





### PLOT BOTH PLANET AND QB/WV data



import numpy as np
import matplotlib.pyplot as plt

# ms6a5 Planet Data Resnet p80
planet_data = {
    "Year": [2010, 2011, 2013, 2016, 2017, 2024],
    "cover_frac_binary": [0.229041434, 0.226833511, 0.224149539, 0.237237165, 0.204995447, 0.234840664],
    "cover_frac_cont": [0.27219877, 0.2686711, 0.20672591, 0.26237714, 0.19497123, 0.23558721],
    "ci_low_b": [0.202662308, 0.19988976, 0.19505043, 0.209853744, 0.178049898, 0.206527767],
    "ci_up_b": [0.25542056, 0.253777262, 0.253248649, 0.264620586, 0.231940995, 0.26315356],
    "ci_low_cont": [0.255355516, 0.252297429, 0.191150309, 0.244570933, 0.180358182, 0.217764049],
    "ci_up_cont": [0.289042017, 0.285044762, 0.222301511, 0.280183353, 0.209584285, 0.25341037]
}

# ms6a5 QB/WV Data Resnet p80
wv_data = {
    "Year": [2009, 2013, 2011, 2016],
    "cover_frac_binary": [0.18440339, 0.217861503, 0.171963379, 0.194364354],
    "cover_frac_cont": [0.12033467, 0.271407872, 0.075794302, 0.263193309],
    "ci_low_b": [0.16618059, 0.198617273, 0.155949313, 0.178632828],
    "ci_up_b": [0.202626189, 0.237105734, 0.187977445, 0.21009588],
    "ci_low_cont": [0.110528341, 0.255519828, 0.069204137, 0.253075519],
    "ci_up_cont": [0.130140999, 0.287295917, 0.082384467, 0.2733111]
}

## calibration
# Scale Planet data
weighted_cal_slope = 0.32
planet_data["cover_frac_binary"] = [v * weighted_cal_slope for v in planet_data["cover_frac_binary"]]
planet_data["cover_frac_cont"] = [v * weighted_cal_slope for v in planet_data["cover_frac_cont"]]
planet_data["ci_low_b"] = [v * weighted_cal_slope for v in planet_data["ci_low_b"]]
planet_data["ci_up_b"] = [v * weighted_cal_slope for v in planet_data["ci_up_b"]]
planet_data["ci_low_cont"] = [v * weighted_cal_slope for v in planet_data["ci_low_cont"]]
planet_data["ci_up_cont"] = [v * weighted_cal_slope for v in planet_data["ci_up_cont"]]

# Scale WV/QB data
weighted_cal_slope = 0.94
wv_data["cover_frac_binary"] = [v * weighted_cal_slope for v in wv_data["cover_frac_binary"]]
wv_data["cover_frac_cont"] = [v * weighted_cal_slope for v in wv_data["cover_frac_cont"]]
wv_data["ci_low_b"] = [v * weighted_cal_slope for v in wv_data["ci_low_b"]]
wv_data["ci_up_b"] = [v * weighted_cal_slope for v in wv_data["ci_up_b"]]
wv_data["ci_low_cont"] = [v * weighted_cal_slope for v in wv_data["ci_low_cont"]]
wv_data["ci_up_cont"] = [v * weighted_cal_slope for v in wv_data["ci_up_cont"]]

# Function to calculate error bars
def calculate_error(data, low_key, high_key, main_key):
    lower_error = np.array(data[main_key]) - np.array(data[low_key])
    upper_error = np.array(data[high_key]) - np.array(data[main_key])
    return [lower_error, upper_error]

# Calculate error bars for both datasets
planet_error_binary = calculate_error(planet_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
planet_error_cont = calculate_error(planet_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")
wv_error_binary = calculate_error(wv_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
wv_error_cont = calculate_error(wv_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")

# Plot
plt.figure(figsize=(14, 10))
# Planet cover_frac_binary
plt.scatter(planet_data["Year"], planet_data["cover_frac_binary"], color='blue', label='Planet Binary', s=80)
plt.errorbar(planet_data["Year"], planet_data["cover_frac_binary"], yerr=planet_error_binary, fmt='none', ecolor='blue', capsize=5)
# Planet cover_frac_cont
plt.scatter(planet_data["Year"], planet_data["cover_frac_cont"], color='cyan', label='Planet Continuous', s=80, marker='s')
plt.errorbar(planet_data["Year"], planet_data["cover_frac_cont"], yerr=planet_error_cont, fmt='none', ecolor='cyan', capsize=5)
# WV/QB cover_frac_binary
plt.scatter(wv_data["Year"], wv_data["cover_frac_binary"], color='red', label='WV/QB Binary', s=80)
plt.errorbar(wv_data["Year"], wv_data["cover_frac_binary"], yerr=wv_error_binary, fmt='none', ecolor='red', capsize=5)
# WV/QB cover_frac_cont
plt.scatter(wv_data["Year"], wv_data["cover_frac_cont"], color='orange', label='WV/QB Continuous', s=80, marker='s')
plt.errorbar(wv_data["Year"], wv_data["cover_frac_cont"], yerr=wv_error_cont, fmt='none', ecolor='orange', capsize=5)
# Customize plot
plt.xticks(np.arange(2009, 2025, 2), fontsize=20, fontweight='bold')
plt.yticks(np.linspace(0, 0.4, 9), fontsize=20, fontweight='bold')
plt.ylim(0, 0.5)
plt.xlabel('Year', fontsize=20, fontweight='bold')
plt.ylabel('Cover Fraction', fontsize=20, fontweight='bold')
plt.title('Calibrated Cover Fraction Over Time (Planet vs. \n WV/QB Data) based on ResNet50 predictions and dynamic \n p80 thresholding for site6 area5', fontsize=26, fontweight='bold')
plt.legend(fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show(block=True)









import numpy as np
import matplotlib.pyplot as plt

# ms6a5 Planet VGG p80 Data for
planet_data = {
    "Year": [2016, 2010, 2013, 2024, 2011, 2017],
    "cover_frac_binary": [0.293512606, 0.276701522, 0.336419812, 0.285848673, 0.301730879, 0.277878051],
    "cover_frac_cont": [0.3160347, 0.30911997, 0.31771117, 0.3289696, 0.31092212, 0.21454535],
    "ci_low_b": [0.272075648, 0.255150692, 0.31787134, 0.263155064, 0.283718346, 0.257343024],
    "ci_up_b": [0.314949563, 0.298252352, 0.354968283, 0.308542283, 0.319743413, 0.298413078],
    "ci_low_cont": [0.307173163, 0.299148655, 0.311532333, 0.320810652, 0.303639404, 0.208315646],
    "ci_up_cont": [0.324896246, 0.319091284, 0.323890016, 0.337128543, 0.318204828, 0.220775062]
}

# ms6a5 QB/WV VGG p80 Data for
wv_data = {
    "Year": [2009, 2013, 2011, 2016],
    "cover_frac_binary": [0.191660076, 0.21144712, 0.182209402, 0.211346507],
    "cover_frac_cont": [0.1512862, 0.244304717, 0.122852847, 0.361324012],
    "ci_low_b": [0.171537586, 0.19130977, 0.162698794, 0.191601159],
    "ci_up_b": [0.211782567, 0.231584469, 0.201720011, 0.231091855],
    "ci_low_cont": [0.139891801, 0.228866131, 0.113185953, 0.347079641],
    "ci_up_cont": [0.162680598, 0.259743302, 0.132519741, 0.375568384]
}


## calibration
# Scale Planet data
weighted_cal_slope = 0.62
planet_data["cover_frac_binary"] = [v * weighted_cal_slope for v in planet_data["cover_frac_binary"]]
planet_data["cover_frac_cont"] = [v * weighted_cal_slope for v in planet_data["cover_frac_cont"]]
planet_data["ci_low_b"] = [v * weighted_cal_slope for v in planet_data["ci_low_b"]]
planet_data["ci_up_b"] = [v * weighted_cal_slope for v in planet_data["ci_up_b"]]
planet_data["ci_low_cont"] = [v * weighted_cal_slope for v in planet_data["ci_low_cont"]]
planet_data["ci_up_cont"] = [v * weighted_cal_slope for v in planet_data["ci_up_cont"]]

# Scale WV/QB data
weighted_cal_slope = 0.83
wv_data["cover_frac_binary"] = [v * weighted_cal_slope for v in wv_data["cover_frac_binary"]]
wv_data["cover_frac_cont"] = [v * weighted_cal_slope for v in wv_data["cover_frac_cont"]]
wv_data["ci_low_b"] = [v * weighted_cal_slope for v in wv_data["ci_low_b"]]
wv_data["ci_up_b"] = [v * weighted_cal_slope for v in wv_data["ci_up_b"]]
wv_data["ci_low_cont"] = [v * weighted_cal_slope for v in wv_data["ci_low_cont"]]
wv_data["ci_up_cont"] = [v * weighted_cal_slope for v in wv_data["ci_up_cont"]]


# Function to calculate error bars
def calculate_error(data, low_key, high_key, main_key):
    lower_error = np.array(data[main_key]) - np.array(data[low_key])
    upper_error = np.array(data[high_key]) - np.array(data[main_key])
    return [lower_error, upper_error]

# Calculate error bars for both datasets
planet_vgg_error_binary = calculate_error(planet_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
planet_vgg_error_cont = calculate_error(planet_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")
wv_vgg_error_binary = calculate_error(wv_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
wv_vgg_error_cont = calculate_error(wv_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")

# Plot
plt.figure(figsize=(14, 10))
# Planet VGG p80 cover_frac_binary
plt.scatter(planet_data["Year"], planet_data["cover_frac_binary"], color='blue', label='Planet VGG Binary', s=80)
plt.errorbar(planet_data["Year"], planet_data["cover_frac_binary"], yerr=planet_vgg_error_binary, fmt='none', ecolor='blue', capsize=5)

# Planet VGG p80 cover_frac_cont
plt.scatter(planet_data["Year"], planet_data["cover_frac_cont"], color='cyan', label='Planet VGG Continuous', s=80, marker='s')
plt.errorbar(planet_data["Year"], planet_data["cover_frac_cont"], yerr=planet_vgg_error_cont, fmt='none', ecolor='cyan', capsize=5)

# WV/QB VGG p80 cover_frac_binary
plt.scatter(wv_data["Year"], wv_data["cover_frac_binary"], color='red', label='WV/QB VGG Binary', s=80)
plt.errorbar(wv_data["Year"], wv_data["cover_frac_binary"], yerr=wv_vgg_error_binary, fmt='none', ecolor='red', capsize=5)

# WV/QB VGG p80 cover_frac_cont
plt.scatter(wv_data["Year"], wv_data["cover_frac_cont"], color='orange', label='WV/QB VGG Continuous', s=80, marker='s')
plt.errorbar(wv_data["Year"], wv_data["cover_frac_cont"], yerr=wv_vgg_error_cont, fmt='none', ecolor='orange', capsize=5)

# Customize plot
plt.xticks(np.arange(2009, 2025, 2), fontsize=20, fontweight='bold')
plt.yticks(np.linspace(0, 0.4, 9), fontsize=20, fontweight='bold')
plt.ylim(0, 0.5)
plt.xlabel('Year', fontsize=20, fontweight='bold')
plt.ylabel('Cover Fraction', fontsize=20, fontweight='bold')
plt.title('Calibrated Cover Fraction Over Time (Planet vs.\n WV/QB Data) based on VGG19 predictions and dynamic\n p80 thresholding for site6 area5', fontsize=26, fontweight='bold')
plt.legend(fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show(block=True)







import numpy as np
import matplotlib.pyplot as plt

# Planet  Data p80  ms1a1 Resnet
planet_resnet_data = {
    "Year": [2011, 2013, 2016, 2017, 2018, 2019, 2020, 2020, 2022, 2023],
    "cover_frac_binary": [0.259858448, 0.269033972, 0.267021951, 0.260733953, 0.260593228, 0.233746811, 0.318603427, 0.281244984, 0.258008058, 0.332101171],
    "cover_frac_cont": [0.2470459, 0.28837106, 0.22713505, 0.18377042, 0.19468798, 0.16649501, 0.40879232, 0.1771378, 0.26740575, 0.41554162],
    "ci_low_b": [0.229790761, 0.238575443, 0.237313574, 0.231036996, 0.231888074, 0.205317099, 0.288969025, 0.249964166, 0.228430297, 0.301103575],
    "ci_up_b": [0.289926135, 0.2994925, 0.296730328, 0.290430911, 0.289298382, 0.262176523, 0.34823783, 0.312525802, 0.287585819, 0.363098766],
    "ci_low_cont": [0.229231312, 0.269047344, 0.210132868, 0.168754169, 0.180048702, 0.152447661, 0.389231843, 0.163520917, 0.247592954, 0.393264348],
    "ci_up_cont": [0.264860497, 0.307694768, 0.244137227, 0.198786667, 0.209327253, 0.180542359, 0.428352791, 0.190754697, 0.287218543, 0.43781889]
}

# QB/WV Resnet Data p80
wv_resnet_data = {
    "Year": [2005, 2013, 2020, 2017],
    "cover_frac_binary": [0.235004708, 0.228953376, 0.226377979, 0.240319967],
    "cover_frac_cont": [0.284336209, 0.30708304, 0.288468689, 0.3054488],
    "ci_low_b": [0.208352077, 0.202690562, 0.201047317, 0.214906128],
    "ci_up_b": [0.261657339, 0.255216191, 0.25170864, 0.265733806],
    "ci_low_cont": [0.262447874, 0.285459499, 0.267371455, 0.284346664],
    "ci_up_cont": [0.306224544, 0.328706582, 0.309565922, 0.326550937]
}


## calibration
# Scale Planet data
weighted_cal_slope = 0.32
planet_resnet_data["cover_frac_binary"] = [v * weighted_cal_slope for v in planet_resnet_data["cover_frac_binary"]]
planet_resnet_data["cover_frac_cont"] = [v * weighted_cal_slope for v in planet_resnet_data["cover_frac_cont"]]
planet_resnet_data["ci_low_b"] = [v * weighted_cal_slope for v in planet_resnet_data["ci_low_b"]]
planet_resnet_data["ci_up_b"] = [v * weighted_cal_slope for v in planet_resnet_data["ci_up_b"]]
planet_resnet_data["ci_low_cont"] = [v * weighted_cal_slope for v in planet_resnet_data["ci_low_cont"]]
planet_resnet_data["ci_up_cont"] = [v * weighted_cal_slope for v in planet_resnet_data["ci_up_cont"]]

# Scale WV/QB data
weighted_cal_slope = 0.94
wv_resnet_data["cover_frac_binary"] = [v * weighted_cal_slope for v in wv_resnet_data["cover_frac_binary"]]
wv_resnet_data["cover_frac_cont"] = [v * weighted_cal_slope for v in wv_resnet_data["cover_frac_cont"]]
wv_resnet_data["ci_low_b"] = [v * weighted_cal_slope for v in wv_resnet_data["ci_low_b"]]
wv_resnet_data["ci_up_b"] = [v * weighted_cal_slope for v in wv_resnet_data["ci_up_b"]]
wv_resnet_data["ci_low_cont"] = [v * weighted_cal_slope for v in wv_resnet_data["ci_low_cont"]]
wv_resnet_data["ci_up_cont"] = [v * weighted_cal_slope for v in wv_resnet_data["ci_up_cont"]]

# Function to calculate error bars
def calculate_error(data, low_key, high_key, main_key):
    lower_error = np.array(data[main_key]) - np.array(data[low_key])
    upper_error = np.array(data[high_key]) - np.array(data[main_key])
    return [lower_error, upper_error]

# Calculate error bars for both datasets
planet_resnet_error_binary = calculate_error(planet_resnet_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
planet_resnet_error_cont = calculate_error(planet_resnet_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")
wv_resnet_error_binary = calculate_error(wv_resnet_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
wv_resnet_error_cont = calculate_error(wv_resnet_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")

plt.figure(figsize=(14, 8))
# Planet Resnet cover_frac_binary
plt.scatter(planet_resnet_data["Year"], planet_resnet_data["cover_frac_binary"], color='blue', label='Planet Resnet Binary', s=80)
plt.errorbar(planet_resnet_data["Year"], planet_resnet_data["cover_frac_binary"], yerr=planet_resnet_error_binary, fmt='none', ecolor='blue', capsize=5)
# Planet Resnet cover_frac_cont
plt.scatter(planet_resnet_data["Year"], planet_resnet_data["cover_frac_cont"], color='cyan', label='Planet Resnet Continuous', s=80, marker='s')
plt.errorbar(planet_resnet_data["Year"], planet_resnet_data["cover_frac_cont"], yerr=planet_resnet_error_cont, fmt='none', ecolor='cyan', capsize=5)
# WV/QB Resnet cover_frac_binary
plt.scatter(wv_resnet_data["Year"], wv_resnet_data["cover_frac_binary"], color='red', label='WV/QB Resnet Binary', s=80)
plt.errorbar(wv_resnet_data["Year"], wv_resnet_data["cover_frac_binary"], yerr=wv_resnet_error_binary, fmt='none', ecolor='red', capsize=5)
# WV/QB Resnet cover_frac_cont
plt.scatter(wv_resnet_data["Year"], wv_resnet_data["cover_frac_cont"], color='orange', label='WV/QB Resnet Continuous', s=80, marker='s')
plt.errorbar(wv_resnet_data["Year"], wv_resnet_data["cover_frac_cont"], yerr=wv_resnet_error_cont, fmt='none', ecolor='orange', capsize=5)
# Customize plot
plt.xticks(np.arange(2005, 2025, 2), fontsize=20, fontweight='bold')
plt.yticks(np.linspace(0, 0.4, 9), fontsize=20, fontweight='bold')
plt.ylim(0, 0.5)
plt.xlabel('Year', fontsize=20, fontweight='bold')
plt.ylabel('Cover Fraction', fontsize=20, fontweight='bold')
plt.title('Calibrated Cover Fraction Over Time (Planet vs. \nWV/QB Data) based on ResNet50 predictions and \n dynamic p80 thresholding for site1 area1', fontsize=26, fontweight='bold')
plt.legend(fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2, loc="upper left",)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show(block=True)







import numpy as np
import matplotlib.pyplot as plt

# ms1a1 VGG Planet  p80 Data
planet_vgg_data = {
    "Year": [2011, 2016, 2017, 2019, 2023, 2022, 2018, 2013, 2020],
    "cover_frac_binary": [0.255896926, 0.230670918, 0.230489573, 0.215800879, 0.352209447, 0.251392553, 0.220235331, 0.272014082, 0.213627831],
    "cover_frac_cont": [0.41245943, 0.30448365, 0.13846314, 0.084084645, 0.65004534, 0.3969747, 0.20219491, 0.47845283, 0.14386797],
    "ci_low_b": [0.2294816, 0.20539511, 0.204255691, 0.190695026, 0.320239355, 0.221342738, 0.193242416, 0.245258207, 0.189025977],
    "ci_up_b": [0.282312252, 0.255946727, 0.256723455, 0.240906733, 0.384179538, 0.281442368, 0.247228246, 0.298769957, 0.238229684],
    "ci_low_cont": [0.39847079, 0.292373854, 0.131366301, 0.080241772, 0.634788766, 0.378935303, 0.192985286, 0.465946079, 0.137415718],
    "ci_up_cont": [0.426448076, 0.31659345, 0.145559978, 0.087927518, 0.665301904, 0.415014122, 0.211404543, 0.490959584, 0.150320221]
}

# ms1a1 QB/WV VGG p80 Data
wv_vgg_data = {
    "Year": [2005, 2013, 2020, 2017],
    "cover_frac_binary": [0.196508452, 0.209639013, 0.206775188, 0.215322658],
    "cover_frac_cont": [0.242334321, 0.288615167, 0.270637453, 0.283032328],
    "ci_low_b": [0.171381139, 0.182683075, 0.180300136, 0.188752616],
    "ci_up_b": [0.221635765, 0.236594951, 0.233250241, 0.241892701],
    "ci_low_cont": [0.222665889, 0.266879156, 0.249289384, 0.26148197],
    "ci_up_cont": [0.262002753, 0.310351179, 0.291985521, 0.304582686]
}



## calibration
# Scale Planet data
weighted_cal_slope = 0.62
planet_vgg_data["cover_frac_binary"] = [v * weighted_cal_slope for v in planet_vgg_data["cover_frac_binary"]]
planet_vgg_data["cover_frac_cont"] = [v * weighted_cal_slope for v in planet_vgg_data["cover_frac_cont"]]
planet_vgg_data["ci_low_b"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_low_b"]]
planet_vgg_data["ci_up_b"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_up_b"]]
planet_vgg_data["ci_low_cont"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_low_cont"]]
planet_vgg_data["ci_up_cont"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_up_cont"]]

# Scale WV/QB data
weighted_cal_slope = 0.83
wv_vgg_data["cover_frac_binary"] = [v * weighted_cal_slope for v in wv_vgg_data["cover_frac_binary"]]
wv_vgg_data["cover_frac_cont"] = [v * weighted_cal_slope for v in wv_vgg_data["cover_frac_cont"]]
wv_vgg_data["ci_low_b"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_low_b"]]
wv_vgg_data["ci_up_b"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_up_b"]]
wv_vgg_data["ci_low_cont"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_low_cont"]]
wv_vgg_data["ci_up_cont"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_up_cont"]]


# Function to calculate error bars
def calculate_error(data, low_key, high_key, main_key):
    lower_error = np.array(data[main_key]) - np.array(data[low_key])
    upper_error = np.array(data[high_key]) - np.array(data[main_key])
    return [lower_error, upper_error]

# Calculate error bars for both datasets
planet_vgg_error_binary = calculate_error(planet_vgg_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
planet_vgg_error_cont = calculate_error(planet_vgg_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")
wv_vgg_error_binary = calculate_error(wv_vgg_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
wv_vgg_error_cont = calculate_error(wv_vgg_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")

# Plot
plt.figure(figsize=(14, 8))
# Planet VGG p80 cover_frac_binary
plt.scatter(planet_vgg_data["Year"], planet_vgg_data["cover_frac_binary"], color='blue', label='Planet VGG Binary', s=80)
plt.errorbar(planet_vgg_data["Year"], planet_vgg_data["cover_frac_binary"], yerr=planet_vgg_error_binary, fmt='none', ecolor='blue', capsize=5)
# Planet VGG p80 cover_frac_cont
plt.scatter(planet_vgg_data["Year"], planet_vgg_data["cover_frac_cont"], color='cyan', label='Planet VGG Continuous', s=80, marker='s')
plt.errorbar(planet_vgg_data["Year"], planet_vgg_data["cover_frac_cont"], yerr=planet_vgg_error_cont, fmt='none', ecolor='cyan', capsize=5)
# WV/QB VGG p80 cover_frac_binary
plt.scatter(wv_vgg_data["Year"], wv_vgg_data["cover_frac_binary"], color='red', label='WV/QB VGG Binary', s=80)
plt.errorbar(wv_vgg_data["Year"], wv_vgg_data["cover_frac_binary"], yerr=wv_vgg_error_binary, fmt='none', ecolor='red', capsize=5)
# WV/QB VGG p80 cover_frac_cont
plt.scatter(wv_vgg_data["Year"], wv_vgg_data["cover_frac_cont"], color='orange', label='WV/QB VGG Continuous', s=80, marker='s')
plt.errorbar(wv_vgg_data["Year"], wv_vgg_data["cover_frac_cont"], yerr=wv_vgg_error_cont, fmt='none', ecolor='orange', capsize=5)

# Customize plot
plt.xticks(np.arange(2005, 2025, 2), fontsize=20, fontweight='bold')
plt.yticks(np.linspace(0, 0.4, 9), fontsize=20, fontweight='bold')
plt.ylim(0, 0.5)
plt.xlabel('Year', fontsize=20, fontweight='bold')
plt.ylabel('Cover Fraction', fontsize=20, fontweight='bold')
# plt.title('Calibrated VGG19 Tall Shrub Cover Fraction Planet vs. \nWV/QB with Mean Dynamic p80 \nthresholding for site1 area1', fontsize=26, fontweight='bold')
plt.title('Calibrated Cover Fraction Over Time (Planet vs.\n WV/QB Data) based on VGG19 predictions and dynamic\n p80 thresholding for site1 area1', fontsize=26, fontweight='bold')
plt.legend(fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2, loc="upper left",)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show(block=True)










import numpy as np
import matplotlib.pyplot as plt

# Planet Resnet p80 Data for ms4a1
planet_resnet_data = {
    "Year": [2010, 2013, 2014, 2015, 2016, 2017, 2019, 2020, 2021, 2022],
    "cover_frac_binary": [0.297423418, 0.289099027, 0.280680668, 0.304188205, 0.310675085, 0.307440634, 0.299284656, 0.321843395, 0.301558889, 0.328914135],
    "cover_frac_cont": [0.15402018, 0.21810424, 0.16232288, 0.20591816, 0.20891201, 0.28245172, 0.1660665, 0.3237891, 0.17394297, 0.38470143],
    "ci_low_b": [0.268278737, 0.258862183, 0.251113045, 0.274836058, 0.282433985, 0.281157747, 0.270863433, 0.293590785, 0.27221743, 0.299982191],
    "ci_up_b": [0.3265681, 0.31933587, 0.310248291, 0.333540351, 0.338916185, 0.33372352, 0.327705878, 0.350096005, 0.330900348, 0.357846078],
    "ci_low_cont": [0.143398144, 0.204172087, 0.146275052, 0.193717015, 0.196887131, 0.268528427, 0.156567889, 0.307807366, 0.162945492, 0.367560412],
    "ci_up_cont": [0.164642207, 0.2320364, 0.178370705, 0.218119311, 0.220936899, 0.296375012, 0.175565106, 0.339770813, 0.184940445, 0.40184245]
}

# QB/WV Resnet50 p80 Data for ms4a1
wv_resnet_data = {
    "Year": [2002, 2010, 2013, 2014, 2018],
    "cover_frac_binary": [0.140869573, 0.149459124, 0.182025656, 0.143235967, 0.153737217],
    "cover_frac_cont": [0.043089043, 0.025216402, 0.157611936, 0.05698733, 0.051988434],
    "ci_low_b": [0.123305311, 0.131314115, 0.16249857, 0.127321361, 0.137972358],
    "ci_up_b": [0.158433835, 0.167604132, 0.201552742, 0.159150572, 0.169502077],
    "ci_low_cont": [0.037246106, 0.021730621, 0.143596747, 0.050502703, 0.046495384],
    "ci_up_cont": [0.048931981, 0.028702184, 0.171627125, 0.063471958, 0.057481484]
}


## calibration
# Scale Planet data
weighted_cal_slope = 0.32
planet_resnet_data["cover_frac_binary"] = [v * weighted_cal_slope for v in planet_resnet_data["cover_frac_binary"]]
planet_resnet_data["cover_frac_cont"] = [v * weighted_cal_slope for v in planet_resnet_data["cover_frac_cont"]]
planet_resnet_data["ci_low_b"] = [v * weighted_cal_slope for v in planet_resnet_data["ci_low_b"]]
planet_resnet_data["ci_up_b"] = [v * weighted_cal_slope for v in planet_resnet_data["ci_up_b"]]
planet_resnet_data["ci_low_cont"] = [v * weighted_cal_slope for v in planet_resnet_data["ci_low_cont"]]
planet_resnet_data["ci_up_cont"] = [v * weighted_cal_slope for v in planet_resnet_data["ci_up_cont"]]

# Scale WV/QB data
weighted_cal_slope = 0.94
wv_resnet_data["cover_frac_binary"] = [v * weighted_cal_slope for v in wv_resnet_data["cover_frac_binary"]]
wv_resnet_data["cover_frac_cont"] = [v * weighted_cal_slope for v in wv_resnet_data["cover_frac_cont"]]
wv_resnet_data["ci_low_b"] = [v * weighted_cal_slope for v in wv_resnet_data["ci_low_b"]]
wv_resnet_data["ci_up_b"] = [v * weighted_cal_slope for v in wv_resnet_data["ci_up_b"]]
wv_resnet_data["ci_low_cont"] = [v * weighted_cal_slope for v in wv_resnet_data["ci_low_cont"]]
wv_resnet_data["ci_up_cont"] = [v * weighted_cal_slope for v in wv_resnet_data["ci_up_cont"]]


# Function to calculate error bars
def calculate_error(data, low_key, high_key, main_key):
    lower_error = np.array(data[main_key]) - np.array(data[low_key])
    upper_error = np.array(data[high_key]) - np.array(data[main_key])
    return [lower_error, upper_error]

# Calculate error bars for both datasets
planet_resnet_error_binary = calculate_error(planet_resnet_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
planet_resnet_error_cont = calculate_error(planet_resnet_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")
wv_resnet_error_binary = calculate_error(wv_resnet_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
wv_resnet_error_cont = calculate_error(wv_resnet_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")

# Plot
plt.figure(figsize=(14, 10))

# Planet Resnet p80 cover_frac_binary
plt.scatter(planet_resnet_data["Year"], planet_resnet_data["cover_frac_binary"], color='blue', label='Planet Resnet Binary', s=80)
plt.errorbar(planet_resnet_data["Year"], planet_resnet_data["cover_frac_binary"], yerr=planet_resnet_error_binary, fmt='none', ecolor='blue', capsize=5)

# Planet Resnet p80 cover_frac_cont
plt.scatter(planet_resnet_data["Year"], planet_resnet_data["cover_frac_cont"], color='cyan', label='Planet Resnet Continuous', s=80, marker='s')
plt.errorbar(planet_resnet_data["Year"], planet_resnet_data["cover_frac_cont"], yerr=planet_resnet_error_cont, fmt='none', ecolor='cyan', capsize=5)

# WV/QB Resnet50 p80 cover_frac_binary
plt.scatter(wv_resnet_data["Year"], wv_resnet_data["cover_frac_binary"], color='red', label='WV/QB Resnet Binary', s=80)
plt.errorbar(wv_resnet_data["Year"], wv_resnet_data["cover_frac_binary"], yerr=wv_resnet_error_binary, fmt='none', ecolor='red', capsize=5)

# WV/QB Resnet50 p80 cover_frac_cont
plt.scatter(wv_resnet_data["Year"], wv_resnet_data["cover_frac_cont"], color='orange', label='WV/QB Resnet Continuous', s=80, marker='s')
plt.errorbar(wv_resnet_data["Year"], wv_resnet_data["cover_frac_cont"], yerr=wv_resnet_error_cont, fmt='none', ecolor='orange', capsize=5)


# Customize plot
plt.xticks(np.arange(2002, 2025, 2), fontsize=20, fontweight='bold')
plt.yticks(np.linspace(0, 0.4, 9), fontsize=20, fontweight='bold')
plt.ylim(0, 0.4)
plt.xlabel('Year', fontsize=20, fontweight='bold')
plt.ylabel('Cover Fraction', fontsize=20, fontweight='bold')
plt.title('Calibrated Cover Fraction Over Time (Planet vs. \n WV/QB Data) based on ResNet50 predictions and dynamic \n p80 thresholding for site4 area1', fontsize=26, fontweight='bold')
plt.legend(fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2, loc="upper left",)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show(block=True)







import numpy as np
import matplotlib.pyplot as plt

# Planet VGG19 p80 Data for ms4a1
planet_vgg_data = {
    "Year": [2019, 2015, 2021, 2022, 2016, 2017, 2014, 2010, 2020, 2013],
    "cover_frac_binary": [0.28330584, 0.297052562, 0.279693919, 0.395655266, 0.310177642, 0.311906364, 0.288866437, 0.291932417, 0.318463836, 0.311299968],
    "cover_frac_cont": [0.12604575, 0.2486879, 0.12258949, 0.61228764, 0.31936744, 0.32932767, 0.22665656, 0.11517232, 0.2356275, 0.34061596],
    "ci_low_b": [0.26340613, 0.274651887, 0.258647214, 0.365983275, 0.29071606, 0.284157867, 0.267898936, 0.26861504, 0.292360237, 0.290661067],
    "ci_up_b": [0.303205551, 0.319453237, 0.300740624, 0.425327257, 0.329639225, 0.33965486, 0.309833939, 0.315249795, 0.344567435, 0.331938869],
    "ci_low_cont": [0.121983325, 0.242088412, 0.118753427, 0.598563578, 0.312597485, 0.316818061, 0.220139292, 0.11060336, 0.226610232, 0.332700054],
    "ci_up_cont": [0.130108172, 0.255287375, 0.126425556, 0.626011703, 0.326137392, 0.341837285, 0.23317382, 0.119741279, 0.244644772, 0.348531862]
}

# QB/WV VGG19 p80 Data for ms4a1
wv_vgg_data = {
    "Year": [2002, 2010, 2013, 2014, 2018],
    "cover_frac_binary": [0.171816677, 0.164991558, 0.196974337, 0.168912888, 0.179254621],
    "cover_frac_cont": [0.083365738, 0.060305141, 0.176645622, 0.077907592, 0.063985847],
    "ci_low_b": [0.153903059, 0.147251257, 0.178751696, 0.151591025, 0.162108626],
    "ci_up_b": [0.189730295, 0.182731858, 0.215196978, 0.18623475, 0.196400616],
    "ci_low_cont": [0.075521195, 0.05408251, 0.163566594, 0.071210386, 0.058805929],
    "ci_up_cont": [0.091210282, 0.066527772, 0.189724649, 0.084604798, 0.069165765]
}


## calibration
# Scale Planet data
weighted_cal_slope = 0.62
planet_vgg_data["cover_frac_binary"] = [v * weighted_cal_slope for v in planet_vgg_data["cover_frac_binary"]]
planet_vgg_data["cover_frac_cont"] = [v * weighted_cal_slope for v in planet_vgg_data["cover_frac_cont"]]
planet_vgg_data["ci_low_b"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_low_b"]]
planet_vgg_data["ci_up_b"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_up_b"]]
planet_vgg_data["ci_low_cont"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_low_cont"]]
planet_vgg_data["ci_up_cont"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_up_cont"]]

# Scale WV/QB data
weighted_cal_slope = 0.83
wv_vgg_data["cover_frac_binary"] = [v * weighted_cal_slope for v in wv_vgg_data["cover_frac_binary"]]
wv_vgg_data["cover_frac_cont"] = [v * weighted_cal_slope for v in wv_vgg_data["cover_frac_cont"]]
wv_vgg_data["ci_low_b"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_low_b"]]
wv_vgg_data["ci_up_b"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_up_b"]]
wv_vgg_data["ci_low_cont"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_low_cont"]]
wv_vgg_data["ci_up_cont"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_up_cont"]]


# Function to calculate error bars
def calculate_error(data, low_key, high_key, main_key):
    lower_error = np.array(data[main_key]) - np.array(data[low_key])
    upper_error = np.array(data[high_key]) - np.array(data[main_key])
    return [lower_error, upper_error]

# Calculate error bars for both datasets
planet_vgg_error_binary = calculate_error(planet_vgg_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
planet_vgg_error_cont = calculate_error(planet_vgg_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")
wv_vgg_error_binary = calculate_error(wv_vgg_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
wv_vgg_error_cont = calculate_error(wv_vgg_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")

# Plot
plt.figure(figsize=(14, 10))

# Planet VGG19 p80 cover_frac_binary
plt.scatter(planet_vgg_data["Year"], planet_vgg_data["cover_frac_binary"], color='blue', label='Planet VGG19 Binary', s=80)
plt.errorbar(planet_vgg_data["Year"], planet_vgg_data["cover_frac_binary"], yerr=planet_vgg_error_binary, fmt='none', ecolor='blue', capsize=5)

# Planet VGG19 p80 cover_frac_cont
plt.scatter(planet_vgg_data["Year"], planet_vgg_data["cover_frac_cont"], color='cyan', label='Planet VGG19 Continuous', s=80, marker='s')
plt.errorbar(planet_vgg_data["Year"], planet_vgg_data["cover_frac_cont"], yerr=planet_vgg_error_cont, fmt='none', ecolor='cyan', capsize=5)

# WV/QB VGG19 p80 cover_frac_binary
plt.scatter(wv_vgg_data["Year"], wv_vgg_data["cover_frac_binary"], color='red', label='WV/QB VGG19 Binary', s=80)
plt.errorbar(wv_vgg_data["Year"], wv_vgg_data["cover_frac_binary"], yerr=wv_vgg_error_binary, fmt='none', ecolor='red', capsize=5)

# WV/QB VGG19 p80 cover_frac_cont
plt.scatter(wv_vgg_data["Year"], wv_vgg_data["cover_frac_cont"], color='orange', label='WV/QB VGG19 Continuous', s=80, marker='s')
plt.errorbar(wv_vgg_data["Year"], wv_vgg_data["cover_frac_cont"], yerr=wv_vgg_error_cont, fmt='none', ecolor='orange', capsize=5)

# Customize plot
plt.xticks(np.arange(2002, 2025, 2), fontsize=20, fontweight='bold')
plt.yticks(np.linspace(0, 0.4, 9), fontsize=20, fontweight='bold')
plt.ylim(0, 0.5)
plt.xlabel('Year', fontsize=20, fontweight='bold')
plt.ylabel('Cover Fraction', fontsize=20, fontweight='bold')
plt.title('Calibrated Cover Fraction Over Time (Planet vs. \n WV/QB Data) based on VGG19 predictions and dynamic \n p80 thresholding for site4 area1', fontsize=26, fontweight='bold')
plt.legend(fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2, loc="upper left",)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show(block=True)









# ## ms6a2 Resnet

# -----------------------------
# 2) Planet ResNet p80
# -----------------------------

planet_resnet_p80_data = {
    "Year": [2016, 2011, 2017, 2013],
    "cover_frac_binary": [0.235067854, 0.273541545, 0.257801974, 0.251677362],
    "cover_frac_cont": [0.28246778, 0.3246061, 0.24824157, 0.3485382],
    "ci_low_cont": [0.267561184, 0.307444016, 0.232054265, 0.332758757],
    "ci_up_cont": [0.297374381, 0.341768166, 0.264428882, 0.364317624],
    "ci_low_b": [0.210847715, 0.244630937, 0.229062138, 0.2268961],
    "ci_up_b": [0.259287992, 0.302452153, 0.286541811, 0.276458624]
}

resnet_p80_data = {
    "Year": [2009, 2011, 2016, 2017],
    "cover_frac_binary": [0.174662158, 0.159218386, 0.204831541, 0.161887541],
    "cover_frac_cont": [0.082558669, 0.060093138, 0.176600531, 0.087627955],
    "ci_low_cont": [0.073158656, 0.052883464, 0.162446466, 0.077852768],
    "ci_up_cont": [0.091958683, 0.067302813, 0.190754596, 0.097403142],
    "ci_low_b": [0.15444781, 0.140829359, 0.184772223, 0.14320206],
    "ci_up_b": [0.194876506, 0.177607413, 0.224890858, 0.180573023]
}




## calibration
# Scale Planet data
weighted_cal_slope = 0.32
planet_resnet_p80_data["cover_frac_binary"] = [v * weighted_cal_slope for v in planet_resnet_p80_data["cover_frac_binary"]]
planet_resnet_p80_data["cover_frac_cont"] = [v * weighted_cal_slope for v in planet_resnet_p80_data["cover_frac_cont"]]
planet_resnet_p80_data["ci_low_b"] = [v * weighted_cal_slope for v in planet_resnet_p80_data["ci_low_b"]]
planet_resnet_p80_data["ci_up_b"] = [v * weighted_cal_slope for v in planet_resnet_p80_data["ci_up_b"]]
planet_resnet_p80_data["ci_low_cont"] = [v * weighted_cal_slope for v in planet_resnet_p80_data["ci_low_cont"]]
planet_resnet_p80_data["ci_up_cont"] = [v * weighted_cal_slope for v in planet_resnet_p80_data["ci_up_cont"]]

# Scale WV/QB data
weighted_cal_slope = 0.94
resnet_p80_data["cover_frac_binary"] = [v * weighted_cal_slope for v in resnet_p80_data["cover_frac_binary"]]
resnet_p80_data["cover_frac_cont"] = [v * weighted_cal_slope for v in resnet_p80_data["cover_frac_cont"]]
resnet_p80_data["ci_low_b"] = [v * weighted_cal_slope for v in resnet_p80_data["ci_low_b"]]
resnet_p80_data["ci_up_b"] = [v * weighted_cal_slope for v in resnet_p80_data["ci_up_b"]]
resnet_p80_data["ci_low_cont"] = [v * weighted_cal_slope for v in resnet_p80_data["ci_low_cont"]]
resnet_p80_data["ci_up_cont"] = [v * weighted_cal_slope for v in resnet_p80_data["ci_up_cont"]]

# Function to calculate error bars
def calculate_error(data, low_key, high_key, main_key):
    lower_error = np.array(data[main_key]) - np.array(data[low_key])
    upper_error = np.array(data[high_key]) - np.array(data[main_key])
    return [lower_error, upper_error]

# Calculate error bars for both datasets
planet_resnet_error_binary = calculate_error(planet_resnet_p80_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
planet_resnet_error_cont = calculate_error(planet_resnet_p80_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")
wv_resnet_error_binary = calculate_error(resnet_p80_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
wv_resnet_error_cont = calculate_error(resnet_p80_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")


plt.figure(figsize=(14, 10))
# Planet VGG19 p80 cover_frac_binary
plt.scatter(planet_resnet_p80_data["Year"], planet_resnet_p80_data["cover_frac_binary"], color='blue', label='Planet VGG19 Binary', s=80)
plt.errorbar(planet_resnet_p80_data["Year"], planet_resnet_p80_data["cover_frac_binary"], yerr=planet_resnet_error_binary, fmt='none', ecolor='blue', capsize=5)

# Planet VGG19 p80 cover_frac_cont
plt.scatter(planet_resnet_p80_data["Year"], planet_resnet_p80_data["cover_frac_cont"], color='cyan', label='Planet VGG19 Continuous', s=80, marker='s')
plt.errorbar(planet_resnet_p80_data["Year"], planet_resnet_p80_data["cover_frac_cont"], yerr=planet_resnet_error_cont, fmt='none', ecolor='cyan', capsize=5)

# WV/QB VGG19 p80 cover_frac_binary
plt.scatter(resnet_p80_data["Year"], resnet_p80_data["cover_frac_binary"], color='red', label='WV/QB VGG19 Binary', s=80)
plt.errorbar(resnet_p80_data["Year"], resnet_p80_data["cover_frac_binary"], yerr=wv_resnet_error_binary, fmt='none', ecolor='red', capsize=5)

# WV/QB VGG19 p80 cover_frac_cont
plt.scatter(resnet_p80_data["Year"], resnet_p80_data["cover_frac_cont"], color='orange', label='WV/QB VGG19 Continuous', s=80, marker='s')
plt.errorbar(resnet_p80_data["Year"], resnet_p80_data["cover_frac_cont"], yerr=wv_resnet_error_cont, fmt='none', ecolor='orange', capsize=5)

# Customize plot
plt.xticks(np.arange(2009, 2019, 2), fontsize=20, fontweight='bold')
plt.yticks(np.linspace(0, 0.4, 9), fontsize=20, fontweight='bold')
plt.ylim(0, 0.4)
plt.xlabel('Year', fontsize=20, fontweight='bold')
plt.ylabel('Cover Fraction', fontsize=20, fontweight='bold')
# plt.title('Calibrated ResNet50 Tall Shrub Cover Fraction Planet vs. \nWV/QB with Mean Dynamic p80 \nthresholding for site6 area2', fontsize=26, fontweight='bold')
plt.title('Calibrated Cover Fraction Over Time (Planet vs. \n WV/QB Data) based on ResNet50 predictions and dynamic \n p80 thresholding for site6 area2', fontsize=26, fontweight='bold')
plt.legend(fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()












# ## ms6a2 VGG

# -----------------------------
# 2) Planet VGG p80
# -----------------------------
#  Planet VGG data from the spreadsheet:
planet_vgg_data = {
    "Year": [2009, 2011, 2016, 2017],
    "cover_frac_binary": [0.174662158, 0.159218386, 0.204831541, 0.161887541],
    "cover_frac_cont": [0.082558669, 0.060093138, 0.176600531, 0.087627955],
    "ci_low_cont": [0.073158656, 0.052883464, 0.162446466, 0.077852768],
    "ci_up_cont": [0.091958683, 0.067302813, 0.190754596, 0.097403142],
    "ci_low_b": [0.15444781, 0.140829359, 0.184772223, 0.14320206],
    "ci_up_b": [0.194876506, 0.177607413, 0.224890858, 0.180573023]
}

# WorldView VGG data from the spreadsheet:
wv_vgg_data = {
    "Year": [2009, 2011, 2016, 2017],
    "cover_frac_binary": [0.178552568, 0.177282795, 0.222688526, 0.175229222],
    "cover_frac_cont": [0.108389109, 0.100028664, 0.234103978, 0.11286886],
    "ci_low_cont": [0.097353364, 0.089318111, 0.214836287, 0.101910682],
    "ci_up_cont": [0.119424855, 0.110739217, 0.253371668, 0.123827039],
    "ci_low_b": [0.157449135, 0.155380593, 0.198351422, 0.155163386],
    "ci_up_b": [0.199656001, 0.199184998, 0.24702563, 0.195295058]
}



## calibration
# Scale Planet data
weighted_cal_slope = 0.62
planet_vgg_data["cover_frac_binary"] = [v * weighted_cal_slope for v in planet_vgg_data["cover_frac_binary"]]
planet_vgg_data["cover_frac_cont"] = [v * weighted_cal_slope for v in planet_vgg_data["cover_frac_cont"]]
planet_vgg_data["ci_low_b"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_low_b"]]
planet_vgg_data["ci_up_b"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_up_b"]]
planet_vgg_data["ci_low_cont"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_low_cont"]]
planet_vgg_data["ci_up_cont"] = [v * weighted_cal_slope for v in planet_vgg_data["ci_up_cont"]]

# Scale WV/QB data
weighted_cal_slope = 0.83
wv_vgg_data["cover_frac_binary"] = [v * weighted_cal_slope for v in wv_vgg_data["cover_frac_binary"]]
wv_vgg_data["cover_frac_cont"] = [v * weighted_cal_slope for v in wv_vgg_data["cover_frac_cont"]]
wv_vgg_data["ci_low_b"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_low_b"]]
wv_vgg_data["ci_up_b"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_up_b"]]
wv_vgg_data["ci_low_cont"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_low_cont"]]
wv_vgg_data["ci_up_cont"] = [v * weighted_cal_slope for v in wv_vgg_data["ci_up_cont"]]

# Function to calculate error bars
def calculate_error(data, low_key, high_key, main_key):
    lower_error = np.array(data[main_key]) - np.array(data[low_key])
    upper_error = np.array(data[high_key]) - np.array(data[main_key])
    return [lower_error, upper_error]

# Calculate error bars for both datasets
planet_vgg_error_binary = calculate_error(planet_vgg_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
planet_vgg_error_cont = calculate_error(planet_vgg_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")
wv_vgg_error_binary = calculate_error(wv_vgg_data, "ci_low_b", "ci_up_b", "cover_frac_binary")
wv_vgg_error_cont = calculate_error(resnet_p80_data, "ci_low_cont", "ci_up_cont", "cover_frac_cont")


plt.figure(figsize=(14, 10))
# Planet VGG19 p80 cover_frac_binary
plt.scatter(planet_vgg_data["Year"], planet_vgg_data["cover_frac_binary"], color='blue', label='Planet VGG19 Binary', s=80)
plt.errorbar(planet_vgg_data["Year"], planet_vgg_data["cover_frac_binary"], yerr=planet_vgg_error_binary, fmt='none', ecolor='blue', capsize=5)

# Planet VGG19 p80 cover_frac_cont
plt.scatter(planet_vgg_data["Year"], planet_vgg_data["cover_frac_cont"], color='cyan', label='Planet VGG19 Continuous', s=80, marker='s')
plt.errorbar(planet_vgg_data["Year"], planet_vgg_data["cover_frac_cont"], yerr=planet_vgg_error_cont, fmt='none', ecolor='cyan', capsize=5)

# WV/QB VGG19 p80 cover_frac_binary
plt.scatter(wv_vgg_data["Year"], wv_vgg_data["cover_frac_binary"], color='red', label='WV/QB VGG19 Binary', s=80)
plt.errorbar(wv_vgg_data["Year"], wv_vgg_data["cover_frac_binary"], yerr=wv_vgg_error_binary, fmt='none', ecolor='red', capsize=5)

# WV/QB VGG19 p80 cover_frac_cont
plt.scatter(wv_vgg_data["Year"], wv_vgg_data["cover_frac_cont"], color='orange', label='WV/QB VGG19 Continuous', s=80, marker='s')
plt.errorbar(wv_vgg_data["Year"], wv_vgg_data["cover_frac_cont"], yerr=wv_vgg_error_cont, fmt='none', ecolor='orange', capsize=5)

# Customize plot
plt.xticks(np.arange(2009, 2019, 2), fontsize=20, fontweight='bold')
plt.yticks(np.linspace(0, 0.4, 9), fontsize=20, fontweight='bold')
plt.ylim(0, 0.4)
plt.xlabel('Year', fontsize=20, fontweight='bold')
plt.ylabel('Cover Fraction', fontsize=20, fontweight='bold')
# plt.title('Calibrated ResNet50 Tall Shrub Cover Fraction Planet vs. \nWV/QB with Mean Dynamic p80 \nthresholding for site6 area2', fontsize=26, fontweight='bold')
plt.title('Calibrated Cover Fraction Over Time (Planet vs. \n WV/QB Data) based on VGG19 predictions and dynamic \n p80 thresholding for site6 area2', fontsize=26, fontweight='bold')
plt.legend(fontsize=20, labelspacing=0.1, handletextpad=0.1, borderpad=0.1, markerscale=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



