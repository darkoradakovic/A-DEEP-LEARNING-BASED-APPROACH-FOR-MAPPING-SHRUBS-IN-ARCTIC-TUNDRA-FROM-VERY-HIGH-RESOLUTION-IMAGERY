
## Convert 3m and 5m Planet imagery for "planet_UNET_TRAINING_DATA_v2" into equal uniform images 3x3px

## 28-1-2025


import os
import subprocess
import rasterio
from pyproj import Transformer


## BOTH INPUT OF PLANET IMAGES AND Y-LABELS ARE resized to 667x667px with 3m resolution


def get_planet_info(in_path):
    """
    Returns:
      - the Planet image's bounding box (left,bottom,right,top) in its *native* CRS
      - the Planet image's CRS as a PROJ or EPSG string (e.g. "EPSG:32605")
    Using rasterio.
    """
    with rasterio.open(in_path) as ds:
        left, bottom, right, top = ds.bounds
        crs = ds.crs  # e.g. CRS("EPSG:32605")
    return (left, bottom, right, top, str(crs))

def transform_point(x, y, src_crs, dst_crs):
    """
    Transform a single point from src_crs to dst_crs using pyproj.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x2, y2 = transformer.transform(x, y)
    return (x2, y2)

def warp_planet_2km_3m_utm_to_albers(
    planet_in,
    planet_out,
    left, bottom, right, top,
    planet_crs,
    compress="LZW"
):
    """
    We:
      - compute the center in planet's CRS,
      - transform center to ESRI:102001,
      - define +/- 1000m bounding box,
      - do gdalwarp from planet_crs => ESRI:102001, -ts 667 667,
        keep first 4 bands if it has 5.
    """
    # 1) center in planet's native CRS
    centerX = 0.5*(left + right)
    centerY = 0.5*(bottom + top)

    # 2) transform center to ESRI:102001
    centerX_alb, centerY_alb = transform_point(centerX, centerY, planet_crs, "ESRI:102001")

    # 2x2 km => half_size=1000
    half_size = 1000
    xMin = centerX_alb - half_size
    xMax = centerX_alb + half_size
    yMin = centerY_alb - half_size
    yMax = centerY_alb + half_size

    # check band count to see if we keep 4
    import rasterio
    with rasterio.open(planet_in) as src:
        band_count = src.count

    cmd = [
        "/opt/homebrew/bin/gdalwarp",
        "-s_srs", planet_crs,           # source is e.g. EPSG:32605
        "-t_srs", "ESRI:102001",
        "-te", str(xMin), str(yMin), str(xMax), str(yMax),
        "-ts", "667", "667",
        "-co", f"COMPRESS={compress}",
        "-co", "TILED=YES",
        "-r", "bilinear"
    ]

    # If 5 bands, keep first 4
    if band_count == 5:
        cmd += ["-b", "1", "-b", "2", "-b", "3", "-b", "4"]

    cmd += [planet_in, planet_out]
    print("Warping Planet ->", os.path.basename(planet_out))
    subprocess.run(cmd, check=True)

def warp_label_2km_3m_albers(
    label_in, label_out,
    planet_centerX_alb, planet_centerY_alb
):
    """
    label is already in ESRI:102001,
    so no -s_srs needed. We do the same bounding box:
      center +/- 1000m, 667x667, -r mode
    """
    half_size = 1000
    xMin = planet_centerX_alb - half_size
    xMax = planet_centerX_alb + half_size
    yMin = planet_centerY_alb - half_size
    yMax = planet_centerY_alb + half_size

    cmd = [
        "/opt/homebrew/bin/gdalwarp",
        "-t_srs", "ESRI:102001",
        "-te", str(xMin), str(yMin), str(xMax), str(yMax),
        "-ts", "667", "667",
        "-r", "mode",  # discrete
        label_in,
        label_out
    ]
    print("Warping Label ->", os.path.basename(label_out))
    subprocess.run(cmd, check=True)

def process_planet_and_labels(
    planet_in_dir,
    planet_out_dir,
    labels_in_dir,
    labels_out_dir
):
    """
    For each Planet image (in various UTM zones),
    read bounding box + crs => warp to a 2km box in ESRI:102001 at 3m => 667x667,
    and warp the label to the same bounding box (center).
    """
    os.makedirs(planet_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    planet_fnames = sorted([f for f in os.listdir(planet_in_dir) if f.lower().endswith(".tif")])
    label_fnames = sorted([f for f in os.listdir(labels_in_dir) if f.lower().endswith(".tif")])

    label_dict = {}
    for lf in label_fnames:
        base = os.path.splitext(lf)[0]
        label_dict[base] = lf

    for pf in planet_fnames:
        planet_in_path = os.path.join(planet_in_dir, pf)
        base_p = os.path.splitext(pf)[0]
        planet_out_name = base_p + "_2km3m.tif"
        planet_out_path = os.path.join(planet_out_dir, planet_out_name)

        # read bounding box in native CRS
        left, bottom, right, top, planet_crs = get_planet_info(planet_in_path)

        # warp planet
        warp_planet_2km_3m_utm_to_albers(
            planet_in=planet_in_path,
            planet_out=planet_out_path,
            left=left, bottom=bottom, right=right, top=top,
            planet_crs=planet_crs
        )

        # get the center in Planet's CRS => transform to Albers
        centerX = 0.5*(left + right)
        centerY = 0.5*(bottom + top)
        centerX_alb, centerY_alb = transform_point(centerX, centerY, planet_crs, "ESRI:102001")

        # warp label if we have one
        if base_p in label_dict:
            label_in_path = os.path.join(labels_in_dir, label_dict[base_p])
            label_out_name = base_p + "_2km3m.tif"
            label_out_path = os.path.join(labels_out_dir, label_out_name)
            warp_label_2km_3m_albers(
                label_in=label_in_path,
                label_out=label_out_path,
                planet_centerX_alb=centerX_alb,
                planet_centerY_alb=centerY_alb
            )
        else:
            print(f"No matching label for {pf}")

if __name__ == "__main__":
    planet_in_dir = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw"
    planet_out_dir = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train"
    labels_in_dir = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/y_train_raw/all_multisites_QB_WV_resolution"
    labels_out_dir = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/y_train/final"

    process_planet_and_labels(planet_in_dir, planet_out_dir, labels_in_dir, labels_out_dir)
    print("All done!")


# ## Gliht clip1 planet
# if __name__ == "__main__":
#     planet_in_dir = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/G_LIHT/clip1/Planet clip1"
#     planet_out_dir = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/G_LIHT/clip1/Planet clip1"
#     labels_in_dir = ""
#     labels_out_dir = ""
#
#     process_planet_and_labels(planet_in_dir, planet_out_dir, labels_in_dir, labels_out_dir)
#     print("All done!")
#



### Other maps convert, other dimension with bounding box

import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
import numpy as np
import math


def reproject_and_clip(src_path, dst_crs, dst_width, dst_height, bbox, res):
    """
    Reprojects, clips, and resamples the source image in one step.

    Parameters:
      src_path (str): Path to the original Planet image.
      dst_crs (str): Target CRS (e.g., "ESRI:102001").
      dst_width (int): Output width in pixels.
      dst_height (int): Output height in pixels.
      bbox (tuple): Bounding box (xmin, ymin, xmax, ymax) in the target CRS.
      res (float): Target resolution (meters per pixel).

    Returns:
      out_arr (np.ndarray): The reprojected, clipped, and resampled image array.
      dst_transform (Affine): The affine transform for the output image.
    """
    xmin, ymin, xmax, ymax = bbox

    # Define the affine transform for the target image.
    # For a north-up image, transform = (pixel_size, 0, xmin, 0, -pixel_size, ymax)
    dst_transform = Affine(res, 0, xmin, 0, -res, ymax)

    with rasterio.open(src_path) as src:
        # Create an output array with shape (bands, dst_height, dst_width)
        out_arr = np.zeros((src.count, dst_height, dst_width), dtype=src.dtypes[0])
        for i in range(src.count):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=out_arr[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear  # change if needed
            )
    return out_arr, dst_transform


# ------------------------------------------
# Step 1. Read your bounding box shapefile.
# ------------------------------------------
bbox_gdf = gpd.read_file(
    # "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/ms6_area5_clip_1950m.shp"
    "/Volumes/OWC Express 1M2/nasa_above/clip_georef/ms6_area5_100m_clip_1950m_georef/ms6_area5_clip_1950m.shp"
)

# gliht clip1
bbox_gdf = gpd.read_file(
    # "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/ms6_area5_clip_1950m.shp"
    "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/G_LIHT/clip1/extent_clip1.shp"
)


# Ensure the shapefile is in the target CRS (ESRI:102001)
bbox_gdf = bbox_gdf.to_crs("ESRI:102001")

# ------------------------------------------
# Step 2. Extract the bounding box coordinates.
# ------------------------------------------
xmin, ymin, xmax, ymax = bbox_gdf.total_bounds
bbox_target = (xmin, ymin, xmax, ymax)
print("Target bounding box:", bbox_target)

# ------------------------------------------
# Step 3. Set target parameters.
# ------------------------------------------
dst_crs = "ESRI:102001"
res = 3  # desired resolution in meters per pixel

# Instead of hardcoding the pixel dimensions, compute them from the bounding box:
dst_width = int(math.ceil((xmax - xmin) / res))
dst_height = int(math.ceil((ymax - ymin) / res))

print(f"Computed dimensions: width={dst_width} pixels, height={dst_height} pixels")

# ------------------------------------------
# Step 4. Reproject, clip, and resample the image.
# ------------------------------------------
## 2024 ms6a5
# src_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/planet validation toolik download large images/20240719_205848_70_24a8_3B_AnalyticMS_SR_clip.tif"
## 2016 ms6a5
# src_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/planet validation toolik download large images/670810_2016-07-25_RE2_3A_Analytic_SR_clip.tif'
## 2010 ms6a5
src_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/planet validation toolik download large images/670810_2010-06-25_RE4_3A_Analytic_SR_clip.tif'

## 2011 ms6a5
src_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_2011_2013-2x_2016_2017_reorthotile_analytic_sr/REOrthoTile/ms6a5_570820_2011-07-09_RE2_3A_Analytic_SR_clip.tif'
## 2013 ms6a5
src_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_2011_2013-2x_2016_2017_reorthotile_analytic_sr/REOrthoTile/ms6a5_570820_2013-08-26_RE4_3A_Analytic_SR_clip.tif'
## 2017 ms6a5
src_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_2011_2013-2x_2016_2017_psscene_analytic_sr_udm2/PSScene/ms6a5_20170728_205413_1025_3B_AnalyticMS_SR_clip.tif'


## Gliht clip1 planet
# PLANET 2012
src_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/G_LIHT/clip1/Planet clip1/clip1_2012_669716_2012-08-13_RE3_3A_Analytic_SR_clip.tif'
# PLANET 2016
src_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/G_LIHT/clip1/Planet clip1/clip1_2016_669716_2016-07-11_RE3_3A_Analytic_SR_clip.tif'
src_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/G_LIHT/clip1/Planet clip1/clip1_2016_669716_2016-07-11_RE3_3A_Analytic_SR_clip.tif'
# PLANET 2019
src_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/G_LIHT/clip1/Planet clip1/clip1_2019_20190822_205842_1024_3B_AnalyticMS_SR_clip.tif'



output_array, out_transform = reproject_and_clip(
    src_path=src_path,
    dst_crs=dst_crs,
    dst_width=dst_width,
    dst_height=dst_height,
    bbox=bbox_target,
    res=res
)

# Now, output_array will exactly cover the bounding box defined by bbox_target
# at a resolution of 3 m/pixel.

# At this point:
# - output_array will have shape (bands, 667, 667).
# - You can then process or save the output as needed.

# save_filename = src_path.split('/')[-1].replace('.tif','_1950m.tif')
save_filename = src_path.split('/')[-1].replace('.tif','_1950m.tif')
savepath_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet'
output_path = os.path.join(savepath_dir,save_filename)

dst_meta = {
    "driver": "GTiff",
    "height": output_array.shape[1],
    "width": output_array.shape[2],
    "count": output_array.shape[0],  # number of bands
    "dtype": output_array.dtype,
    "crs": dst_crs,  # for example "ESRI:102001"
    "transform": out_transform,
}

# Write the output image.
with rasterio.open(output_path, "w", **dst_meta) as dst:
    dst.write(output_array)









##### --- --- --- --- --- Validation Planet --- --- --- --- ---  --- --- --- --- ---

import matplotlib.pyplot as plt
import rasterio
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from functionsdarko import (slice_image,combine_tiles_to_large_image_predictionsoneyear,
                            rasterize_shapefile_to_binary, DinoV2DPTSegModel
                            )



from functionsdarko import (slice_image, load_sliced_images_and_metadata,
    process_and_store_predictions_with_validation4,
    process_validation4, process_validation4_smooth,
    process_and_store_predictions_with_validation5,
    process_images_evi5,
    process_and_store_predictions_with_validation44_4,
    process_and_store_predictions_with_validation44_6,
    cnn_segmentation_model, upgraded_cnn_comb_model, f1_score,
    apply_clustering, process_and_cluster_images,
    make_prediction,
    dynamic_threshold_adjustment,
    calculate_metrics_allthreshold, predictions_pshp_p1bs_val, predictions_vit_p1bs_val,
    DinoV2DPTSegModel
    )


from skimage.filters import threshold_yen
# from sklearn.metrics import confusion_matrix
from skimage.filters import threshold_otsu
from skimage.filters import threshold_li

def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        meta = src.meta.copy()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image, meta


### MODEL LOADING CHANGE inference_on_single_planet_tif arr_batch for amount of bands/channels

# v1 Planet VGG ep17
model = load_model(
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/PLANET_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_672px_PSHP_b0_b3_v1_ADAPT_ep17.tf'
)

# v2 Planet ResNet ep40
model = load_model(
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/PLANET_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_SR_672px_b0_b3_v2_ADAPT_ep40.tf'
)
# v2 Planet ResNet ep66
model = load_model(
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/PLANET_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_SR_672px_b0_b3_v2_ADAPT_ep66.tf'
)


def inference_on_single_planet_tif(tif_path, model, pad_to=672):
    # 1) read
    with rasterio.open(tif_path) as src:
        arr = src.read()  # shape (bands, H, W)
    arr = np.moveaxis(arr, 0, -1)  # => (H, W, bands), e.g. 4 bands

    # 2) if shape is not multiple-of-32, pad
    H, W, C = arr.shape
    padH = (32 - (H % 32)) % 32
    padW = (32 - (W % 32)) % 32
    arr_padded = np.pad(arr, ((0,padH),(0,padW),(0,0)), mode='constant')

    # 3) normalize
    arr_norm = arr_padded / 5000.0  # or your scale factor

    # 4) expand dims => (1, newH, newW, 4)
    arr_batch = np.expand_dims(arr_norm, axis=0)

    # arr_batch = arr_batch[..., :3]  # RESNET
    arr_batch = arr_batch[..., :4]  # VGG19

    # 5) run inference
    preds = model.predict(arr_batch)  # => shape (1, newH, newW, 1)

    # 6) squeeze => (newH, newW)
    preds = preds[0,...,0]

    # 7) remove padding => back to original H, W
    preds = preds[:H, :W]

    return preds  # e.g. a float mask

# ms6a5 2010
# tif_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2010_670810_2010-06-25_RE4_3A_Analytic_SR_clip_1950m.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2010_670810_2010-06-25_RE4_3A_Analytic_SR_clip_1950m.tif'

# ms6a5 2011
tif_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2010_670810_2010-06-25_RE4_3A_Analytic_SR_clip_1950m.tif'

prediction = inference_on_single_planet_tif(tif_path, model)
# prediction = combined_image
# prediction = np.squeeze(combined_predictions['QB02_2013'])

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(prediction))
plt.show(block=True)

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


# Example usage:
# preds shape => (667,667)
# Suppose you have your final predictions array 'preds'
tile_size = 100
thresh, binary = apply_local_mean_p70(prediction, tile_size=tile_size, percentile_value=70)
thresh2, binary2 = apply_local_mean_p70(prediction, tile_size=tile_size, percentile_value=80)

# binary_resized = np.squeeze(pred_binary_dict['QB02_2013']['p70']['binary_mask'])

## Resize binary mask to 3901x3901 (nearest-neighbor preserves 0/1)
from skimage.transform import resize
binary_resized = resize(
    binary.astype(np.uint8),
    (3901, 3901),
    order=0,  # Nearest-neighbor
    preserve_range=True,
    anti_aliasing=False
).astype(np.uint8)



plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(binary))
plt.show(block=True)

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(binary_resized))
plt.show(block=True)




### test 2

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix


def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        meta = src.meta.copy()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image, meta

def reproject_validation_to_prediction(validation_path, pred_meta, shrub_classes=[5, 11, 12]):
    """
    Reprojects and resamples a high-resolution validation map to match the prediction grid.

    In addition to creating a binary mask (1 for shrub classes, 0 otherwise), this function
    also creates a "valid mask" that indicates where the original validation map had data (nonzero).

    Parameters:
      validation_path (str): Path to the validation GeoTIFF.
      pred_meta (dict): Metadata dictionary from the prediction (including transform, width, height, and CRS).
      shrub_classes (list): List of integer classes that represent tall shrubs.

    Returns:
      binary_resampled (np.ndarray): Binary validation map matching the prediction grid.
      valid_resampled (np.ndarray): A mask (1=valid, 0=no-data) matching the prediction grid.
    """
    with rasterio.open(validation_path) as src:
        # Read the original validation map (assumed one band)
        val_img = src.read(1)  # shape (rows, cols)

        # Create binary mask for shrub classes:
        # Pixels belonging to any of the shrub classes become 1, all others become 0.
        binary_mask = np.isin(val_img, shrub_classes).astype(np.uint8)

        # Create valid mask: valid where the original image has values greater than 0.
        # (Assuming that 0 indicates "no data" or "not covered".)
        valid_mask = (val_img > 0).astype(np.uint8)

        # Prepare output parameters from pred_meta.
        dst_height = pred_meta['height']
        dst_width = pred_meta['width']
        dst_crs = pred_meta['crs']
        dst_transform = pred_meta['transform']

        # Prepare output arrays.
        binary_resampled = np.zeros((dst_height, dst_width), dtype=np.uint8)
        valid_resampled = np.zeros((dst_height, dst_width), dtype=np.uint8)

        # Reproject binary mask.
        reproject(
            source=binary_mask,
            destination=binary_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest  # preserve categorical values
        )

        # Reproject valid mask.
        reproject(
            source=valid_mask,
            destination=valid_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

    return binary_resampled, valid_resampled




# Example usage:
# Suppose your prediction metadata (pred_meta) is something like:
# pred_meta = {'height': 650, 'width': 650, 'crs': "ESRI:102001", 'transform': <Affine object>}
# validation_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/y_train_toolik/toolik_veg_classmap_area5_clip1950m.tif"
validation_path = "/Volumes/OWC Express 1M2/nasa_above/clip_georef/ms6_area5_100m_clip_1950m_georef/y_train_toolik/toolik_veg_classmap_area5_clip1950m.tif"
# binary_val_map, valid_mask = reproject_validation_to_prediction(validation_path, metadata_dict['2010'][0], shrub_classes=[5, 11, 12])
# ms6a5 2010
# tif_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2010_670810_2010-06-25_RE4_3A_Analytic_SR_clip_1950m.tif'


# validation map of 3901px
_, meta = load_image(validation_path)
binary_val_map_3901px, valid_mask_3901px = reproject_validation_to_prediction(validation_path, meta, shrub_classes=[5, 11, 12])
# binary_val_map_3901px = np.squeeze(combined_val_images['QB02_2013'])

# validation map of 650px NEEDED FOR ROC if PLANET
tif_path_650px = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2010_670810_2010-06-25_RE4_3A_Analytic_SR_clip_1950m.tif'
_, meta = load_image(tif_path_650px)
binary_val_map_650px, valid_mask_650px = reproject_validation_to_prediction(validation_path, meta, shrub_classes=[5, 11, 12])



plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(binary_val_map_3901px))
plt.show(block=True)

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(valid_mask_3901px))
plt.show(block=True)



if len(np.unique(binary_val_map_650px.flatten())) > 1:  # Ensure there are at least two classes present
    fpr, tpr, thresholds = roc_curve(binary_val_map_650px.flatten(), prediction.flatten())
    roc_auc = auc(fpr, tpr)
else:
    roc_auc = np.nan  # Append NaN if ROC AUC cannot be calculated


# Mask validation data (use your existing valid_mask)
y_true = binary_val_map_3901px[valid_mask_3901px == 1].flatten()
y_pred = binary_resized[valid_mask_3901px == 1].flatten()

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
# tp, fn, fp, tn = cm.ravel()
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"{accuracy:.2f}, {precision:.2f}, {recall:.2f}, {f1:.2f}, {(np.sum(binary_resized) / binary_resized.size):.2f}, {roc_auc:.2f}")


# ## ms6a5 2010 VGG
# 0.82, 0.94, 0.84, 0.89, 0.27, 0.78  # with mask

# WV image 2011 ms6a5 VGG
# 0.79, 0.90, 0.84, 0.87, 0.23, 0.78
# 0.78, 0.92, 0.81, 0.86, 0.27, nan  ## QB 2013 VGG
# 0.84, 0.96, 0.84, 0.90, 0.28, nan  ## QB 2013 VGG via validation
# 0.84, 0.49, 0.80, 0.61, 0.28, 0.86  ## QB 2013 VGG via validation with correct CN order
# 0.84, 0.49, 0.80, 0.61, 0.28, 0.86


# ## CONFUSION METRICS ON 650x650px gives too good results

y_true = binary_val_map_650px[valid_mask_650px == 1].flatten()
y_pred = binary[valid_mask_650px == 1].flatten()

# Calculate the confusion matrix.
cm = confusion_matrix(y_true, y_pred)
# tp, fn, fp, tn = cm.ravel()
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / np.sum(cm)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"{accuracy:.2f}, {precision:.2f}, {recall:.2f}, {f1:.2f}, {(np.sum(binary) / binary.size):.2f}, {roc_auc:.2f}")



# ## ms6a5 2010 REsnet
# 0.75, 0.93, 0.76, 0.83, 0.32, 0.76   # with mask ep40
# 0.76, 0.92, 0.77, 0.84, 0.32, 0.75   # with mask ep66

# ## ms6a5 2010 VGG
# 0.76, 0.21, 0.72, 0.32, 0.27, 0.78  # without mask
# 0.82, 0.94, 0.84, 0.89, 0.27, 0.78  # with mask





# LOOP PLANET VALIDATION METRICS
##### --- --- --- --- --- Validation Planet --- --- --- --- ---  --- --- --- --- ---

import os
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from functionsdarko import (slice_image,combine_tiles_to_large_image_predictionsoneyear,
                            rasterize_shapefile_to_binary, DinoV2DPTSegModel
                            )



from functionsdarko import (slice_image, load_sliced_images_and_metadata,
    process_and_store_predictions_with_validation4,
    process_validation4, process_validation4_smooth,
    process_and_store_predictions_with_validation5,
    process_images_evi5,
    process_and_store_predictions_with_validation44_4,
    process_and_store_predictions_with_validation44_6,
    cnn_segmentation_model, upgraded_cnn_comb_model, f1_score,
    apply_clustering, process_and_cluster_images,
    make_prediction,
    dynamic_threshold_adjustment,
    calculate_metrics_allthreshold, predictions_pshp_p1bs_val, predictions_vit_p1bs_val,
    DinoV2DPTSegModel
    )


from skimage.filters import threshold_yen
# from sklearn.metrics import confusion_matrix
from skimage.filters import threshold_otsu
from skimage.filters import threshold_li

def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        meta = src.meta.copy()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image, meta


### MODEL LOADING CHANGE inference_on_single_planet_tif arr_batch for amount of bands/channels

# v1 Planet VGG ep17
model = load_model(
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/PLANET_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_672px_PSHP_b0_b3_v1_ADAPT_ep17.tf'
)




def inference_on_single_planet_tif(tif_path, model, pad_to=672):
    # 1) read
    with rasterio.open(tif_path) as src:
        arr = src.read()  # shape (bands, H, W)
    arr = np.moveaxis(arr, 0, -1)  # => (H, W, bands), e.g. 4 bands

    # 2) if shape is not multiple-of-32, pad
    H, W, C = arr.shape
    padH = (32 - (H % 32)) % 32
    padW = (32 - (W % 32)) % 32
    arr_padded = np.pad(arr, ((0,padH),(0,padW),(0,0)), mode='constant')

    # 3) normalize
    arr_norm = arr_padded / 5000.0  # or your scale factor

    # 4) expand dims => (1, newH, newW, 4)
    arr_batch = np.expand_dims(arr_norm, axis=0)

    # arr_batch = arr_batch[..., :3]  # RESNET
    arr_batch = arr_batch[..., :4]  # VGG19

    # 5) run inference
    preds = model.predict(arr_batch)  # => shape (1, newH, newW, 1)

    # 6) squeeze => (newH, newW)
    preds = preds[0,...,0]

    # 7) remove padding => back to original H, W
    preds = preds[:H, :W]

    return preds  # e.g. a float mask


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
            patch = preds[i:i + tile_size, j:j + tile_size]
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


import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix


def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        meta = src.meta.copy()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image, meta


def reproject_validation_to_prediction(validation_path, pred_meta, shrub_classes=[5, 11, 12]):
    """
    Reprojects and resamples a high-resolution validation map to match the prediction grid.

    In addition to creating a binary mask (1 for shrub classes, 0 otherwise), this function
    also creates a "valid mask" that indicates where the original validation map had data (nonzero).

    Parameters:
      validation_path (str): Path to the validation GeoTIFF.
      pred_meta (dict): Metadata dictionary from the prediction (including transform, width, height, and CRS).
      shrub_classes (list): List of integer classes that represent tall shrubs.

    Returns:
      binary_resampled (np.ndarray): Binary validation map matching the prediction grid.
      valid_resampled (np.ndarray): A mask (1=valid, 0=no-data) matching the prediction grid.
    """
    with rasterio.open(validation_path) as src:
        # Read the original validation map (assumed one band)
        val_img = src.read(1)  # shape (rows, cols)

        # Create binary mask for shrub classes:
        # Pixels belonging to any of the shrub classes become 1, all others become 0.
        binary_mask = np.isin(val_img, shrub_classes).astype(np.uint8)

        # Create valid mask: valid where the original image has values greater than 0.
        # (Assuming that 0 indicates "no data" or "not covered".)
        valid_mask = (val_img > 0).astype(np.uint8)

        # Prepare output parameters from pred_meta.
        dst_height = pred_meta['height']
        dst_width = pred_meta['width']
        dst_crs = pred_meta['crs']
        dst_transform = pred_meta['transform']

        # Prepare output arrays.
        binary_resampled = np.zeros((dst_height, dst_width), dtype=np.uint8)
        valid_resampled = np.zeros((dst_height, dst_width), dtype=np.uint8)

        # Reproject binary mask.
        reproject(
            source=binary_mask,
            destination=binary_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest  # preserve categorical values
        )

        # Reproject valid mask.
        reproject(
            source=valid_mask,
            destination=valid_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

    return binary_resampled, valid_resampled


# Example usage:
# Suppose your prediction metadata (pred_meta) is something like:
# pred_meta = {'height': 650, 'width': 650, 'crs': "ESRI:102001", 'transform': <Affine object>}
validation_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area5_100m_clip_1950m_georef/y_train_toolik/toolik_veg_classmap_area5_clip1950m.tif"
# validation_path = "/Volumes/OWC Express 1M2/nasa_above/clip_georef/ms6_area5_100m_clip_1950m_georef/y_train_toolik/toolik_veg_classmap_area5_clip1950m.tif"
# binary_val_map, valid_mask = reproject_validation_to_prediction(validation_path, metadata_dict['2010'][0], shrub_classes=[5, 11, 12])
# ms6a5 2010
# tif_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2010_670810_2010-06-25_RE4_3A_Analytic_SR_clip_1950m.tif'


tif_path_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet'

## LOOP PLANET VALIDATION METRCIS
for path in os.listdir(tif_path_dir):
    if path.endswith('.tif'):
        tif_path = os.path.join(tif_path_dir, path)

        prediction = inference_on_single_planet_tif(tif_path, model)
        # prediction = combined_image
        # prediction = np.squeeze(combined_predictions['QB02_2013'])

        # Example usage:
        # preds shape => (667,667)
        # Suppose you have your final predictions array 'preds'
        tile_size = 100
        thresh, binary = apply_local_mean_p70(prediction, tile_size=tile_size, percentile_value=70)
        # thresh2, binary2 = apply_local_mean_p70(prediction, tile_size=tile_size, percentile_value=80)

        # binary_resized = np.squeeze(pred_binary_dict['QB02_2013']['p70']['binary_mask'])

        ## Resize binary mask to 3901x3901 (nearest-neighbor preserves 0/1)
        from skimage.transform import resize
        binary_resized = resize(
            binary.astype(np.uint8),
            (3901, 3901),
            order=0,  # Nearest-neighbor
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.uint8)


        ### test 2


        # validation map of 3901px
        _, meta = load_image(validation_path)
        binary_val_map_3901px, valid_mask_3901px = reproject_validation_to_prediction(validation_path, meta, shrub_classes=[5, 11, 12])
        # binary_val_map_3901px = np.squeeze(combined_val_images['QB02_2013'])

        # validation map of 650px NEEDED FOR ROC if PLANET
        tif_path2 = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2010_670810_2010-06-25_RE4_3A_Analytic_SR_clip_1950m.tif'
        # tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/validation/ms6a5_validation_planet/ms6a5_2010_670810_2010-06-25_RE4_3A_Analytic_SR_clip_1950m.tif'
        _, meta = load_image(tif_path2)
        binary_val_map_650px, valid_mask_650px = reproject_validation_to_prediction(validation_path, meta, shrub_classes=[5, 11, 12])



        if len(np.unique(binary_val_map_650px.flatten())) > 1:  # Ensure there are at least two classes present
            fpr, tpr, thresholds = roc_curve(binary_val_map_650px.flatten(), prediction.flatten())
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = np.nan  # Append NaN if ROC AUC cannot be calculated


        # Mask validation data (use your existing valid_mask)
        y_true = binary_val_map_3901px[valid_mask_3901px == 1].flatten()
        y_pred = binary_resized[valid_mask_3901px == 1].flatten()

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # tp, fn, fp, tn = cm.ravel()
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(path)
        print(f"{accuracy:.2f}, {precision:.2f}, {recall:.2f}, {f1:.2f}, {(np.sum(binary_resized) / binary_resized.size):.2f}, {roc_auc:.2f}")


