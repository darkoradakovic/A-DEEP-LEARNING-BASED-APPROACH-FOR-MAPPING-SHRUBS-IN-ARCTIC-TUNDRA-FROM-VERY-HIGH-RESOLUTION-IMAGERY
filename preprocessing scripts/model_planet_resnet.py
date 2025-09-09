
### RESNET for PLANET IMAGERY 30-1-25


import os
import numpy as np
import tensorflow as tf
import rasterio
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input, Lambda, UpSampling2D, Conv2D, BatchNormalization, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)


# # # shrubs
# pshp_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train'
# y_train_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/y_train'


# # shrubs OWC Express
# pshp_dir = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train'
# y_train_dir = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/y_train'


# # ADAPT
pshp_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/planet/x_train'
y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/planet/y_train'



##########################################
# 1. Data Generator & Preprocessing Code #
##########################################

def slice_image(image_path, target_slice_size_meters=200):
    """
    Reads the full image from the provided TIFF file.
    Returns a list containing the image array.
    """
    slices = []
    with rasterio.open(image_path) as src:
        # Read the image data: shape (bands, H, W)
        image_data = src.read()
        slices.append(image_data)
    return slices


def data_generator(triplets, target_slice_size_meters=200, scale_factor=5000.0):
    """
    For each (image, label) pair:
      - Reads the data.
      - Reorders the axes to (H, W, channels).
      - Pads from (667, 667) to (672, 672).
      - Normalizes the input image by scale_factor.
    Yields:
      x: Input image with shape (672, 672, 4)
      y: Label with shape (672, 672, 1)
    """
    for image_path, label_path in triplets:
        slices_img = slice_image(image_path, target_slice_size_meters)
        slices_label = slice_image(label_path, target_slice_size_meters)
        for image_path, label_path in triplets:
            slices_img = slice_image(image_path, target_slice_size_meters)
            slices_label = slice_image(label_path, target_slice_size_meters)
            for s_img, s_label in zip(slices_img, slices_label):
                # Convert from (bands, H, W) to (H, W, bands)
                x = np.moveaxis(s_img, 0, -1)  # e.g. (667, 667, 4)
                # Select only the first 3 channels (e.g., RGB)
                x = x[..., :3]  # now shape becomes (667, 667, 3)
                y = np.moveaxis(s_label, 0, -1)  # e.g. (667, 667, 1)

                # Pad images to 672x672 (if original is 667x667)
                x_padded = np.pad(x, ((0, 5), (0, 5), (0, 0)), mode='constant')
                y_padded = np.pad(y, ((0, 5), (0, 5), (0, 0)), mode='constant')

                # Normalize inputs (ensure both training and validation use the same factor)
                x_norm = x_padded.astype(np.float32) / scale_factor

                yield x_norm, y_padded.astype(np.float32)


### MEMORY REDUCUCTION LOAD IMAGES

# Build list of (image_path, label_path) pairs
triplets = []
for file_name in os.listdir(pshp_dir):
    if file_name.endswith('.tif'):
        img_path = os.path.join(pshp_dir, file_name)
        label_path = None
        # Look for the corresponding label file (adjust matching logic as needed)
        for yfile in os.listdir(y_train_dir):
            if file_name[:-4] in yfile and yfile.endswith('.tif'):
                label_path = os.path.join(y_train_dir, yfile)
                break
        if label_path is not None:
            triplets.append((img_path, label_path))

# Split the data into training and validation sets
train_triplets, val_triplets = train_test_split(triplets, test_size=0.1, random_state=42)

# Create tf.data.Dataset objects for training and validation.
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_triplets),
    output_signature=(
        tf.TensorSpec(shape=(672, 672, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(672, 672, 1), dtype=tf.float32)
    )
).batch(8).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_triplets),
    output_signature=(
        tf.TensorSpec(shape=(672, 672, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(672, 672, 1), dtype=tf.float32)
    )
).batch(8).prefetch(tf.data.AUTOTUNE)



##### ---- ---- ---- ----
##### ---- ---- ---- ----  ResNet50 + U-Net-Style Decode
##### ---- ---- ---- ----


## ## ## -- -- --  Download pretrained ResNet model with exact shape
## if issues with Certificates on macos, do this in command:
# ' open "/Applications/Python 3.10/Install Certificates.command" '
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(672, 672, 3))
# base_model.save('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/resnet50_672x672_PLANET_imagenet_pretrained_savedmodel')

# QB WV 400x400x3
# resnet_model = tf.keras.models.load_model('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/resnet50_400x400_QBWV_imagenet_pretrained_savedmodel')
## Planet pretrained on 672x672x3
# resnet_model = tf.keras.models.load_model('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/resnet50_672x672_PLANET_imagenet_pretrained_savedmodel')

# ADAPT 400x400x3
# resnet_model = tf.keras.models.load_model('/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/resnet50_imagenet_savedmodel')
# ADAPT 672x672x3
resnet_model = tf.keras.models.load_model('/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/resnet50_672x672_PLANET_imagenet_pretrained_savedmodel')


def build_resnet50_unet(loaded_resnet, freeze_encoder=True):
    """
    Given a loaded pretrained ResNet50 (with input shape (672, 672, 3)),
    build a U-Net segmentation model that uses its intermediate layers as skip connections.

    Parameters:
      - loaded_resnet: a pretrained ResNet50 model (saved and loaded from disk)
      - freeze_encoder: if True, freeze the encoder weights.

    Returns:
      - A new Keras Model with the same input as loaded_resnet and an added decoder.
    """
    # Use the loaded model's input. It already expects 3-channel input.
    inputs = loaded_resnet.input  # shape: (None, 672, 672, 3)

    # Optionally freeze the encoder layers.
    if freeze_encoder:
        for layer in loaded_resnet.layers:
            layer.trainable = False

    # Extract skip connections by name.
    # (These names come from the original ResNet50 architecture.)
    skip1 = loaded_resnet.get_layer('conv1_relu').output  # e.g., shape: (None, 336, 336, 64)
    skip2 = loaded_resnet.get_layer('conv2_block3_out').output  # e.g., shape: (None, 168, 168, 256)
    skip3 = loaded_resnet.get_layer('conv3_block4_out').output  # e.g., shape: (None, 84, 84, 512)
    skip4 = loaded_resnet.get_layer('conv4_block6_out').output  # e.g., shape: (None, 42, 42, 1024)
    encoder_out = loaded_resnet.get_layer('conv5_block3_out').output  # e.g., shape: (None, 21, 21, 2048)

    # Build the decoder: progressively upsample and merge with the skip connections.
    x = UpSampling2D((2, 2), interpolation='bilinear')(encoder_out)
    x = Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = concatenate([x, skip4])

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = concatenate([x, skip3])

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = concatenate([x, skip2])

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = concatenate([x, skip1])

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Final segmentation output: here we use a 1-channel sigmoid activation.
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='output')(x)

    # Create the new model. Its input is the same as the loaded ResNet50.
    model = Model(inputs=inputs, outputs=outputs, name='ResNet50_U-Net')
    return model


# resnet_model = tf.keras.models.load_model("resnet50_imagenet_savedmodel")
# model_5ch = build_resnet50_unet(resnet_model=resnet_model, input_shape=(400, 400, 5))
model_5ch = build_resnet50_unet(resnet_model, freeze_encoder=True)
model_5ch.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model_5ch.summary()

# checkpoint = ModelCheckpoint('shrub_RESNET50_pretrained_imagenet_weights_trainingdata3_RADnor_400px_PSHP_P1BS_v2_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
# checkpoint = ModelCheckpoint('shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_P1BS_v3_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
checkpoint = ModelCheckpoint('PLANET_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_SR_672px_b0_b3_v2_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                              verbose=1, min_lr=1e-7)

# Train the model
history = model_5ch.fit(
    train_dataset,
    epochs=300,
    validation_data=val_dataset,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)










##### --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
##### --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
##### --- --- --- --- --- --- --- --- --- INFERENCE --- --- --- --- --- --- --- --- --- --- --- ---

import matplotlib.pyplot as plt
import rasterio
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from functionsdarko import (slice_image,combine_tiles_to_large_image_predictionsoneyear,
                            rasterize_shapefile_to_binary, DinoV2DPTSegModel,
                            block_bootstrap_shrub_cover_continuous, block_bootstrap_shrub_cover_binary
                            )

from skimage.filters import threshold_yen
# from sklearn.metrics import confusion_matrix
from skimage.filters import threshold_otsu
from skimage.filters import threshold_li

def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image



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

    arr_batch = arr_batch[...,:3]

    # 5) run inference
    preds = model.predict(arr_batch)  # => shape (1, newH, newW, 1)

    # 6) squeeze => (newH, newW)
    preds = preds[0,...,0]

    # 7) remove padding => back to original H, W
    preds = preds[:H, :W]

    return preds  # e.g. a float mask

# 5 bands ms1a1
tif_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms1a1_370417_2013-08-20_RE5_3A_Analytic_SR_clip_joined4.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms1a1_370417_2013-08-20_RE5_3A_Analytic_SR_clip_joined4.tif'

# 4 bands ms1a1 2020
tif_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms1a1_20200720_203034_83_1067_3B_AnalyticMS_SR_clip.tif'
tif_path = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train_raw/ms1a1_20200720_203034_83_1067_3B_AnalyticMS_SR_clip.tif'

# processed
tif_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train/ms1a1_370417_2013-08-20_RE5_3A_Analytic_SR_clip_joined4_2km3m.tif'

predictions = inference_on_single_planet_tif(tif_path, model)

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(predictions))
plt.show(block=True)

#
# image_test = load_image(tif_path)
#
# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(image_test[3,:,:]))
# plt.show(block=True)






#
# # Calculate thresholds for each image
# thresholds_accumulated = []
# method = 'MEAN_p80'
#
# pred_flat = predictions.flatten()
# pred_flat = pred_flat[~np.isnan(pred_flat)]  # Remove NaN values
# if method == 'MEAN_otsu':   # method='MEAN_otsu'
#     thresholds_accumulated.append(threshold_otsu(pred_flat))
# elif method == 'MEAN_yen':
#     thresholds_accumulated.append(threshold_yen(pred_flat))
# elif method == 'MEAN_li':
#     thresholds_accumulated.append(threshold_li(pred_flat))
# elif method == 'MEAN_p70':
#     thresholds_accumulated.append(np.percentile(pred_flat, 70))
# elif method == 'MEAN_p80':
#     thresholds_accumulated.append(np.percentile(pred_flat, 80))
# elif method == 'MEAN_90':
#     thresholds_accumulated.append(np.percentile(pred_flat, 90))
# else:
#     thresholds_accumulated.append(np.nan)
# threshold_mean = np.mean(thresholds_accumulated)
# combined_predictionimage_MEAN_p70 = predictions > threshold_mean
#
#
#
#
# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(combined_predictionimage_MEAN_p70))
# plt.show(block=True)






### RAW IMAGES

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer

def transform_point(x, y, src_crs, dst_crs="ESRI:102001"):
    """Transform a single point (x,y) from src_crs to dst_crs using pyproj."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x2, y2 = transformer.transform(x, y)
    return x2, y2


import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from affine import Affine


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
        out_arr = out_arr[:3]
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

    # Run inference
    preds = model.predict(input_tensor)
    preds = preds[0, ..., 0]  # Remove batch and channel dimensions

    # Remove the padding to revert to original warped size
    preds = preds[:H, :W]

    # Apply water mask to predictions
    preds[water_mask == 1] = 0.0  # Set water areas to 0.0

    return preds


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

pred_flat = preds.flatten()
pred_flat = pred_flat[~np.isnan(pred_flat)]  # Remove NaN values
thresholds_accumulated.append(threshold_yen(pred_flat))
threshold_mean = np.mean(thresholds_accumulated)
binary_preds = preds > threshold_mean

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(binary_preds))
plt.show(block=True)


# blocksize QB/WV = 200 --- which is 100m --- Planet 100m is ~30px
mean_cov_cont, std_cov_cont, ci_low_cont, ci_up_cont = block_bootstrap_shrub_cover_continuous(preds, block_size=(30, 30), n_boot=10000)
mean_cov_b, std_cov_b, ci_low_b, ci_up_b = block_bootstrap_shrub_cover_binary(binary_preds, block_size=(30, 30), n_boot=10000)



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







# List of methods to evaluate
methods = [
    'MEAN_yen',
    'MEAN_otsu',
    'MEAN_li',
    # 'MEAN_p70',
    # 'MEAN_p80',
    # 'MEAN_90',
    't0.5',
    't0.7',
    't0.8'
]

# Initialize a dictionary to store results
results = {}

# Flatten predictions and remove NaN values
pred_flat = preds.flatten()
pred_flat = pred_flat[~np.isnan(pred_flat)]

# Loop through each method
for method in methods:
    thresholds_accumulated = []

    # Calculate threshold based on the method
    if method == 'MEAN_otsu':
        thresholds_accumulated.append(threshold_otsu(pred_flat))
    elif method == 'MEAN_yen':
        thresholds_accumulated.append(threshold_yen(pred_flat))
    elif method == 'MEAN_li':
        thresholds_accumulated.append(threshold_li(pred_flat))
    elif method == 'MEAN_p70':
        thresholds_accumulated.append(np.percentile(pred_flat, 70))
    elif method == 'MEAN_p80':
        thresholds_accumulated.append(np.percentile(pred_flat, 80))
    elif method == 'MEAN_90':
        thresholds_accumulated.append(np.percentile(pred_flat, 90))
    elif method == 't0.5':
        thresholds_accumulated.append(0.5)
    elif method == 't0.7':
        thresholds_accumulated.append(0.7)
    elif method == 't0.8':
        thresholds_accumulated.append(0.8)
    else:
        thresholds_accumulated.append(np.nan)

    # Compute the mean threshold
    threshold_mean = np.mean(thresholds_accumulated)

    # Apply the threshold to the predictions
    combined_prediction_image = preds > threshold_mean

    # Calculate shrub cover
    shrub_cover = np.sum(combined_prediction_image) / combined_prediction_image.size

    # Store results
    results[method] = {
        'threshold': threshold_mean,
        'shrub_cover': shrub_cover
    }

# Print results
print(f"Results for {tif_path.split('/')[-1]}:")
for method, result in results.items():
    print(f"Method: {method}")
    # print(f"  Threshold: {result['threshold']}")
    print(f"  Shrub Cover: {result['shrub_cover']}")
    print()



plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(combined_predictionimage_MEAN_p70))
plt.show(block=True)





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


# Example usage:
# preds shape => (667,667)
# Suppose you have your final predictions array 'preds'
tile_size = 100
thresh, binary = apply_local_mean_p70(preds, tile_size=tile_size, percentile_value=70)
thresh2, binary2 = apply_local_mean_p70(preds, tile_size=tile_size, percentile_value=80)

#
# # Calculate shrub cover
# combined_prediction_image = preds > thresh
# combined_prediction_image2 = preds > thresh2
# print('shrub cover p70: ', np.sum(combined_prediction_image) / combined_prediction_image.size)
# print('shrub cover p80: ', np.sum(combined_prediction_image2) / combined_prediction_image2.size)


# Now 'binary' is shape (667,667) with 0 or 1
# you can visualize
import matplotlib.pyplot as plt


plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(binary))
plt.show(block=True)


plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(binary2))
plt.show(block=True)

