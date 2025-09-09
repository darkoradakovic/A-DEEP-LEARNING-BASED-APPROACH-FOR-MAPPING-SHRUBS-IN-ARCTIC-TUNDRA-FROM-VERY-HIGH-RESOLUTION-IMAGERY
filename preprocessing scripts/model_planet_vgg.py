
### VGG MODEL 28-1-25

import os
import rasterio
from rasterio.windows import Window
from skimage.transform import resize
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
from tensorflow.keras.applications import ResNet50

from functionsdarko import (slice_image,combine_tiles_to_large_image_predictionsoneyear
                            )

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, concatenate,
                                     BatchNormalization, Activation, Dropout, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, Cropping2D
from tensorflow.keras.regularizers import l2

from ndvi_threshold2_ndwi_shapefile import tiff_path

# # shrubs
# pshp_dir = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/x_train'
# # p1sb_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/P1BS_files'
# y_train_dir = '/Volumes/OWC Express 1M2/nasa_above/Planet images UNET_TRAINING_DATA_v2/planet_UNET_TRAINING_DATA_v2/y_train'
#
# vgg19_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


# # ADAPT
pshp_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/planet/x_train'
# p1sb_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/P1BS_files'
#y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/final'
y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/planet/y_train'

vgg19_path = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'




def slice_image(image_path, target_slice_size_meters=100):
    slices = []
    metadata = []  # List to store metadata for each slice
    with rasterio.open(image_path) as src:
        # pixel_size = src.res[0]
        sliced_data = src.read()
        slice_meta = src.meta.copy()
        # slice_size_pixels = int(target_slice_size_meters / pixel_size)
        #
        # for i in range(0, src.width, slice_size_pixels):
        #     for j in range(0, src.height, slice_size_pixels):
        #         width = min(slice_size_pixels, src.width - i)
        #         height = min(slice_size_pixels, src.height - j)
        #         window = Window(i, j, width, height)
        #         sliced_data = src.read(window=window)
        #
        #         # Copy metadata and update relevant fields for the slice
        #         slice_meta = src.meta.copy()
        #         slice_meta.update({
        #             'height': height,
        #             'width': width,
        #             'transform': rasterio.windows.transform(window, src.transform)
        #         })

        slices.append(sliced_data)
        metadata.append(slice_meta)  # Store metadata for the slice

    return slices, metadata


target_slice_size_meters = 200  ## in meter


### MEMORY REDUCUCTION LOAD IMAGES

# Collect image triplets
image_triplets = []

for file_name in os.listdir(pshp_dir):
    if file_name.endswith('.tif'):
        # Full path to the PSHP and P1BS images
        pshp_image_path = os.path.join(pshp_dir, file_name)
        # p1bs_image_path = pshp_image_path.replace('x_train', 'P1BS_files').replace('PSHP', 'P1BS')

        # Find corresponding y_train image
        image_path_y_train = None
        for yfilename in os.listdir(y_train_dir):
            if file_name[:-4] in yfilename and yfilename.endswith('.tif'):
                image_path_y_train = os.path.join(y_train_dir, yfilename)
                break  # Stop searching once the correct label is found

        # If corresponding y_train image exists, proceed
        if image_path_y_train:
            # image_triplets.append((pshp_image_path, p1bs_image_path, image_path_y_train))
            image_triplets.append((pshp_image_path, image_path_y_train))


from sklearn.model_selection import train_test_split

train_triplets, val_triplets = train_test_split(image_triplets, test_size=0.1, random_state=42)




# Define the data generator function
def data_generator_from_triplets(triplets, target_slice_size_meters):
    for pshp_image_path, image_path_y_train in triplets:
        # Slice PSHP and P1BS images
        slices_pshp, _ = slice_image(pshp_image_path, target_slice_size_meters)
        # slices_p1bs, _ = slice_image(p1bs_image_path, target_slice_size_meters)
        slices_label, _ = slice_image(image_path_y_train, target_slice_size_meters)

        # For each slice, process and yield
        for s_pshp, s_label in zip(slices_pshp, slices_label):
            # Move axis to have channels at the end

            x_slice = np.moveaxis(s_pshp, 0, -1)  # Shape: (height, width, channels)
            # s_p1bs = np.moveaxis(s_p1bs, 0, -1)  # Shape: (height, width, channels)
            x_slice = x_slice.astype(np.float32)

            ## original input size in height/width is not multiple-of-32 (for 5 pooling layers)
            # you often get an off-by-one mismatch in the final upsample steps. For instance, 667 / 32 = 20.8..., not an integer, so at some stage you end up with 82 vs. 83 in the skip connections.
            # If image is 667, you can do 5 px of zero-padding
            # => 667 + 5 = 672
            # Then your model sees 672, which is 21×32
            # after reading the 667×667 image, do manual zero-padding to (672,672):
            padded = np.pad(x_slice, ((0, 5), (0, 5), (0, 0)), mode='constant')  # top/left, or center pad
            # feed that to the model

            # Concatenate PSHP and P1BS along the channel axis to form x_slice
            # x_slice = np.concatenate((s_pshp, s_p1bs), axis=-1)  # Shape: (height, width, 5)
            # x_slice = x_slice / 255.0
            x_slice = padded / 5000.0

            # Process y_slice
            y_slice = np.moveaxis(s_label, 0, -1).astype('float32')  # Shape: (height, width, 1)
            y_slice_padded = np.pad(y_slice, ((0, 5), (0, 5), (0, 0)), mode='constant')

            yield x_slice, y_slice_padded




batch_size = 8  # cnn_model3 (test wet tundra)

# Create datasets with output_signature
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator_from_triplets(train_triplets, target_slice_size_meters),
    output_signature=(
        # tf.TensorSpec(shape=(667, 667, 4), dtype=tf.float32),
        # tf.TensorSpec(shape=(667, 667, 1), dtype=tf.float32)
        tf.TensorSpec(shape=(672, 672, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(672, 672, 1), dtype=tf.float32)
    )
# )
).batch(8).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator_from_triplets(val_triplets, target_slice_size_meters),
    output_signature=(
        # tf.TensorSpec(shape=(667, 667, 4), dtype=tf.float32),
        # tf.TensorSpec(shape=(667, 667, 1), dtype=tf.float32)
        tf.TensorSpec(shape=(672, 672, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(672, 672, 1), dtype=tf.float32)
    )
# )
).batch(8).prefetch(tf.data.AUTOTUNE)


# # Configure datasets
# train_dataset = train_dataset.shuffle(buffer_size=1000)
# train_dataset = train_dataset.batch(batch_size)
# train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
#
# val_dataset = val_dataset.batch(batch_size)
# val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)




### --- --- --- VGG --- --- ---

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, UpSampling2D, Concatenate,
                                     BatchNormalization, Activation, Cropping2D)
from tensorflow.keras.applications import VGG19  # or VGG16




def build_vgg_unet_5channels(
    vgg_weights_path: str,
    input_shape=(400,400,5)
):
    """
    Creates a U-Net with a VGG19 encoder. The top-level input has shape (400,400,5),
    but only 3 out of those 5 channels get passed to VGG19. The rest are effectively
    ignored by the encoder.

    Steps:
      1) Input of shape (400,400,5).
      2) Slice out the first 3 channels -> shape (400,400,3).
      3) Pass that to a standard VGG19 (include_top=False).
      4) Grab skip outputs from the blocks.
      5) Build a U-Net style decoder to upsample + skip-conn.
      6) Output is a 1-channel segmentation mask.
    """

    ### (A) Top-level input: (400,400,5)
    inputs_5ch = Input(shape=input_shape, name="input_5ch")

    ### (B) Slice out 3 channels
    # For example, channels [0..2] in the last dimension:
    three_ch = Lambda(
        lambda x: x[:,:,:,0:3],
        name="slice_3ch"
    )(inputs_5ch)

    ### (C) Create a plain VGG19 (no top), with shape = (None, None, 3).
    # We do *not* pass input_tensor here. We'll handle skip outputs manually.
    vgg_base = VGG19(
        weights=None,
        include_top=False,
        input_shape=(None, None, 3)  # dynamically sized
    )
    # Load the pretrained "notop" weights from a local .h5
    vgg_base.load_weights(vgg_weights_path)

    ### Identify skip layers by name:
    #  block1_conv2 => ~ (None, H/2,  W/2,   64)
    #  block2_conv2 => ~ (None, H/4,  W/4,  128)
    #  block3_conv3 => ~ (None, H/8,  W/8,  256)
    #  block4_conv3 => ~ (None, H/16, W/16, 512)
    #  block5_conv3 => ~ (None, H/32, W/32, 512)
    s1 = vgg_base.get_layer('block1_conv2').output
    s2 = vgg_base.get_layer('block2_conv2').output
    s3 = vgg_base.get_layer('block3_conv3').output
    s4 = vgg_base.get_layer('block4_conv3').output
    encoder_out = vgg_base.get_layer('block5_conv3').output

    ### Build an encoder sub-model to get skip outputs:
    encoder_sub = Model(
        inputs=vgg_base.input,   # shape(None,None,3)
        outputs=[s1, s2, s3, s4, encoder_out],
        name='vgg19_encoder_sub'
    )

    ### Actually run the 3-ch tensor through that sub-model:
    skip1, skip2, skip3, skip4, enc_out = encoder_sub(three_ch)

    ### (D) U-Net decoder
    # enc_out shape ~ (None,H/32,W/32,512)
    x = enc_out

    # Step 1 => skip4
    x = UpSampling2D((2,2))(x)  # H/32 -> H/16
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, skip4])

    # Step 2 => skip3
    x = UpSampling2D((2,2))(x)  # H/16->H/8
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, skip3])

    # Step 3 => skip2
    x = UpSampling2D((2,2))(x)  # H/8->H/4
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, skip2])

    # Step 4 => skip1
    x = UpSampling2D((2,2))(x)  # H/4->H/2
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, skip1])

    # # final upsample from H/2 -> H
    # x = UpSampling2D((2,2))(x)
    # x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)

    # final mask => 1 channel, sigmoid
    outputs = Conv2D(1, (1,1), padding='same', activation='sigmoid')(x)

    ### Build the top-level:
    #   input = (400,400,5), output = (400,400,1)
    model_5ch = Model(inputs=inputs_5ch, outputs=outputs, name="vgg19_unet_5ch")
    return model_5ch



import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Build the model
# vgg_unet_5ch = build_vgg_unet_5channels(input_shape=(400,400,5))
# vgg_unet_5ch = build_vgg_unet(input_shape=(256,256,3))
model = build_vgg_unet_5channels(
        vgg_weights_path=vgg19_path,
        # input_shape=(667,667,4)
        input_shape=(672,672,4)  # with padding
    )

# # Freeze some layers if you only want to train decoder
# # (for example, freeze all VGG layers)
# for layer in vgg_unet_5ch.layers:
#     if 'block' in layer.name:  # or use another heuristic
#         layer.trainable = False

model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

# Create callbacks
checkpoint = ModelCheckpoint('PLANET_shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_672px_PSHP_b0_b3_v1_ADAPT.tf',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

# checkpoint = ModelCheckpoint('shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)



# Train
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=300,
    callbacks=[early_stopping, checkpoint]
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
                            rasterize_shapefile_to_binary, DinoV2DPTSegModel
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

image_test = load_image(tif_path)

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(image_test[3,:,:]))
plt.show(block=True)


plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(predictions))
plt.show(block=True)





# Calculate thresholds for each image
thresholds_accumulated = []
method = 'MEAN_p80'

pred_flat = predictions.flatten()
pred_flat = pred_flat[~np.isnan(pred_flat)]  # Remove NaN values
if method == 'MEAN_otsu':   # method='MEAN_otsu'
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
else:
    thresholds_accumulated.append(np.nan)
threshold_mean = np.mean(thresholds_accumulated)
combined_predictionimage_MEAN_p70 = predictions > threshold_mean




plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(combined_predictionimage_MEAN_p70))
plt.show(block=True)






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

def warp_planet_in_memory(
    in_path,
    out_crs="ESRI:102001",
    out_width=667,
    out_height=667,
    half_size=1000.0
):
    """
    1) Reads bounding box + crs from 'in_path'.
    2) center => transform to out_crs => define (centerX ± half_size, centerY ± half_size).
    3) reproject the input to that exact bounding box & size => array shape (bands, out_height, out_width).
    4) Return the in-memory array + an updated transform.

    We do NOT write any intermediate file to disk.
    """
    with rasterio.open(in_path) as src:
        src_crs = src.crs
        left, bottom, right, top = src.bounds
        # center in native CRS
        cx = 0.5*(left + right)
        cy = 0.5*(bottom + top)
        # transform center => out_crs
        cx_alb, cy_alb = transform_point(cx, cy, src_crs, out_crs)

        # define bounding box => 2km box => center ± half_size
        xMin = cx_alb - half_size
        xMax = cx_alb + half_size
        yMin = cy_alb - half_size
        yMax = cy_alb + half_size

        # We want an in-memory reproject at exactly out_width x out_height.
        # We'll define an affine transform manually for that bounding box + size.
        from affine import Affine
        # Suppose each pixel covers (xMax - xMin)/out_width in X, similarly in Y
        pixel_size_x = (xMax - xMin) / out_width
        pixel_size_y = (yMin - yMax) / out_height  # note yMin < yMax => negative height
        # The top-left corner transform => for row,col => x = xMin + col*px, y = yMax + row*py
        # in typical georaster coords, 'top-left' => (xMin, yMax)
        # but we have to remember that Y is top->bottom in raster, so the transform typically is:
        transform = Affine(pixel_size_x, 0, xMin,
                           0, pixel_size_y, yMax)

        # Prepare an output array in memory
        band_count = src.count
        src_dtype = src.dtypes[0]
        # Let's read only the first 4 bands if you have 5 and you want 4
        # or check band_count > 4 => we keep 4
        keep_bands = min(band_count, 4)
        out_arr = np.zeros((keep_bands, out_height, out_width), dtype=src_dtype)

        # reproject each band
        for i in range(keep_bands):
            reproject(
                source=rasterio.band(src, i+1),
                destination=out_arr[i],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=out_crs,
                resampling=Resampling.bilinear if i < 3 else Resampling.bilinear
                # or "mode" if you have discrete data in band 4
            )
        # out_arr => shape (4, 667, 667) typically
        return out_arr, transform

def inference_planet_tif(
    in_path,
    model,
    half_size=1000,
    pad_to_multiple_of_32=True,
    scale_factor=5000.0
):
    """
    1) warp the Planet file in memory to 2km => 667x667
    2) if pad_to_multiple_of_32 => e.g. pad from 667->672
    3) scale => arr / scale_factor
    4) model.predict => shape => (1, newH, newW, 1)
    5) unpad => final HxW
    returns preds => shape (H, W)
    """
    # 1) warp in memory => (bands, 667, 667)
    out_arr, transform = warp_planet_in_memory(
        in_path,
        out_crs="ESRI:102001",
        out_width=667,
        out_height=667,
        half_size=half_size
    )  # shape => (4,667,667) if keep 4 bands

    # 2) move axis => (H, W, channels)
    out_arr = np.moveaxis(out_arr, 0, -1)  # => (667,667,4)
    H, W, C = out_arr.shape

    # Compute NDVI and water mask
    red = out_arr[..., 2].astype(np.float32) / scale_factor  # Red band (adjust index if needed)
    nir = out_arr[..., 3].astype(np.float32) / scale_factor  # NIR band (adjust index if needed)
    ndvi = (nir - red) / (nir + red + 1e-8)  # Add epsilon to avoid division by zero
    water_mask = (ndvi < 0.0).astype(int)  # Water where NDVI < 0

    # optional pad to multiple-of-32 => 667->672
    padH = (32 - (H % 32)) % 32 if pad_to_multiple_of_32 else 0
    padW = (32 - (W % 32)) % 32 if pad_to_multiple_of_32 else 0
    arr_padded = np.pad(out_arr, ((0,padH),(0,padW),(0,0)), mode='constant')

    # 3) scale
    arr_norm = arr_padded / scale_factor
    # expand dims => (1,newH,newW,4)
    arr_norm = np.expand_dims(arr_norm, axis=0)

    # 4) run inference
    # preds = model.predict(arr_norm)  # => (1,newH,newW,1)
    preds = model.predict(arr_norm)[0, ..., 0]  # => (newH, newW)

    # preds = preds[0]  # => (newH,newW,1)
    # preds = preds[...,0] # => (newH,newW)

    # 5) unpad => (H,W)
    preds = preds[:H, :W]
    # print(preds.shape)

    # Apply water mask to predictions
    preds[water_mask == 1] = 0.0  # Set water areas to 0.0

    return preds, ndvi

# ms1a1
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
preds, ndvi = inference_planet_tif(tif_path, model)
# preds shape => (667,667)



plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(preds))
plt.show(block=True)

#
# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(ndvi))
# plt.show(block=True)



# Calculate thresholds for each image
thresholds_accumulated = []
method = 'MEAN_yen'

pred_flat = preds.flatten()
pred_flat = pred_flat[~np.isnan(pred_flat)]  # Remove NaN values
thresholds_accumulated.append(threshold_yen(pred_flat))
threshold_mean = np.mean(thresholds_accumulated)
combined_predictionimage_MEAN_p70 = preds > threshold_mean

plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(combined_predictionimage_MEAN_p70))
plt.show(block=True)





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



