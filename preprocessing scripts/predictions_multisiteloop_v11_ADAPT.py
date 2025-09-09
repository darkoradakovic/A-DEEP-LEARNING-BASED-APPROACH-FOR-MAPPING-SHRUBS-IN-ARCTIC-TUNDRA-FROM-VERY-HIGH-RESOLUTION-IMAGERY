
## predictions_multisiteloop_v10_ADAPT.py
# version 07 feb 2025

import re
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from rasterio.windows import Window
from skimage.transform import resize
import numpy as np
import rasterio
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm                    # for the DINOv2-based backbone if available here
# or from your local library if you have a custom 'DINOv2' code
import safetensors.torch       # if your weights are in .safetensors

## BRIGHTNESS REMOVAL MODELS
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from functionsdarko import (slice_image, load_sliced_images_and_metadata, load_sliced_REF_PCA_images,
    process_and_store_predictions_with_validation4,
    process_validation4,process_and_store_predictions_with_validation5, process_images_evi5,
    process_and_store_predictions_with_validation44_4, process_and_store_predictions_with_validation44_6, process_and_store_predictions_with_validation44_4cloud,
    cnn_segmentation_model, upgraded_cnn_comb_model, f1_score, calculate_ndvi, apply_clustering, process_and_cluster_images,
    make_prediction,dynamic_threshold_adjustment,evaluate_thresholds_for_year2,remove_edge_lines, calculate_metrics44_2,collect_image_paths2,
    specify_cloud_areas,cloud_check, predict_clouds, filter_slices, filter_slices_v2, combine_tiles_to_large_image, save_combined_image,
    load_sliced_PSHP_P1BS_images, predictions_pshp_p1bs, combine_tiles_to_large_image_predictionsoneyear2,
    evaluate_thresholds_for_year3_combined_image, predictions_vit_p1bs, multisite_configurations_ADAPT,
    DinoV2DPTSegModel,
    combine_tiles_to_large_image_overlap_preserve_edges, load_sliced_REF_PCA_images_with_overlap, load_sliced_PSHP_P1BS_images_overlap
    )




# ### ---- ---- SHRUB MODELS ---- ---- ###

# # # shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14.tf
# model = load_model(
#     '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14.tf'
# )

# # resnet pretrained v4
# model = load_model(
#       '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6.tf'
# )

# # VGG19 pretrained model v1
# model = load_model(
#       '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19.tf'
# )

# # UNET256fil
# model = load_model(
#        '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20.tf'
# )


# ### DINOv2 model ## VIT models DINOv2 + DPT head trained on 48 hours ADAPT
# ### VIT Dino+DPT model
# model_name = "vit_large_patch14_dinov2.lvd142m"  # or your DINOv2 large/huge
# # local_weights_path = "/path/to/dinov2_vitl14.safetensors"  # optional
# model_dinodpt = DinoV2DPTSegModel(
#     backbone_name=model_name,
#     out_channels=1,
#     slice_3ch=True,
#     # pretrained_weights_path=local_weights_path
# )
#
# checkpoint_path = "/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h.pth"
# # checkpoint_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h.pth"
# checkpoint = torch.load(checkpoint_path, map_location="cpu")
# model_dinodpt.load_state_dict(checkpoint)


# ### ---- ---- WETTUNDRA MODELS ---- ---- ###

# # MAIN MODELS CNN, UNET, VGG, RESNET, VIT 2025
# modelbasedir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/'
#
# # 512fil CNN ep14
# modelname = 'wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18.tf'
# model = load_model(os.path.join(modelbasedir,modelname), compile=False)
#
# # Resnet50 pretrained with UNET decoder, using band 0:3
# modelname = 'wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21.tf'
# model = load_model(os.path.join(modelbasedir,modelname))
#
# # VGG19 pretrained with UNET decoder, using band 0:3
# modelname = 'wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21.tf'
# model = load_model(os.path.join(modelbasedir,modelname))
#
# ## UNET MODEL 256fil 21-1-2025
# modelname = 'wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4.tf'
# model = load_model(os.path.join(modelbasedir,modelname))
#
### DINOv2 model ## VIT models DINOv2 + DPT head trained on 48 hours ADAPT
### VIT Dino+DPT model
# model_name = "vit_large_patch14_dinov2.lvd142m"  # or your DINOv2 large/huge
# # local_weights_path = "/path/to/dinov2_vitl14.safetensors"  # optional
# model_dinodpt = DinoV2DPTSegModel(
#     backbone_name=model_name,
#     out_channels=1,
#     slice_3ch=True,
#     # pretrained_weights_path=local_weights_path
# )
# checkpoint_path = "/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48hours.pth"
# # checkpoint_path = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h.pth"
# checkpoint = torch.load(checkpoint_path, map_location="cpu")
# model_dinodpt.load_state_dict(checkpoint, strict=False)






# # ### ---- ---- LAKES MODELS ---- ---- ###
#
# MAIN MODELS CNN, UNET, VGG, RESNET, VIT 2025
modelbasedir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/'

# 512fil CNN ep14
# modelname = 'lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24.tf'
# model = load_model(os.path.join(modelbasedir,modelname), compile=False)
#
# # Resnet50 pretrained with UNET decoder, using band 0:3
# modelname = 'lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep22.tf'
# model = load_model(os.path.join(modelbasedir,modelname))
#
# # VGG19 pretrained with UNET decoder, using band 0:3
# modelname = 'lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10.tf'
# model = load_model(os.path.join(modelbasedir,modelname))
#
# # ## UNET MODEL 256fil 21-1-2025
# modelname = 'lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24.tf'
# model = load_model(os.path.join(modelbasedir,modelname))
#
## DINOv2 model ## VIT models DINOv2 + DPT head trained on 48 hours ADAPT
## VIT Dino+DPT model
model_name = "vit_large_patch14_dinov2.lvd142m"  # or your DINOv2 large/huge
# local_weights_path = "/path/to/dinov2_vitl14.safetensors"  # optional
model_dinodpt = DinoV2DPTSegModel(
    backbone_name=model_name,
    out_channels=1,
    slice_3ch=True,
    # pretrained_weights_path=local_weights_path
)
checkpoint_path = "/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/lakesrivers_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_16ep.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model_dinodpt.load_state_dict(checkpoint, strict=False)




# Directory to save the SHRUB predictions
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_watermask'
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6_watermask'
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19_watermask'
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20_watermask'
# # save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h_watermask'
# os.makedirs(save_dir, exist_ok=True)


# Directory to save the WETTUNDRA predictions
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18'
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21'
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21'
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4'
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48hours.pth'
# os.makedirs(save_dir, exist_ok=True)


# Directory to save the LAKES predictions
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24'
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep22'
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10'
# save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24'
save_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/predictions/lakesrivers_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_16ep'
os.makedirs(save_dir, exist_ok=True)


# Iterate over multisite configurations and process each one
for ms, config in multisite_configurations_ADAPT.items():

    # # # ------>------->------>
    # #                         # <---------- <--------- <--------
    # # # ------>------->------>
    # #                         #  <--------- <--------
    # #         # ------>------->
    # if ms in ('ms4'):
    # # ## _______________________________REMOVE________________

        print('ms: ', ms)
        base_path = config['base_path']
        excluded_timeseries = config['excluded_timeseries']
        pattern_prefix = config['pattern_prefix']
        areas_with_clouds = config['areas_with_clouds']
        filenames = config['filenames']
        solar_zenith_angles = config['solar_zenith_angles']

        # Collect image paths and cloud information
        pshp_paths2, ndvi_paths2, areas, p1bs_paths2 = collect_image_paths2(base_path, excluded_timeseries, pattern_prefix=pattern_prefix)
        cloud_info = specify_cloud_areas(pattern_prefix, areas_with_clouds)

        # Initialize storage for predictions
        predictions_complete = {}
        metadata_dict2_complete = {}

        for area, year_paths in pshp_paths2.items():
            print('area: ', area)
            predictions_complete[area] = {}
            slice_stack_pshp = []
            slice_stack_ndvi = []
            slice_stack_p1bs = []
            for (year, pathpsph), (_, pathndvi), (_, pathp1bs),  in zip(year_paths.items(), ndvi_paths2[area].items(), p1bs_paths2[area].items()):
                slice_stack_pshp.append(pathpsph[0])
                slice_stack_ndvi.append(pathndvi[0])
                slice_stack_p1bs.append(pathp1bs[0])

            # imgid = 'PCA'
            imgid = 'REF'
            #  # LOAD PSHP and P1BS nov 2024
            # sliced_ms_images_RAD, sliced_ms_images_REFNDVI, sliced_ms_images_P1BS, metadata_dict = load_sliced_PSHP_P1BS_images(
            #     slice_stack_pshp, target_slice_size_meters=200, resize_to=(400, 400))

            # LOAD PSHP and P1BS feb 2025
            # LOAD PSHP and P1BS feb 2025
            sliced_ms_images_RAD, sliced_ms_images_REFNDVI, sliced_ms_images_P1BS, metadata_dict = load_sliced_PSHP_P1BS_images_overlap(
                slice_stack_pshp, target_slice_size_meters=200, resize_to=(400, 400))

            # Filter: preserve original image shape by replacing any “empty” slice with a placeholder array, TESTED, works with Predictions
            sliced_ms_images_RAD_filtered, sliced_ms_images_P1BS_filtered, _, sliced_ms_images_REFNDVI_filtered, _, nonvalid_index2 = filter_slices_v2(
                sliced_ms_images_RAD, sliced_ms_images_P1BS, sliced_ms_images_P1BS, sliced_ms_images_REFNDVI,
                metadata_dict,
                imgid, threshold=0.6)

            # # PSHP P1BS predictions with normalization --Nov 2024 for UNET, CNN, VGG, RESNET
            # predictions_dict, water_dict2, metadata_dict2 = predictions_pshp_p1bs(
            #     sliced_ms_images_RAD_filtered, sliced_ms_images_REFNDVI_filtered, sliced_ms_images_P1BS_filtered,
            #     cloud_info, area, model, metadata_dict)

            # For VIT models 2025
            predictions_dict, water_dict2, metadata_dict2 = predictions_vit_p1bs(
               sliced_ms_images_RAD_filtered, sliced_ms_images_REFNDVI_filtered, sliced_ms_images_P1BS_filtered,
               cloud_info, area, model_dinodpt, metadata_dict)

            # # Mask Out Water in Predictions
            # for year in predictions_dict:
            #     for i in range(len(predictions_dict[year])):
            #         shrub_prob = predictions_dict[year][i]  # shape (H, W, 1)
            #         water_mask = water_dict2[year][i]  # shape (H, W) with 1 for water
            #
            #         # water_mask_expanded = np.expand_dims(water_mask, axis=-1)  # Shape: (H, W, 1)
            #         # Force shrub probability to zero wherever water_mask == 1
            #         # Since shrub_prob is (H, W, 1), we select the last dimension [..., 0].
            #         # shrub_prob[water_mask == 1, 0] = 0.0
            #         shrub_prob[water_mask == 1] = 0.0  # No need for [..., 0]

            # shrub_cover_per_year = {}
            # predictions_dict_binary = {}
            # for year in predictions_dict:
            #     shrub_cover_per_year[year], predictions_dict_binary[year] = evaluate_thresholds_for_year2 \
            #         (predictions_dict[year], 'MEAN_p70')
            #
            # metrics_per_year, shrub_cover_per_year, shrub_cover_per_year2 = calculate_metrics44_2(predictions_dict_binary)
            #
            # for year, metric in metrics_per_year.items():
            #     print(f"{year}  {metric:.2f}")

            # SAVING
            predictions_complete[area] = predictions_dict
            metadata_dict2_complete[area] = metadata_dict2

        # Save the predictions for the current multisite
        # with open(os.path.join(save_dir, f'shrub_predictions_complete_{ms}.pkl'), 'wb') as f:
        # with open(os.path.join(save_dir, f'wettundra_predictions_complete_{ms}.pkl'), 'wb') as f:
        with open(os.path.join(save_dir, f'lakesrivers_predictions_complete_{ms}.pkl'), 'wb') as f:
            pickle.dump(predictions_complete, f)
        # with open(os.path.join(save_dir, f'shrub_meta_complete_{ms}.pkl'), 'wb') as f:
        # with open(os.path.join(save_dir, f'wettundra_meta_complete_{ms}.pkl'), 'wb') as f:
        with open(os.path.join(save_dir, f'lakesrivers_meta_complete_{ms}.pkl'), 'wb') as f:
            pickle.dump(metadata_dict2_complete, f)








