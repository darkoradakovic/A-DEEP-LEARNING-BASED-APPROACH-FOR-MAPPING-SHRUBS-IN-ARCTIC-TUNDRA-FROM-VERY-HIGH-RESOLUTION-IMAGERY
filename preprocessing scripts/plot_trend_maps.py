

# LOOP CREATE FIGURES FOR MULTIPLE MODELS AND SITES/AREAS

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from tensorflow.keras.models import load_model
from skimage.transform import resize
import torch
import torch.nn.functional as F
from functionsdarko import (slice_overlap_simple, combine_tiles_to_large_image_predictionsoneyear, DinoV2DPTSegModel,
    combine_tiles_to_large_image_overlap_preserve_edges)

# 1) where your models live
# MODEL_DIR = "/Volumes/OWC Express 1M2/nasa_above/models"
# MODEL_DIR = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/"
MODEL_DIR = "/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2//models"
#shrub
# OUTPUT_DIR = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 2/trend_maps"
#wet tundra
# OUTPUT_DIR = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 2/trend_maps"
OUTPUT_DIR = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/figures/trend_maps'
#lakes
# OUTPUT_DIR = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/qgis/model predictions other sites LAKES'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 2) models shrub
# MODELS = [
#     { "name":"CNN512",   "type":"tf",  "file":"shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14.tf" },
#     { "name":"ResNet50", "type":"tf",  "file":"shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6.tf" },
#     { "name":"VGG19",    "type":"tf",  "file":"shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19.tf" },
#     { "name":"UNET256",  "type":"tf",  "file":"shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20.tf" },
#     { "name":"VIT",      "type":"vit", "file":"shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h.pth" }
# ]

## 2) models wet tundra
# MODELS = [
#     { "name":"CNN512",   "type":"tf",  "file":"wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18.tf" },
#     { "name":"ResNet50", "type":"tf",  "file":"wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21.tf" },
#     { "name":"VGG19",    "type":"tf",  "file":"wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21.tf" },
#     { "name":"UNET256",  "type":"tf",  "file":"wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4.tf" },
#     { "name":"VIT",      "type":"vit", "file":"wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5_ep7.pth" }
# ]
# # 2) models lakes
MODELS = [
    { "name":"CNN512",   "type":"tf",  "file":"lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24.tf" },
    { "name":"ResNet50", "type":"tf",  "file":"lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep11.tf" },
    { "name":"VGG19",    "type":"tf",  "file":"lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10.tf" },
    { "name":"UNET256",  "type":"tf",  "file":"lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24.tf" },
    { "name":"VIT",      "type":"vit", "file":"lakes_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_localCPU_v2.pth" }
]

# # 3) images
# PSHP_PATHS = [
#     ## ms6a2 LAPTOP NoN = Georef
#     # "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m.tif",
#     '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2_GEOREF_clip1950m.tif',
#     '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2_GEOREF_clip1950m.tif',
#     '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m.tif',
#     '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m.tif'
# ]
# ## OWC Express
# PSHP_PATHS = [
#     # ## ms6a1 OWC Express paths NoN=Georef
#     # '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_1_240323_215047/QB02_20090718220421_1010010009F45F00_P001_area1_-1948141_4014035/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area1.tif',
#     # '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area1_-1948141_4014035/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area1.tif',
#     # '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area1_-1948141_4014035/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area1.tif',
#     # '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area1_-1948141_4014035/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area1.tif'
#     ## ms6a2 OWC Express paths NoN=Georef
#     '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_1_240323_215047/QB02_20090718220421_1010010009F45F00_P001_area2_-1948141_4012041/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2.tif',
#     '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area2_-1948141_4012041/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'
#     '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area2_-1948141_4012041/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif',
#     '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area2_-1948141_4012041/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'
# ]
# PSHP_PATHS = [
#     ## ms6a2 LAPTOP NoN = Georef
#     {'name':'ms6a2', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/multisites/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_4_180423_222328/WV02_20110625223400_103001000B99EE00_P006_area2_-1948141_4012041/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2.tif'},
#     {'name':'ms6a2', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/multisites/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area2_-1948141_4012041/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'},
#     {'name':'ms6a2', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/multisites/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area2_-1948141_4012041/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif'},
#     {'name':'ms6a2', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/multisites/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area2_-1948141_4012041/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'}
# ]
## ADAPT
PSHP_PATHS = [
    # '/explore/nobackup/people/dradako1/sites/timeseries_multisite1/'
    #ms1a1
    # {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_1_10323_153130/QB02_20050724225626_1010010004654800_P001_area1_-2386945_4406130/QB02_20050724225626_1010010004654800_PSHP_P001_NT_2000m_area1.tif'},
    {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_4_110323_154523/WV02_20130820230414_1030010025076500_P003_area1_-2386945_4406130/WV02_20130820230414_1030010025076500_PSHP_P003_NT_2000m_area1.tif'},
    {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_7_110323_155000/WV03_20170813230621_1040010031CAA600_P003_area1_-2386945_4406130/WV03_20170813230621_1040010031CAA600_PSHP_P003_NT_2000m_area1.tif'},
    {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_9_110323_155547/WV02_20200716223511_10300100A97DD900_P003_area1_-2386945_4406130/WV02_20200716223511_10300100A97DD900_PSHP_P003_NT_2000m_area1.tif'},
    #ms4a1
    {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_1_240323_202958/QB02_20020729211432_1010010000E40D00_P001_area1_-1746242_3978542/QB02_20020729211432_1010010000E40D00_PSHP_P001_NT_2000m_area1.tif'},
    {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_4_240323_203154/QB02_20100701212531_101001000BE81F00_P002_area1_-1746242_3978542/QB02_20100701212531_101001000BE81F00_PSHP_P002_NT_2000m_area1.tif'},
    {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_6_240323_203526/WV02_20130710213601_1030010024B9FD00_P007_area1_-1746242_3978542/WV02_20130710213601_1030010024B9FD00_PSHP_P007_NT_2000m_area1.tif'},
    {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_13_240323_204307/WV02_20180826223220_103001008265EA00_P006_area1_-1746242_3978542/WV02_20180826223220_103001008265EA00_PSHP_P006_NT_2000m_area1.tif'},
    #ms10a1
    {'name':'ms10a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite10/timeseries_multisite10_1_240323_221916/QB02_20050724225630_1010010004654800_P002_area1_-2399591_4394681/QB02_20050724225630_1010010004654800_PSHP_P002_NT_2000m_area1.tif'},
    {'name':'ms10a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite10/timeseries_multisite10_2_260423_125313/WV02_20140707224108_1030010033485600_P006_area1_-2399591_4394681/WV02_20140707224108_1030010033485600_PSHP_P006_NT_2000m_area1.tif'},
    {'name':'ms10a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite10/timeseries_multisite10_4_240323_222216/WV03_20170813230622_1040010031CAA600_P004_area1_-2399591_4394681/WV03_20170813230622_1040010031CAA600_PSHP_P004_NT_2000m_area1.tif'},
    # ms6a1 OWC Express paths NoN=Georef
    {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_1_240323_215047/QB02_20090718220421_1010010009F45F00_P001_area1_-1948141_4014035/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area1.tif'},
    # {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_4_180423_222328
    # {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area1_-1948141_4014035/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area1.tif'}, #cloudy
    {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area1_-1948141_4014035/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area1.tif'},
    # {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area1_-1948141_4014035/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area1.tif'},  # missing part
    # ms6a2 OWC Express paths NoN=Georef
    # {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_1_240323_215047/QB02_20090718220421_1010010009F45F00_P001_area2_-1948141_4012041/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2.tif'},
    {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_4_180423_222328/WV02_20110625223400_103001000B99EE00_P006_area2_-1948141_4012041/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2.tif'},
    # {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area2_-1948141_4012041/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'},
    {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area2_-1948141_4012041/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif'},
    {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area2_-1948141_4012041/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'},
]


areas = sorted({entry['name'] for entry in PSHP_PATHS})



years = []
predictions = {}
for model_info in MODELS:
    # load model
    if model_info["type"] == "tf":
        mdl = load_model(os.path.join(MODEL_DIR, model_info["file"]), compile=False)
    else:
        # Dino+DPT
        mdl = DinoV2DPTSegModel(
            backbone_name="vit_large_patch14_dinov2.lvd142m",
            out_channels=1, slice_3ch=True,
            pretrained_weights_path=None
        )
        ckpt = torch.load(os.path.join(MODEL_DIR, model_info["file"]), map_location="cpu")
        # if your checkpoint is just state_dict:
        if "model_state_dict" in ckpt:
            mdl.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            mdl.load_state_dict(ckpt, strict=False)
        mdl.eval()

    for area in areas:
        predictions = {}
        print(f"  Area = {area}")

        # 2) select just the PSHP paths belonging to this area
        area_entries = [e for e in PSHP_PATHS if e['name'] == area]

        for pshp_info in area_entries:
            pshp = pshp_info['file']
            # print(pshp)
            p1bs = pshp.replace("PSHP","P1BS")  # --for using TIMESERIES FOLDER
            # p1bs = pshp.replace("x_train", "x_train2_remove_brightness/P1BS").replace("PSHP","P1BS") # --FOR GEOREF FOLDER ms6
            # p1bs = p1bs.replace('ms6_timeseries_P1BS_as_TIFF', 'ms6_timeseries_PSHP_as_TIFF') # --FOR GEOREF FOLDER ms6

            # slice
            tiles_pshp, metas = slice_overlap_simple(pshp, 200, overlap=0.25, resize_to=(400,400))
            tiles_p1bs, _    = slice_overlap_simple(p1bs,200, overlap=0.25, resize_to=(400,400))
            X = np.concatenate([np.stack(tiles_pshp), np.stack(tiles_p1bs)], axis=-1).astype("float32")/255.0

            filename = os.path.basename(pshp)
            # sensor = filename[:4]
            year = filename.split('_')[1][:4]
            # identifier = f"{sensor}_{year}"
            print(year)
            years.append(int(year))
            # predict
            if model_info["type"] == "tf":
                preds = []
                for batch in np.array_split(X, 10):
                    preds.append(mdl.predict(batch, verbose=0))
                preds = np.concatenate(preds, axis=0)
            else:
                preds = []
                with torch.no_grad():
                    for batch in np.array_split(X, 8):
                        bt = torch.from_numpy(np.moveaxis(batch, -1,1)).float()

                        # # apply ImageNet norm for VIT:
                        # ### ----------  ----------WET TUNDRA---------- ----------
                        # m0 = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
                        # s0 = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
                        # bt[:,:3] = (bt[:,:3]-m0)/s0
                        # ### ----------  ----------WET TUNDRA---------- ----------

                        out = mdl(bt)
                        p = torch.sigmoid(out).cpu().numpy()
                        preds.append(np.moveaxis(p,1,-1))
                preds = np.concatenate(preds, axis=0)

            # stitch
            # combined, _ = combine_tiles_to_large_image_predictionsoneyear(preds, metas)
            combined = combine_tiles_to_large_image_overlap_preserve_edges(preds, metas)

            #### ----------  ----------WATER MASK FOR SHRUBS ---------- ----------
            ndvi_path = pshp.replace("PSHP", "NDVI")  # for using TIMESERIES FOLDER
            # ndvi = pshp.replace("x_train", "x_train2_remove_brightness/NDVI").replace(".tif", "_NDVI.tif")
            # ndvi_path = ndvi.replace('ms6_timeseries_NDVI_as_TIFF', 'ms6_timeseries_PSHP_as_TIFF')
            # 1) load the full NDVI raster once
            with rasterio.open(ndvi_path) as src_ndvi:
                ndvi_full = src_ndvi.read(1)  # shape (H_ndvi, W_ndvi)
            # 2) resize it to match your combined’s shape (Hc, Wc)
            ndvi_resized = resize(
                ndvi_full,
                output_shape=combined.shape,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(ndvi_full.dtype)
            # 3) build a mask and zero out all pixels where NDVI < 0.05
            # mask = ndvi_resized < 0.05   ### ----------------------------------------> SHURBS
            mask = ndvi_resized > 0.05   ### ----------------------------------------> LAKES, remove land trends
            combined[mask] = 0.0
            #### ----------  ----------WATER MASK ---------- ----------


            if year not in predictions:
                predictions[year] = []

            predictions[year].append(combined)


        # assume predictions is { '2009':[map_2009], '2013':[map_2013], … }
        years = sorted(predictions.keys(), key=int)
        covers = []
        for y in years:
            img = predictions[y][0]        # each is a 2D array e.g. (H,W) or (H,W,1)
            if img.ndim==3 and img.shape[2]==1:
                img = img[...,0]
            # resize to 4000×4000, preserve data‐range
            img4k = resize(img,
                           output_shape=(4000,4000),
                           order=1,            # bilinear
                           preserve_range=True,
                           anti_aliasing=True).astype(img.dtype)
            covers.append(img4k)

        # 3) Stack into a 3D array of shape (4, H, W):
        stack = np.stack(covers, axis=0)  # shape = (n_years, H, W)
        y_mean = stack.mean(axis=0)                 # shape (H,W)
        years_cal = np.array(years, dtype=float)  # ms6
        # years_cal = np.array([2009, 2013, 2016, 2017], dtype=float)  # ms6
        # years_cal = np.array([2011, 2013, 2016, 2017], dtype=float)  # ms6 ------------->>>> NO 2009
        years2 = years_cal - years_cal[0]
        yr_mean = years2.mean()
        cov = (years2[:,None,None] * stack).mean(axis=0) - yr_mean * y_mean
        var = (years2**2).mean() - yr_mean**2
        slope_map = cov / var                       # shape (H,W)
        slope_map_pct = slope_map * 100.0   # now in % per year

        pct_min, pct_max = np.percentile(slope_map_pct, [1, 99])


        plt.figure(figsize=(8,8))
        im = plt.imshow(
            slope_map_pct,
            cmap="RdBu",
            vmin=-2,        # e.g. ±2 % yr⁻¹
            vmax= 2)
        # im = plt.imshow(slope_map_pct, cmap="RdBu", vmin=pct_min, vmax=pct_max)  # Clip percentile stretch, preserves dynamic range
        cbar = plt.colorbar(im, fraction=0.036, pad=0.04)
        cbar.set_label("Cover Change (% per year)", fontsize=20, fontweight="bold")
        cbar.ax.tick_params(labelsize=20)
        # plt.title("Per‐pixel trend in cover (2009→2017)", fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        ## SAVE
        # outfn = f"{model_info['name']}_TRENDMAP_{pshp_info['name']}_{os.path.basename(pshp).replace('.tif', '_SHRUBS.jpg')}"
        # outfn = f"{model_info['name']}_TRENDMAP_{pshp_info['name']}_{os.path.basename(pshp).replace('.tif','_WETTUNDRA.jpg')}"
        outfn = f"{model_info['name']}_TRENDMAP_{pshp_info['name']}_{os.path.basename(pshp).replace('.tif', '_LAKES.jpg')}"
        # plt.savefig(os.path.join(OUTPUT_DIR,outfn), dpi=200, bbox_inches="tight", pad_inches=0.1)
        # plt.close()
        # print(outfn)
        plt.show()

        # predictions = {}
        years =[]








### OLD version LOOP CREATE TREND MAPS FROM MULTIPLE AREAS
# June 10 2025

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from tensorflow.keras.models import load_model
from skimage.transform import resize
import torch
import torch.nn.functional as F
from functionsdarko import (slice_overlap_simple, combine_tiles_to_large_image_predictionsoneyear, DinoV2DPTSegModel,
    combine_tiles_to_large_image_overlap_preserve_edges)

# 1) where your models live
MODEL_DIR = "/Volumes/OWC Express 1M2/nasa_above/models"
# MODEL_DIR = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/"
#wet tundra
OUTPUT_DIR = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/trend_maps'
#lakes
OUTPUT_DIR = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/trend_maps'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2) models shrub
MODELS = [
    # { "name":"CNN512",   "type":"tf",  "file":"shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14.tf" },
    { "name":"ResNet50", "type":"tf",  "file":"shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6.tf" },
    # { "name":"VGG19",    "type":"tf",  "file":"shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19.tf" },
    # { "name":"UNET256",  "type":"tf",  "file":"shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20.tf" },
    # { "name":"VIT",      "type":"vit", "file":"shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h.pth" }
]

# 2) models wet tundra
MODELS = [
    # { "name":"CNN512",   "type":"tf",  "file":"wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18.tf" },
    { "name":"ResNet50", "type":"tf",  "file":"wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21.tf" },
    # { "name":"VGG19",    "type":"tf",  "file":"wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21.tf" },
    # { "name":"UNET256",  "type":"tf",  "file":"wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4.tf" },
    # { "name":"VIT",      "type":"vit", "file":"wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5.pth" }
]

# 3) images
PSHP_PATHS = [
    ## ms6a2 LAPTOP NoN = Georef
    "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m.tif",
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2_GEOREF_clip1950m.tif',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m.tif',
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m.tif'
]

PSHP_PATHS = [
    # ## ms6a1 OWC Express paths NoN=Georef
    # '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_1_240323_215047/QB02_20090718220421_1010010009F45F00_P001_area1_-1948141_4014035/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area1.tif',
    # '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area1_-1948141_4014035/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area1.tif',
    # '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area1_-1948141_4014035/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area1.tif',
    # '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area1_-1948141_4014035/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area1.tif'
    ## ms6a2 OWC Express paths NoN=Georef
    '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_1_240323_215047/QB02_20090718220421_1010010009F45F00_P001_area2_-1948141_4012041/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2.tif',
    '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area2_-1948141_4012041/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'
    '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area2_-1948141_4012041/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif',
    '/Volumes/OWC Express 1M2/nasa_above/TIMESERIES/timeseries_ms6/timeseries_multisite6_areas/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area2_-1948141_4012041/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'
]



predictions = {}
for model_info in MODELS:
    # load model
    if model_info["type"] == "tf":
        mdl = load_model(os.path.join(MODEL_DIR, model_info["file"]), compile=False)
    else:
        # Dino+DPT
        mdl = DinoV2DPTSegModel(
            backbone_name="vit_large_patch14_dinov2.lvd142m",
            out_channels=1, slice_3ch=True,
            pretrained_weights_path=None
        )
        ckpt = torch.load(os.path.join(MODEL_DIR, model_info["file"]), map_location="cpu")
        # if your checkpoint is just state_dict:
        if "model_state_dict" in ckpt:
            mdl.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            mdl.load_state_dict(ckpt, strict=False)
        mdl.eval()

    for pshp in PSHP_PATHS:
        # p1bs = pshp.replace("PSHP","P1BS")
        p1bs = pshp.replace("x_train", "x_train2_remove_brightness/P1BS").replace("PSHP","P1BS")
        p1bs = p1bs.replace('ms6_timeseries_P1BS_as_TIFF', 'ms6_timeseries_PSHP_as_TIFF')
        ndvi = pshp.replace("x_train", "x_train2_remove_brightness/NDVI").replace(".tif","_NDVI.tif")
        ndvi_path = ndvi.replace('ms6_timeseries_NDVI_as_TIFF', 'ms6_timeseries_PSHP_as_TIFF')

        # slice
        tiles_pshp, metas = slice_overlap_simple(pshp, 200, overlap=0.25, resize_to=(400,400))
        tiles_p1bs, _    = slice_overlap_simple(p1bs,200, overlap=0.25, resize_to=(400,400))

        X = np.concatenate([np.stack(tiles_pshp), np.stack(tiles_p1bs)], axis=-1).astype("float32")/255.0

        filename = os.path.basename(pshp)
        # sensor = filename[:4]
        year = filename.split('_')[1][:4]
        # identifier = f"{sensor}_{year}"
        print(year)
        # predict
        if model_info["type"] == "tf":
            preds = []
            for batch in np.array_split(X, 10):
                preds.append(mdl.predict(batch, verbose=0))
            preds = np.concatenate(preds, axis=0)
        else:
            preds = []
            with torch.no_grad():
                for batch in np.array_split(X, 8):
                    bt = torch.from_numpy(np.moveaxis(batch, -1,1)).float()

                    # # apply ImageNet norm for VIT:
                    # m0 = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
                    # s0 = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
                    # bt[:,:3] = (bt[:,:3]-m0)/s0

                    out = mdl(bt)
                    p = torch.sigmoid(out).cpu().numpy()
                    preds.append(np.moveaxis(p,1,-1))
            preds = np.concatenate(preds, axis=0)

        # stitch
        # combined, _ = combine_tiles_to_large_image_predictionsoneyear(preds, metas)
        combined = combine_tiles_to_large_image_overlap_preserve_edges(preds, metas)

        #### ----------  ----------WATER MASK ---------- ----------
        # 1) load the full NDVI raster once
        with rasterio.open(ndvi_path) as src_ndvi:
            ndvi_full = src_ndvi.read(1)  # shape (H_ndvi, W_ndvi)
        # 2) resize it to match your combined’s shape (Hc, Wc)
        ndvi_resized = resize(
            ndvi_full,
            output_shape=combined.shape,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(ndvi_full.dtype)
        # 3) build a mask and zero out all pixels where NDVI < 0.05
        mask = ndvi_resized < 0.05
        combined[mask] = 0.0
        #### ----------  ----------WATER MASK ---------- ----------

        if year not in predictions:
            predictions[year] = []

        predictions[year].append(combined)



from skimage.transform import resize

# assume predictions is { '2009':[map_2009], '2013':[map_2013], … }
years = sorted(predictions.keys(), key=int)
covers = []
for y in years:
    img = predictions[y][0]        # each is a 2D array e.g. (H,W) or (H,W,1)
    if img.ndim==3 and img.shape[2]==1:
        img = img[...,0]
    # resize to 4000×4000, preserve data‐range
    img4k = resize(img,
                   output_shape=(4000,4000),
                   order=1,            # bilinear
                   preserve_range=True,
                   anti_aliasing=True).astype(img.dtype)
    covers.append(img4k)

# 1) Suppose you have four prediction arrays (H×W), one per year:
#    cover_2011, cover_2013, cover_2016, cover_2017  (each shape (H,W))
#    load them however you like; here we assume they’re already in memory:
# covers = [cover_2011, cover_2013, cover_2016, cover_2017]

# 2) And the corresponding years:

# years = sorted(predictions.keys(), key=int)
# covers = [ predictions[year] for y in years ]


# 3) Stack into a 3D array of shape (4, H, W):
stack = np.stack(covers, axis=0)  # shape = (n_years, H, W)

# 4) Vectorized slope‐of‐-linear‐fit per pixel:
#    slope = cov(years, y) / var(years)
#    cov(years, y) = mean(years*y) - mean(years)*mean(y)
y_mean = stack.mean(axis=0)                 # shape (H,W)

years_cal = np.array([2009, 2013, 2016, 2017], dtype=float)  # ms6
# years_cal = np.array([2011, 2013, 2016, 2017], dtype=float)  # ms6 ------------->>>> NO 2009
years = years_cal - years_cal[0]

yr_mean = years.mean()
cov = (years[:,None,None] * stack).mean(axis=0) - yr_mean * y_mean
var = (years**2).mean() - yr_mean**2
slope_map = cov / var                       # shape (H,W)


slope_map_pct = slope_map * 100.0   # now in % per year

plt.figure(figsize=(8,8))
im = plt.imshow(
    slope_map_pct,
    cmap="RdBu",
    vmin=-2,        # e.g. ±2 % yr⁻¹
    vmax= 2
)
cbar = plt.colorbar(im, fraction=0.036, pad=0.04)
cbar.set_label("Cover Change (% per year)", fontsize=20, fontweight="bold")
cbar.ax.tick_params(labelsize=20)
# plt.title("Per‐pixel trend in cover (2009→2017)", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show(block=True)












##### Test COMBINE TRENDS SHRUBS and WET TUNDRA

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from tensorflow.keras.models import load_model
from skimage.transform import resize
import torch
import torch.nn.functional as F
from functionsdarko import (slice_overlap_simple, combine_tiles_to_large_image_predictionsoneyear, DinoV2DPTSegModel,
    combine_tiles_to_large_image_overlap_preserve_edges)

# 1) where your models live
# MODEL_DIR = "/Volumes/OWC Express 1M2/nasa_above/models"
# MODEL_DIR = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/"
MODEL_DIR = "/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2//models"
#shrub
# OUTPUT_DIR = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 2/trend_maps"
#wet tundra
# OUTPUT_DIR = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 2/trend_maps"
OUTPUT_DIR = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/figures/trend_maps'
#lakes
# OUTPUT_DIR = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/figures chapter 3/qgis/model predictions other sites LAKES'
os.makedirs(OUTPUT_DIR, exist_ok=True)



# # # 3) images
# PSHP_PATHS = [
#     ## ms6a2 LAPTOP NoN = Georef
#     # "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m.tif",
#     # {'name':'ms6a2', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2_GEOREF_clip1950m.tif'},
#     # {'name':'ms6a2', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2_GEOREF_clip1950m.tif'},
#     # {'name':'ms6a2', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m.tif'},
#     # {'name':'ms6a2', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/TIFF_2000m/ms6_timeseries_PSHP_as_TIFF/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m.tif'}
#     {'name':'uavsar_clip2a', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/UAVSAR/clip2a/QB02_20110727210843_101001000DD9B100_PSHP_P003_NT_clip2a.tif'},
#     {'name':'uavsar_clip2a', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/UAVSAR/clip2a/WV02_20170821223606_103001006F67D900_PSHP_P002_NT_clip2a.tif'},
#     {'name':'uavsar_clip2a', 'file':'/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/UAVSAR/clip2a/WV03_20190708223623_104001004E0F3D00_PSHP_P009_NT_clip2a.tif'},
# ]

## ADAPT
# PSHP_PATHS = [
#     # '/explore/nobackup/people/dradako1/sites/timeseries_multisite1/'
#     #ms1a1
#     # {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_1_10323_153130/QB02_20050724225626_1010010004654800_P001_area1_-2386945_4406130/QB02_20050724225626_1010010004654800_PSHP_P001_NT_2000m_area1.tif'},
#     {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_4_110323_154523/WV02_20130820230414_1030010025076500_P003_area1_-2386945_4406130/WV02_20130820230414_1030010025076500_PSHP_P003_NT_2000m_area1.tif'},
#     {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_7_110323_155000/WV03_20170813230621_1040010031CAA600_P003_area1_-2386945_4406130/WV03_20170813230621_1040010031CAA600_PSHP_P003_NT_2000m_area1.tif'},
#     {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_9_110323_155547/WV02_20200716223511_10300100A97DD900_P003_area1_-2386945_4406130/WV02_20200716223511_10300100A97DD900_PSHP_P003_NT_2000m_area1.tif'},
#     #ms4a1
#     {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_1_240323_202958/QB02_20020729211432_1010010000E40D00_P001_area1_-1746242_3978542/QB02_20020729211432_1010010000E40D00_PSHP_P001_NT_2000m_area1.tif'},
#     {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_4_240323_203154/QB02_20100701212531_101001000BE81F00_P002_area1_-1746242_3978542/QB02_20100701212531_101001000BE81F00_PSHP_P002_NT_2000m_area1.tif'},
#     {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_6_240323_203526/WV02_20130710213601_1030010024B9FD00_P007_area1_-1746242_3978542/WV02_20130710213601_1030010024B9FD00_PSHP_P007_NT_2000m_area1.tif'},
#     {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_13_240323_204307/WV02_20180826223220_103001008265EA00_P006_area1_-1746242_3978542/WV02_20180826223220_103001008265EA00_PSHP_P006_NT_2000m_area1.tif'},
#     #ms10a1
#     {'name':'ms10a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite10/timeseries_multisite10_1_240323_221916/QB02_20050724225630_1010010004654800_P002_area1_-2399591_4394681/QB02_20050724225630_1010010004654800_PSHP_P002_NT_2000m_area1.tif'},
#     {'name':'ms10a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite10/timeseries_multisite10_2_260423_125313/WV02_20140707224108_1030010033485600_P006_area1_-2399591_4394681/WV02_20140707224108_1030010033485600_PSHP_P006_NT_2000m_area1.tif'},
#     {'name':'ms10a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite10/timeseries_multisite10_4_240323_222216/WV03_20170813230622_1040010031CAA600_P004_area1_-2399591_4394681/WV03_20170813230622_1040010031CAA600_PSHP_P004_NT_2000m_area1.tif'},
#     # ms6a1 OWC Express paths NoN=Georef
#     {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_1_240323_215047/QB02_20090718220421_1010010009F45F00_P001_area1_-1948141_4014035/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area1.tif'},
#     # {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_4_180423_222328
#     # {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area1_-1948141_4014035/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area1.tif'}, #cloudy
#     {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area1_-1948141_4014035/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area1.tif'},
#     # {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area1_-1948141_4014035/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area1.tif'},  # missing part
#     # ms6a2 OWC Express paths NoN=Georef
#     # {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_1_240323_215047/QB02_20090718220421_1010010009F45F00_P001_area2_-1948141_4012041/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2.tif'},
#     {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_4_180423_222328/WV02_20110625223400_103001000B99EE00_P006_area2_-1948141_4012041/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2.tif'},
#     # {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area2_-1948141_4012041/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'},
#     {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area2_-1948141_4012041/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif'},
#     {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area2_-1948141_4012041/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'},
# ]

## ADAPT only two years for each site
PSHP_PATHS = [
    # '/explore/nobackup/people/dradako1/sites/timeseries_multisite1/'
    #ms1a1
    # {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_1_10323_153130/QB02_20050724225626_1010010004654800_P001_area1_-2386945_4406130/QB02_20050724225626_1010010004654800_PSHP_P001_NT_2000m_area1.tif'},
    # {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_4_110323_154523/WV02_20130820230414_1030010025076500_P003_area1_-2386945_4406130/WV02_20130820230414_1030010025076500_PSHP_P003_NT_2000m_area1.tif'},
    {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_7_110323_155000/WV03_20170813230621_1040010031CAA600_P003_area1_-2386945_4406130/WV03_20170813230621_1040010031CAA600_PSHP_P003_NT_2000m_area1.tif'},
    {'name':'ms1a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite1/timeseries_multisite1_9_110323_155547/WV02_20200716223511_10300100A97DD900_P003_area1_-2386945_4406130/WV02_20200716223511_10300100A97DD900_PSHP_P003_NT_2000m_area1.tif'},
    #ms4a1
    # {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_1_240323_202958/QB02_20020729211432_1010010000E40D00_P001_area1_-1746242_3978542/QB02_20020729211432_1010010000E40D00_PSHP_P001_NT_2000m_area1.tif'},
    # {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_4_240323_203154/QB02_20100701212531_101001000BE81F00_P002_area1_-1746242_3978542/QB02_20100701212531_101001000BE81F00_PSHP_P002_NT_2000m_area1.tif'},
    {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_6_240323_203526/WV02_20130710213601_1030010024B9FD00_P007_area1_-1746242_3978542/WV02_20130710213601_1030010024B9FD00_PSHP_P007_NT_2000m_area1.tif'},
    {'name':'ms4a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite4/timeseries_multisite4_13_240323_204307/WV02_20180826223220_103001008265EA00_P006_area1_-1746242_3978542/WV02_20180826223220_103001008265EA00_PSHP_P006_NT_2000m_area1.tif'},
    #ms10a1
    # {'name':'ms10a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite10/timeseries_multisite10_1_240323_221916/QB02_20050724225630_1010010004654800_P002_area1_-2399591_4394681/QB02_20050724225630_1010010004654800_PSHP_P002_NT_2000m_area1.tif'},
    {'name':'ms10a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite10/timeseries_multisite10_2_260423_125313/WV02_20140707224108_1030010033485600_P006_area1_-2399591_4394681/WV02_20140707224108_1030010033485600_PSHP_P006_NT_2000m_area1.tif'},
    {'name':'ms10a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite10/timeseries_multisite10_4_240323_222216/WV03_20170813230622_1040010031CAA600_P004_area1_-2399591_4394681/WV03_20170813230622_1040010031CAA600_PSHP_P004_NT_2000m_area1.tif'},
    # ms6a1 OWC Express paths NoN=Georef
    {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_1_240323_215047/QB02_20090718220421_1010010009F45F00_P001_area1_-1948141_4014035/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area1.tif'},
    # {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_4_180423_222328
    # {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area1_-1948141_4014035/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area1.tif'}, #cloudy
    {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area1_-1948141_4014035/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area1.tif'},
    # {'name':'ms6a1', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area1_-1948141_4014035/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area1.tif'},  # missing part
    # ms6a2 OWC Express paths NoN=Georef
    # {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_1_240323_215047/QB02_20090718220421_1010010009F45F00_P001_area2_-1948141_4012041/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2.tif'},
    # {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_4_180423_222328/WV02_20110625223400_103001000B99EE00_P006_area2_-1948141_4012041/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2.tif'},
    # {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_6_240323_215422/QB02_20130703205900_1010010011A29000_P003_area2_-1948141_4012041/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'},
    {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_9_240323_220244/WV03_20160714215120_104001001F9EF500_P001_area2_-1948141_4012041/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif'},
    {'name':'ms6a2', 'file':'/explore/nobackup/people/dradako1/sites/timeseries_multisite6/timeseries_multisite6_12_240323_221116/WV03_20170811223418_1040010031305A00_P001_area2_-1948141_4012041/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'},
]


# # 2) models shrub
# MODELS1 = [
#     { "name":"CNN512",   "type":"tf",  "file":"shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14.tf" },
#     { "name":"ResNet50", "type":"tf",  "file":"shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep6.tf" },
#     { "name":"VGG19",    "type":"tf",  "file":"shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep19.tf" },
#     { "name":"UNET256",  "type":"tf",  "file":"shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep20.tf" },
#     { "name":"VIT",      "type":"vit", "file":"shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h.pth" }
# ]

# # 2) models wet tundra
# MODELS2 = [
MODELS1 = [
    { "name":"CNN512",   "type":"tf",  "file":"wettundra_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep18.tf" },
    { "name":"ResNet50", "type":"tf",  "file":"wettundra_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT_ep21.tf" },
    { "name":"VGG19",    "type":"tf",  "file":"wettundra_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep21.tf" },
    { "name":"UNET256",  "type":"tf",  "file":"wettundra_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep4.tf" },
    { "name":"VIT",      "type":"vit", "file":"wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_CPUlocal_v5_ep7.pth" }
]
# 2) models lakes
MODELS2 = [
    { "name":"CNN512",   "type":"tf",  "file":"lakesrivers_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_ep24.tf" },
    { "name":"ResNet50", "type":"tf",  "file":"lakesrivers_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v4_ADAPT_ep11.tf" },
    { "name":"VGG19",    "type":"tf",  "file":"lakesrivers_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT_ep10.tf" },
    { "name":"UNET256",  "type":"tf",  "file":"lakesrivers_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep24.tf" },
    { "name":"VIT",      "type":"vit", "file":"lakes_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_localCPU_v2.pth" }
]


areas = sorted({entry['name'] for entry in PSHP_PATHS})
years = []
predictions_shrub = {}
predictions_wettundra = {}
for model_info, model_info2 in zip(MODELS1,MODELS2):
    # load model
    if model_info["type"] == "tf":
        mdl = load_model(os.path.join(MODEL_DIR, model_info["file"]), compile=False)
        mdl2 = load_model(os.path.join(MODEL_DIR, model_info2["file"]), compile=False)
    else:
        # Dino+DPT
        mdl = DinoV2DPTSegModel(
            backbone_name="vit_large_patch14_dinov2.lvd142m",
            out_channels=1, slice_3ch=True,
            pretrained_weights_path=None
        )
        # Dino+DPT
        mdl2 = DinoV2DPTSegModel(
            backbone_name="vit_large_patch14_dinov2.lvd142m",
            out_channels=1, slice_3ch=True,
            pretrained_weights_path=None
        )
        ckpt = torch.load(os.path.join(MODEL_DIR, model_info["file"]), map_location="cpu")
        ckpt2 = torch.load(os.path.join(MODEL_DIR, model_info2["file"]), map_location="cpu")
        # if your checkpoint is just state_dict:
        if "model_state_dict" in ckpt:
            mdl.load_state_dict(ckpt["model_state_dict"], strict=False)
            mdl2.load_state_dict(ckpt2["model_state_dict"], strict=False)
        else:
            mdl.load_state_dict(ckpt, strict=False)
            mdl2.load_state_dict(ckpt2, strict=False)
        mdl.eval()

    for area in areas:
        predictions_shrub = {}
        predictions_wettundra = {}
        print(f"  Area = {area}")

        # 2) select just the PSHP paths belonging to this area
        area_entries = [e for e in PSHP_PATHS if e['name'] == area]

        for pshp_info in area_entries:
            pshp = pshp_info['file']
            # print(pshp)
            p1bs = pshp.replace("PSHP","P1BS")  # --for using TIMESERIES FOLDER
            # p1bs = pshp.replace("x_train", "x_train2_remove_brightness/P1BS").replace("PSHP","P1BS") # --FOR GEOREF FOLDER ms6
            # p1bs = p1bs.replace('ms6_timeseries_P1BS_as_TIFF', 'ms6_timeseries_PSHP_as_TIFF') # --FOR GEOREF FOLDER ms6

            # slice
            tiles_pshp, metas = slice_overlap_simple(pshp, 200, overlap=0.25, resize_to=(400,400))
            tiles_p1bs, _    = slice_overlap_simple(p1bs,200, overlap=0.25, resize_to=(400,400))
            X = np.concatenate([np.stack(tiles_pshp), np.stack(tiles_p1bs)], axis=-1).astype("float32")/255.0
            filename = os.path.basename(pshp)
            year = filename.split('_')[1][:4]
            print(year)
            years.append(int(year))


            #### ---------- predict SHRUB ----------
            if model_info["type"] == "tf":
                preds = []
                for batch in np.array_split(X, 10):
                    preds.append(mdl.predict(batch, verbose=0))
                preds = np.concatenate(preds, axis=0)
            else:
                preds = []
                with torch.no_grad():
                    for batch in np.array_split(X, 8):
                        bt = torch.from_numpy(np.moveaxis(batch, -1,1)).float()
                        out = mdl(bt)
                        p = torch.sigmoid(out).cpu().numpy()
                        preds.append(np.moveaxis(p,1,-1))
                preds = np.concatenate(preds, axis=0)
            combined = combine_tiles_to_large_image_overlap_preserve_edges(preds, metas)

            #### ----------  ----------WATER MASK FOR SHRUBS ---------- ----------
            ndvi_path = pshp.replace("PSHP", "NDVI")    # -------------------------->for using TIMESERIES FOLDER
            # ndvi = pshp.replace("x_train", "x_train2_remove_brightness/NDVI").replace(".tif", "_NDVI.tif")
            # ndvi_path = ndvi.replace('ms6_timeseries_NDVI_as_TIFF', 'ms6_timeseries_PSHP_as_TIFF')
            with rasterio.open(ndvi_path) as src_ndvi:
                ndvi_full = src_ndvi.read(1)  # shape (H_ndvi, W_ndvi)
            ndvi_resized = resize(
                ndvi_full,
                output_shape=combined.shape,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(ndvi_full.dtype)
            mask = ndvi_resized < 0.05   ### ----------------------------------------> SHURBS / WET TUNDRA
            # mask = ndvi_resized > 0.05   ### ----------------------------------------> LAKES
            combined[mask] = 0.0
            #### ----------  ----------WATER MASK ---------- ----------

            if year not in predictions_shrub:
                predictions_shrub[year] = []

            predictions_shrub[year].append(combined)



            ### ---------- WETTUNDRA ----------
            if model_info["type"] == "tf":
                preds = []
                for batch in np.array_split(X, 10):
                    preds.append(mdl2.predict(batch, verbose=0))
                preds = np.concatenate(preds, axis=0)
            else:
                preds = []
                with torch.no_grad():
                    for batch in np.array_split(X, 8):
                        bt = torch.from_numpy(np.moveaxis(batch, -1, 1)).float()

                        # apply ImageNet norm for VIT:
                        ### ----------  ----------WET TUNDRA---------- ----------
                        m0 = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
                        s0 = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
                        bt[:,:3] = (bt[:,:3]-m0)/s0
                        ### ----------  ----------WET TUNDRA---------- ----------

                        out = mdl2(bt)
                        p = torch.sigmoid(out).cpu().numpy()
                        preds.append(np.moveaxis(p, 1, -1))
                preds = np.concatenate(preds, axis=0)

            combined = combine_tiles_to_large_image_overlap_preserve_edges(preds, metas)

            #### ----------  ----------WATER MASK FOR SHRUBS ---------- ----------
            ndvi_path = pshp.replace("PSHP", "NDVI")  # -------------------------->for using TIMESERIES FOLDER
            # ndvi = pshp.replace("x_train", "x_train2_remove_brightness/NDVI").replace(".tif", "_NDVI.tif")
            # ndvi_path = ndvi.replace('ms6_timeseries_NDVI_as_TIFF', 'ms6_timeseries_PSHP_as_TIFF')
            with rasterio.open(ndvi_path) as src_ndvi:
                ndvi_full = src_ndvi.read(1)  # shape (H_ndvi, W_ndvi)
            ndvi_resized = resize(
                ndvi_full,
                output_shape=combined.shape,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(ndvi_full.dtype)
            # mask = ndvi_resized < 0.05  ### ----------------------------------------> SHURBS / WET TUNDRA
            mask = ndvi_resized > 0.05   ### ----------------------------------------> LAKES
            combined[mask] = 0.0
            #### ----------  ----------WATER MASK ---------- ----------

            if year not in predictions_wettundra:
                predictions_wettundra[year] = []

            predictions_wettundra[year].append(combined)



        years = sorted(predictions_shrub.keys(), key=int)
        covers = []
        covers2 = []
        for y in years:
            img = predictions_shrub[y][0]  # each is a 2D array e.g. (H,W) or (H,W,1)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[..., 0]
            # resize to 4000×4000, preserve data‐range
            img4k = resize(img,
                           output_shape=(4000, 4000),
                           order=1,  # bilinear
                           preserve_range=True,
                           anti_aliasing=True).astype(img.dtype)
            covers.append(img4k)

        for y in years:
            img = predictions_wettundra[y][0]  # each is a 2D array e.g. (H,W) or (H,W,1)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[..., 0]
            # resize to 4000×4000, preserve data‐range
            img4k = resize(img,
                           output_shape=(4000, 4000),
                           order=1,  # bilinear
                           preserve_range=True,
                           anti_aliasing=True).astype(img.dtype)
            covers2.append(img4k)

        # 3) Stack into a 3D array of shape (4, H, W):
        stack = np.stack(covers, axis=0)  # shape = (n_years, H, W)
        y_mean = stack.mean(axis=0)  # shape (H,W)
        years_cal = np.array(years, dtype=float)  # ms6
        years2 = years_cal - years_cal[0]
        yr_mean = years2.mean()
        cov = (years2[:, None, None] * stack).mean(axis=0) - yr_mean * y_mean
        var = (years2 ** 2).mean() - yr_mean ** 2
        slope_map_shrub = cov / var  # shape (H,W)
        slope_map_pct_shrub = slope_map_shrub * 100.0  # now in % per year

        # 3) Stack into a 3D array of shape (4, H, W):
        stack = np.stack(covers2, axis=0)  # shape = (n_years, H, W)
        y_mean = stack.mean(axis=0)  # shape (H,W)
        years_cal = np.array(years, dtype=float)  # ms6
        years2 = years_cal - years_cal[0]
        yr_mean = years2.mean()
        cov = (years2[:, None, None] * stack).mean(axis=0) - yr_mean * y_mean
        var = (years2 ** 2).mean() - yr_mean ** 2
        slope_map_wettundra = cov / var  # shape (H,W)
        slope_map_pct_wettundra = slope_map_wettundra * 100.0  # now in % per year



        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.transform import resize

        # --- assume you’ve already computed two slope maps in % per year:
        #      slope_shrub_pct and slope_wet_pct, both shape (H,W)

        # 1) clip to a symmetric range around zero (e.g. ±2% yr⁻¹)
        clip = 2.0
        s_shrub = np.clip(slope_map_pct_shrub, -clip, clip)
        s_wet = np.clip(slope_map_pct_wettundra, -clip, clip)

        # 2) normalize into [0,1], so 0→blue (shrub=0 or wet=0), 0.5→0% trend, 1→+clip
        shrub_norm = (s_shrub / (2 * clip)) + 0.5
        wet_norm = (s_wet / (2 * clip)) + 0.5

        # sanity‐clip
        shrub_norm = np.clip(shrub_norm, 0, 1)
        wet_norm = np.clip(wet_norm, 0, 1)

        # 3) build RGB composite
        rgb = np.dstack([
            shrub_norm,  # R channel = shrub trend
            wet_norm,  # G channel = wet trend
            np.zeros_like(shrub_norm)  # B = 0
        ])

        # if you need to resize to 4000×4000:
        rgb = resize(rgb,
                     output_shape=(4000, 4000, 3),
                     preserve_range=True,
                     anti_aliasing=True)

        # 4) plot
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.axis('off')
        # legend (manually)
        import matplotlib.patches as mpatches

        patches = [
            # mpatches.Patch(color=(1, 0, 0), label='Shrub ↑ only'),
            mpatches.Patch(color=(1, 0, 0), label='Wet ↑ only'),
            # mpatches.Patch(color=(0, 1, 0), label='Wet ↑ only'),
            mpatches.Patch(color=(0, 1, 0), label='Surface-Water ↑ only'),
            mpatches.Patch(color=(1, 1, 0), label='Both ↑'),
            mpatches.Patch(color=(0, 0, 0.5), label='Both ↓'),
        ]
        plt.legend(handles=patches, loc='upper right', fontsize=12, framealpha=0.9)
        plt.tight_layout()

        ## SAVE
        # outfn = f"{model_info['name']}_TRENDMAP_{pshp_info['name']}_{os.path.basename(pshp).replace('.tif', '_SHRUBS.jpg')}"
        # outfn = f"{model_info['name']}_TRENDMAP_{pshp_info['name']}_{os.path.basename(pshp).replace('.tif','_WETTUNDRA.jpg')}"
        # outfn = f"{model_info['name']}_TRENDMAP_{pshp_info['name']}_{os.path.basename(pshp).replace('.tif', '_LAKES.jpg')}"
        # outfn = f"2dimensional_quadrant_TRENDmap_{model_info['name']}_{pshp_info['name']}_{os.path.basename(pshp).replace('.tif', '_SHRUBS_WETTUNDRAv2.jpg')}"
        # outfn = f"2dimensional_quadrant_TRENDmap_{model_info['name']}_{pshp_info['name']}_{os.path.basename(pshp).replace('.tif', '_SHRUBS_WETTUNDRAv3.jpg')}"  # only comparing two years
        # outfn = f"2dimensional_quadrant_TRENDmap_{model_info['name']}_{pshp_info['name']}_{os.path.basename(pshp).replace('.tif', '_SHRUBS_LAKESv3.jpg')}"  # only comparing two years
        outfn = f"2dimensional_quadrant_TRENDmap_{model_info['name']}_{pshp_info['name']}_{os.path.basename(pshp).replace('.tif', '_WETTUNDRA_LAKESv3.jpg')}"  # only comparing two years
        plt.savefig(os.path.join(OUTPUT_DIR, outfn), dpi=200, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(outfn)
        # plt.show()

        # predictions = {}
        years = []




#
#
# areas = sorted({entry['name'] for entry in PSHP_PATHS})
# years = []
# predictions = {}
# for model_info in MODELS2:
#     # load model
#     if model_info["type"] == "tf":
#         mdl = load_model(os.path.join(MODEL_DIR, model_info["file"]), compile=False)
#     else:
#         # Dino+DPT
#         mdl = DinoV2DPTSegModel(
#             backbone_name="vit_large_patch14_dinov2.lvd142m",
#             out_channels=1, slice_3ch=True,
#             pretrained_weights_path=None
#         )
#         ckpt = torch.load(os.path.join(MODEL_DIR, model_info["file"]), map_location="cpu")
#         # if your checkpoint is just state_dict:
#         if "model_state_dict" in ckpt:
#             mdl.load_state_dict(ckpt["model_state_dict"], strict=False)
#         else:
#             mdl.load_state_dict(ckpt, strict=False)
#         mdl.eval()
#
#     for area in areas:
#         predictions = {}
#         print(f"  Area = {area}")
#
#         # 2) select just the PSHP paths belonging to this area
#         area_entries = [e for e in PSHP_PATHS if e['name'] == area]
#
#         for pshp_info in area_entries:
#             pshp = pshp_info['file']
#             # print(pshp)
#             # p1bs = pshp.replace("PSHP","P1BS")  # --for using TIMESERIES FOLDER
#             p1bs = pshp.replace("x_train", "x_train2_remove_brightness/P1BS").replace("PSHP","P1BS") # --FOR GEOREF FOLDER ms6
#             p1bs = p1bs.replace('ms6_timeseries_P1BS_as_TIFF', 'ms6_timeseries_PSHP_as_TIFF') # --FOR GEOREF FOLDER ms6
#
#             # slice
#             tiles_pshp, metas = slice_overlap_simple(pshp, 200, overlap=0.25, resize_to=(400,400))
#             tiles_p1bs, _    = slice_overlap_simple(p1bs,200, overlap=0.25, resize_to=(400,400))
#             X = np.concatenate([np.stack(tiles_pshp), np.stack(tiles_p1bs)], axis=-1).astype("float32")/255.0
#
#             filename = os.path.basename(pshp)
#             # sensor = filename[:4]
#             year = filename.split('_')[1][:4]
#             # identifier = f"{sensor}_{year}"
#             print(year)
#             years.append(int(year))
#             # predict
#             if model_info["type"] == "tf":
#                 preds = []
#                 for batch in np.array_split(X, 10):
#                     preds.append(mdl.predict(batch, verbose=0))
#                 preds = np.concatenate(preds, axis=0)
#             else:
#                 preds = []
#                 with torch.no_grad():
#                     for batch in np.array_split(X, 8):
#                         bt = torch.from_numpy(np.moveaxis(batch, -1,1)).float()
#
#                         # # apply ImageNet norm for VIT:
#                         # ### ----------  ----------WET TUNDRA---------- ----------
#                         # m0 = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
#                         # s0 = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
#                         # bt[:,:3] = (bt[:,:3]-m0)/s0
#                         # ### ----------  ----------WET TUNDRA---------- ----------
#
#                         out = mdl(bt)
#                         p = torch.sigmoid(out).cpu().numpy()
#                         preds.append(np.moveaxis(p,1,-1))
#                 preds = np.concatenate(preds, axis=0)
#
#             # stitch
#             # combined, _ = combine_tiles_to_large_image_predictionsoneyear(preds, metas)
#             combined = combine_tiles_to_large_image_overlap_preserve_edges(preds, metas)
#
#             #### ----------  ----------WATER MASK FOR SHRUBS ---------- ----------
#             # ndvi_path = pshp.replace("PSHP", "NDVI")  # -------------------------->for using TIMESERIES FOLDER
#             ndvi = pshp.replace("x_train", "x_train2_remove_brightness/NDVI").replace(".tif", "_NDVI.tif")
#             ndvi_path = ndvi.replace('ms6_timeseries_NDVI_as_TIFF', 'ms6_timeseries_PSHP_as_TIFF')
#             # 1) load the full NDVI raster once
#             with rasterio.open(ndvi_path) as src_ndvi:
#                 ndvi_full = src_ndvi.read(1)  # shape (H_ndvi, W_ndvi)
#             # 2) resize it to match your combined’s shape (Hc, Wc)
#             ndvi_resized = resize(
#                 ndvi_full,
#                 output_shape=combined.shape,
#                 preserve_range=True,
#                 anti_aliasing=True,
#             ).astype(ndvi_full.dtype)
#             # 3) build a mask and zero out all pixels where NDVI < 0.05
#             mask = ndvi_resized < 0.05   ### ----------------------------------------> SHURBS / WET TUNDRA
#             # mask = ndvi_resized > 0.05   ### ----------------------------------------> LAKES
#             combined[mask] = 0.0
#             #### ----------  ----------WATER MASK ---------- ----------
#
#
#             if year not in predictions:
#                 predictions[year] = []
#
#             predictions[year].append(combined)
#
#
#         predictions_wettundra = predictions.copy()
#
#         # assume predictions is { '2009':[map_2009], '2013':[map_2013], … }
#         years = sorted(predictions_wettundra.keys(), key=int)
#         covers = []
#         for y in years:
#             img = predictions_wettundra[y][0]  # each is a 2D array e.g. (H,W) or (H,W,1)
#             if img.ndim == 3 and img.shape[2] == 1:
#                 img = img[..., 0]
#             # resize to 4000×4000, preserve data‐range
#             img4k = resize(img,
#                            output_shape=(4000, 4000),
#                            order=1,  # bilinear
#                            preserve_range=True,
#                            anti_aliasing=True).astype(img.dtype)
#             covers.append(img4k)
#         # 3) Stack into a 3D array of shape (4, H, W):
#         stack = np.stack(covers, axis=0)  # shape = (n_years, H, W)
#         y_mean = stack.mean(axis=0)  # shape (H,W)
#         years_cal = np.array(years, dtype=float)  # ms6
#         years2 = years_cal - years_cal[0]
#         yr_mean = years2.mean()
#         cov = (years2[:, None, None] * stack).mean(axis=0) - yr_mean * y_mean
#         var = (years2 ** 2).mean() - yr_mean ** 2
#         slope_map_wettundra = cov / var  # shape (H,W)
#         slope_map_pct_wettundra = slope_map_wettundra * 100.0  # now in % per year







        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.transform import resize

        # --- assume you’ve already computed two slope maps in % per year:
        #      slope_shrub_pct and slope_wet_pct, both shape (H,W)

        # 1) clip to a symmetric range around zero (e.g. ±2% yr⁻¹)
        clip = 2.0
        s_shrub = np.clip(slope_map_pct_shrub, -clip, clip)
        s_wet = np.clip(slope_map_pct_wettundra, -clip, clip)

        # 2) normalize into [0,1], so 0→blue (shrub=0 or wet=0), 0.5→0% trend, 1→+clip
        shrub_norm = (s_shrub / (2 * clip)) + 0.5
        wet_norm = (s_wet / (2 * clip)) + 0.5

        # sanity‐clip
        shrub_norm = np.clip(shrub_norm, 0, 1)
        wet_norm = np.clip(wet_norm, 0, 1)

        # 3) build RGB composite
        rgb = np.dstack([
            shrub_norm,  # R channel = shrub trend
            wet_norm,  # G channel = wet trend
            np.zeros_like(shrub_norm)  # B = 0
        ])

        # if you need to resize to 4000×4000:
        rgb = resize(rgb,
                     output_shape=(4000, 4000, 3),
                     preserve_range=True,
                     anti_aliasing=True)

        # 4) plot
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.axis('off')
        # legend (manually)
        import matplotlib.patches as mpatches

        patches = [
            mpatches.Patch(color=(1, 0, 0), label='Shrub ↑ only'),
            mpatches.Patch(color=(0, 1, 0), label='Wet ↑ only'),
            mpatches.Patch(color=(1, 1, 0), label='Both ↑'),
            mpatches.Patch(color=(0, 0, 0.5), label='Both ↓'),
        ]
        plt.legend(handles=patches, loc='lower left', fontsize=12, framealpha=0.9)

        # 5) save
        plt.tight_layout()
        plt.show()





