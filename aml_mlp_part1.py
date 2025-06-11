#meta for myFiftyoneComputerVision 6/10/2025 AML MLP 1 POC AI Doc Vision. Split MLP Vis51 -> Part1 (GPU ok code), dataset $config
# AML: run this code in cloud on GPU for speed (.py only for now, conda env `evn-aml-doc-vision-310`)
# MLP: get images from a dir -> create 51dataset w/ images, embeddings, and PCA dim reductions -> export 51dataset
# $next: Part 2 (off GPU code)

#References:
# refer to https://docs.voxel51.com/tutorials/dimension_reduction.html?highlight=umap
# code src: https://github.com/voxel51/fiftyone/blob/v1.4.0/docs/source/tutorials/dimension_reduction.ipynb
# copy of aml_mlp_part1.py (for internal)

#infra_original: WLaptop + VSCode
#      env: default
#      confirmed Python 3.10.4
#      numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, matplotlib 3.10.0
#      pip 22.0.4, ipykernel 6.29.0, ipython 8.20.0
#fiftyone 1.4.0, fiftyone-brain 0.20.1, fiftyone_db 1.1.7
#umap-learn 0.5.7
#glob2 0.7

#infra2: Azure cloud, env conda activate evn-aml-doc-vision-310
# refer to environment_droplet.yml, environment_droplet_hist.yml and extra steps
# refer to environment_droplet_gpu2.yml, environment_droplet_gpu2_hist.yml


#input: $config 
#        DATA_PATH = 'DOC_DATA' 
#        IMAGES_PATH = 'IMAGES_SAMPLE-INVOICES-250'
#        DATA_SAMPLES_IN = DATA_PATH + '/df_sample-invoices-250_metadata.parquet'
#        EXPORT_PATH='persist51_sample-invoices-250'
#output: DATASET_NAME = "sample-invoices-250"
#        persist51_[DATASET_NAME]


#history (per original)
#5/20/2025 AML SPLIT MLP -> PART 1 (GPU OK CODE)
#      MLP Vis51 Data w/ Dimensionality Reduction,  dataset $config
#      Mostly needed accelarated Embeddings piece
#      Limited reductions, PCA only - GPU Ok code
#      Tracking time
# $next: Part 2 (off GPU code) - add TSNE add TSNE and UMAP reductions, no fields yet


#$MLP (here) part1 -> part2 -> zip and download output dir -> Vis51 mlp_2_deploy.ipynb on-prem

#---------------------------------------------------------------------------------------------------------------------
import os
import sys
import time as time

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', 150)

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

#!pip install -U fiftyone scikit-learn umap-learn
#---------------------------------------------------------------------------------------------------------------------

 #----- $config -----
DATA_PATH = 'DOC_DATA' 
IMAGES_PATH = 'IMAGES_SAMPLE-INVOICES-250'
DATA_SAMPLES_IN = DATA_PATH + '/df_sample-invoices-250_metadata.parquet'
EXPORT_PATH='persist51_sample-invoices-250'

# #dataset
DATASET_NAME = "sample-invoices-250" 
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
# POC AI-DOC-VISION
## Visualizing Data with Dimensionality Reduction Techniques - Part 1 (GPU Ok code)
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#track times
t0 = time.time()

#---------------------------------------------------------------------------------------------------------------------
## 0. Load Data
# Start with 250 docs (invoice samples)
# 0a. Load Pre-created Dataset as a dataframe  
#     `image_file_path`, `image_file_name`, `embeddings_resnet101`, `embeddings_clip
df_doc_samples = pd.read_parquet(DATA_SAMPLES_IN)
print(df_doc_samples.shape)
print(df_doc_samples.info())

t1 = time.time()
print("Loaded dataframe w/ doc data (in min): ", (t1 - t0)/60)
#---------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------
## 1. Create 51Dataset
print("\n")
print(fo.list_datasets())

# step: Load 51Dataset 1st time
try:
    # Load your FiftyOne dataset
    dataset_doc_samples = fo.load_dataset(DATASET_NAME) #class 'fiftyone.core.dataset.Dataset'
    t2 = time.time()
    print("\nLoaded existing 51Dataset (in min): ", (t2 - t1)/60)
except ValueError:
    # If the dataset doesn't exist, create it from a dir of images
    dataset_doc_samples = fo.Dataset.from_images_dir(IMAGES_PATH, name=DATASET_NAME) #class 'fiftyone.core.dataset.Dataset'
    t2 = time.time()
    print("Loaded new 51Dataset (in min): ", (t2 - t1)/60)
    #---------------------------------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------------------------------
    # step: load vis models
    resnet101 = foz.load_zoo_model("resnet101-imagenet-torch")
    t3a = time.time()
    print("\n(re)Loaded vis model Resnet101 (in min): ", (t3a - t2)/60)

    clip = foz.load_zoo_model("clip-vit-base32-torch")
    t3b = time.time()
    print("(re)Loaded vis model CLIP (in min): ", (t3b - t3a)/60)
    print("\n")
    #---------------------------------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------------------------------
    ## compute and store resnet101 embeddings $note: time consuming
    dataset_doc_samples.compute_embeddings(
        resnet101, 
        embeddings_field="resnet101_embeddings"
    )
    t4a = time.time()
    print("\nComputed Resnet101 embeddings (in min): ", (t4a - t3b)/60)

    ## compute and store clip embeddings 
    dataset_doc_samples.compute_embeddings(
        clip, 
        embeddings_field="clip_embeddings"
    )
    t4b = time.time()
    print("Computed CLIP embeddings (in min): ", (t4b - t4a)/60)

    print(dataset_doc_samples)
    #---------------------------------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------------------------------
    #PCA on embeddings
    ## PCA with ResNet101 embeddings
    fob.compute_visualization(
        dataset_doc_samples, 
        embeddings="resnet101_embeddings", 
        method="pca", 
        brain_key="resnet101_pca"
    )
    t5a = time.time()
    print("\nComputed PCA with Resnet101 embeddings (in min): ", (t5a - t4b)/60)

    ## PCA with CLIP embeddings
    fob.compute_visualization(
        dataset_doc_samples, 
        embeddings="clip_embeddings", 
        method="pca", 
        brain_key="clip_pca"
    )
    t5b = time.time()
    print("Computed PCA with CLIP embeddings (in min): ", (t5b - t5a)/60)
    #---------------------------------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------------------------------
    #part 2 -> TSNE and UMPA on embeddings (off GPU compute until figure it out)
    #---------------------------------------------------------------------------------------------------------------------

except :
    print("\nAn error ocurred.")
    for e in sys.exc_info():
        print ("Error details: {} ".format(str(e)))
finally:
    print("\n")
    print(fo.list_datasets())
    print("\n")
    print(dataset_doc_samples)
    print("\n")


#---------------------------------------------------------------------------------------------------------------------
#persist 51dataset
dataset_doc_samples.export(export_dir=EXPORT_PATH, dataset_type=fo.types.FiftyOneDataset)
t10 = time.time()
print("Persisted 51dataset (in min): ", (t10 - t5b)/60)

t20 = time.time()
print("\nTotal time (in min): ", (t20 - t0)/60)
#---------------------------------------------------------------------------------------------------------------------