#meta for myFiftyoneComputerVision 6/10/2025. AML MLP. POC AI Doc Vision. Split MLP Vis51 -> Part2 CPU (off GPU code), dataset $config
# AML: run this code on prem WLap (default env) or in cloud ok on CPU (.py only for now, conda env `evn-aml-doc-vision-310`)
# MLP: export pre-created 51dataset w/ images, embeddings, and PCA dim reductions -> append w/ TSNE and UMAP reductions -> export 51dataset

#References:
# refer to https://docs.voxel51.com/tutorials/dimension_reduction.html?highlight=umap
# code src: https://github.com/voxel51/fiftyone/blob/v1.4.0/docs/source/tutorials/dimension_reduction.ipynb
# copy of aml_mlp_part2.py

#infra_original: WLaptop + VSCode
#      env: default
#      confirmed Python 3.10.4
#      numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, matplotlib 3.10.0
#      pip 22.0.4, ipykernel 6.29.0, ipython 8.20.0
#fiftyone 1.4.0, fiftyone-brain 0.20.1, fiftyone_db 1.1.7
#umap-learn 0.5.7
#glob2 0.7

#infra2: Azure cloud,  CPU compute 'cpu-ml-doc-vision' 
#      env conda activate evn-aml-doc-vision-310
# refer to environment_droplet.yml, environment_droplet_hist.yml 
#       environment_droplet_updateUMAP.yml, environment_droplet_updateUMAP_hist.yml and UMAP works


#input: $config 
#        DATA_PATH = 'DOC_DATA' 
#        IMAGES_PATH = 'IMAGES_SAMPLE-INVOICES-250'
#        DATA_SAMPLES_IN = DATA_PATH + '/df_sample-invoices-250_metadata.parquet'
#        EXPORT_PATH='persist51_sample-invoices-250'
#output: DATASET_NAME = "sample-invoices-250"
#        persist51_[DATASET_NAME]


#previously in aml_mlp_part2.py
#5/20/2025 AML SPLIT MLP -> PART 1 (GPU OK CODE)
#      MLP Vis51 Data w/ Dimensionality Reduction,  dataset $config
#      Mostly needed accelarated Embeddings piece
#      Limited reductions, PCA only - GPU Ok code
# $next: Part 2 (off GPU code) - add TSNE (no UMAP) reductions, no fields yet


#history (per original)
#5/20/2025 AML SPLIT MLP -> PART 2 (OFF GPU CODE)
#      MLP Vis51 Data continues, dataset $config
#      Export pre-created 51dataset w/ images, embeddings, and PCA dim reductions -> append w/ TSNE and UMAP reductions -> export 51dataset
#      OK to run off GPU, until figure out GPU for TSNE and UMAP
# $next: add fields

#6/10/2025 AML SPLIT MLP -> PART 2, SOME WORKS ON GPU
#      Dataset 'sample-invoices-250'
#      No UMAP reductions
#      Which means real 'docs-5329' still errors out, UMAP reductions still error out
#      Confirmed Vis51 on-prem deployment (4 brain keys)
#$next: non-working pieces will work on CPU compute
#
#6/10/2025 AML SPLIT MLP -> PART 2 WORKS ON CPU
#      Dataset 'sample-invoices-250'
#      UMAP reductions
#      Confirmed Vis51 on-prem deployment (6 brain keys)
#per my memory, real 'docs-5329' also works on CPU but not confirmed


#$MLP part1 -> (here) part2 -> zip and download output dir -> Vis51 mlp_2_deploy.ipynb on-prem


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

#---- GlOBAL VARS ----------------------------------------------------------------------------------------------------
##data #$config 
#EXPORT_PATH='persist51_doc-images-2359'
EXPORT_PATH='persist51_sample-invoices-250'

# #dataset
#DATASET_NAME = "doc-images-2359" #$config #"doc-sample-invoices-250" 
DATASET_NAME = "sample-invoices-250"
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
# POC AI-DOC-VISION
## Visualizing Data with Dimensionality Reduction Techniques - Part 2 (off GPU code)
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#track times
t0 = time.time()

#---------------------------------------------------------------------------------------------------------------------
## 1. Append 51Dataset
#part 2 -> TSNE and UMAP on embeddings (off GPU compute until figure it out)
print(fo.list_datasets())

# step: Export pre-created 51Dataset
try:
    # Export FiftyOne dataset
    dataset_doc_samples =  fo.Dataset.from_dir(EXPORT_PATH, dataset_type=fo.types.FiftyOneDataset, name=DATASET_NAME) #class 'fiftyone.core.dataset.Dataset'
    t2 = time.time()
    print("Exported existing 51Dataset (in min): ", (t2 - t0)/60)
    #---------------------------------------------------------------------------------------------------------------------
except ValueError:
    # Load FiftyOne dataset (pre-loaded)
    if DATASET_NAME in fo.list_datasets():
        dataset_doc_samples = fo.load_dataset(DATASET_NAME)
        t2 = time.time()
        print("Loaded existing 51Dataset (in min): ", (t2 - t0)/60)
    else:
        print("Unexpected behavior")
except :
    print("\nAn error ocurred.")
    for e in sys.exc_info():
        print ("Error details: {} ".format(str(e)))
finally:

    #---------------------------------------------------------------------------------------------------------------------
    #TSNE on embeddings
    ## t-SNE with ResNet101 embeddings
    fob.compute_visualization(
        dataset_doc_samples, 
        embeddings="resnet101_embeddings", 
        method="tsne", 
        brain_key="resnet101_tsne"
    )
    t6a = time.time()
    print("\nComputed t-SNE with Resnet101 embeddings (in min): ", (t6a - t2)/60)

    print("Compute t-SNE with CLIP embeddings --> ") #$acdelete

    ## t-SNE with CLIP embeddings
    fob.compute_visualization(
        dataset_doc_samples, 
        embeddings="clip_embeddings", 
        method="tsne", 
        batch_size=1000,
        num_workers=6,
        skip_failures=True,
        brain_key="clip_tsne"
    )
    
    t6b = time.time()
    print("Computed t-SNE with CLIP embeddings (in min): ", (t6b - t6a)/60)
    #---------------------------------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------------------------------
    #UMAP on embeddings
    ## UMAP with ResNet101 embeddings
    fob.compute_visualization(
        dataset_doc_samples, 
        embeddings="resnet101_embeddings", 
        method="umap", 
        brain_key="resnet101_umap"
    )
    t7a = time.time()
    print("Computed UMAP with Resnet101 embeddings (in min): ", (t7a - t6b)/60)

    ## UMAP with CLIP embeddings
    fob.compute_visualization(
        dataset_doc_samples, 
        embeddings="clip_embeddings", 
        method="umap", 
        brain_key="clip_umap"
    )
    t7b = time.time()
    print("Computed UMAP with CLIP embeddings (in min): ", (t7b - t6b)/60)
    #---------------------------------------------------------------------------------------------------------------------

    print("\n")
    print(fo.list_datasets())
    print("\n")
    print(dataset_doc_samples)
    print("\n")


#---------------------------------------------------------------------------------------------------------------------
#persist 51dataset
dataset_doc_samples.export(export_dir=EXPORT_PATH, dataset_type=fo.types.FiftyOneDataset)
t10 = time.time()
print("Persisted 51dataset (in min): ", (t10 - t7b)/60)

t20 = time.time()
print("\nTotal time (in min): ", (t20 - t0)/60)
#---------------------------------------------------------------------------------------------------------------------
