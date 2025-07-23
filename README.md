# myFiftyOne and Computer Vision
Voxel51 Computer Vision - embeddings, dimensionality reduction, clustering, vis and more

## Resources
Overview: https://docs.voxel51.com/index.html  
Tutorials: https://docs.voxel51.com/tutorials/index.html  
- Using Image Embeddings  
- Dimensionality Reduction  
- Clustering Images  
- Exploring Image Uniqueness  
- Anomaly Detection  

Recipes: https://docs.voxel51.com/recipes/index.html  
- Creating Views  
- Deduplication  
- Drawing Labels on Samples 

FiftyOne Repo: https://github.com/voxel51/fiftyone

## Description
Visualizing Data with Dimensionality Reduction Techniques

Using 51 walkthrough, run dimensionality reduction techniques (PCA, t-SNE, UMAP) on shipping docs in FiftyOne!

MLP: get images from a dir -> create 51dataset w/ images, embeddings, and PCA dim reductions -> export 51dataset

## Data Naming Standards 
1.  Dataset names  
-real ds `docs-5329`  
-sample ds `sample-invoices-250`  

2. Folders  
- Folder for metadata  
->`DOC_DATA`  
`df_docs-5329_metadata.parquet`   
`df_sample-invoices-250_metadata.parquet`

- ?Folders for original pdfs to be converted to images  
->`PDFs_[?]` 
 
- Folders for pre-converted images  
-> `IMAGES_[DS NAME]` (yes all in CAPS)  
`IMAGES_DOCS-5329`  
`IMAGES_SAMPLE-INVOICES-250`  
`IMAGES_DOCS-25` tiny sample  
 

- Folders for exported 51datasets (w/ its own structure)  
-> `persist_[ds_name]` (all in lowercase)  
`persist51_docs-5329`  
`persist51_sample-invoices-250`   

## Project status
POC stage, running compute intensive tasks on GPU in Azure Machine Learning Studio and then viewing the results locally in FiftyOne tool.

## Solution design
- Flowchart (image)
![alt Flowchart](vis/vis_mermaid_51Flowchart.PNG)

### Mermaid Diagram Flowchart

```mermaid
  flowchart LR
    a1("`**Azure File Storage**
    Store PDFs`") --> a2("`**Azure ML**
    aml_mlp_0_data.ipynb
    Get PDFs from storage`")

    subgraph "**Azure Images Dataset**"
      a2("`**Azure ML**
      aml_mlp_0_data.ipynb
      Get PDFs from storage`") --> a3("`**Azure ML**
      aml_mlp_1_prep_data.ipynb
      Convert PDFs to images`")
    end       
```

```mermaid

  flowchart LR
    a11("`**Original Docs Storage**
    Images and Metadata`") --> a12("`**WLaptop VSCode**
    mlp_0_data.ipynb
    Tidy metadata
    df_samples.parquet`")

   a12("`**WLaptop VSCode**
    mlp_0_data.ipynb
    Tidy metadata
    df_samples.parquet`") --> b1("`**Azure ML GPU**
      aml_mlp_part1.py
      create 51dataset w/ images, embeddings,
      and PCA dim reductions`")

            
    subgraph "**On-prem Full MLP**"
      a12("`**WLaptop VSCode**
      mlp_0_data.ipynb
      Tidy metadata
      df_samples.parquet`") --> a13("`**WLaptop VSCode**
      mlp_1_prep_data.ipynb
      Full MLP: from images to 51App`")
    end

    subgraph "**Azure MLP: From Images to 51Dataset**"
      b1("`**Azure ML GPU**
      aml_mlp_part1.py
      create 51dataset w/ images, embeddings,
      and PCA dim reductions`") --> b2("`**Azure ML CPU**
      aml_mlp_part2.py
      append w/ TSNE and UMAP reductions`")
      b1("`**Azure ML GPU**
      aml_mlp_part1.py
      create 51dataset w/ images, embeddings,
      and PCA dim reductions`") --> b3[Export 51Dataset]
      b2("`**Azure ML CPU**
      aml_mlp_part2.py
      append w/ TSNE and UMAP reductions`") --> b3[Export 51Dataset]
    end

    subgraph "**On-Prem 51App**"
      b3[Export 51Dataset] --> c1[Visualize 51Dataset]
      a13("`**WLaptop VSCode**
      mlp_1_prep_data.ipynb
      Full MLP: from images to 51App`")  --> c1[Visualize 51Dataset]
    end
```

- Updated Flowchart (image)
![alt Flowchart](vis/vis_mermaid_51Flowchart_SampleInvoices250.PNG)

### Updated: Mermaid Diagram Flowchart for `Sample Invoices 250`
- Mermaid Flowchart
```mermaid
  flowchart LR
    a1("`**Azure File Storage**
    Store PDFs`") --> a2("`**Azure ML**
    aml_mlp_0_data.ipynb
    Get PDFs from storage`")

    subgraph "**Azure Images Dataset**"
      a2("`**Azure ML**
      aml_mlp_0_data.ipynb
      Get PDFs from storage`") --> a3("`**Azure ML**
      aml_mlp_1_prep_data.ipynb
      Convert PDFs to images + Create Metadata`")
    end

  subgraph "**On-prem Full MLP + 51App**"
      a3("`**Azure ML**
      aml_mlp_1_prep_data.ipynb
      Convert PDFs to images + Create Metadata`") --> b1("`**WLaptop VSCode**
      mlp_1_full.ipynb
      Full MLP: from images to 51App`")
      b1("`**WLaptop VSCode**
      mlp_1_full.ipynb
      Full MLP: from images to 51App`") --> b2[Export 51Dataset]
      b1("`**WLaptop VSCode**
      mlp_1_full.ipynb
      Full MLP: from images to 51App`") --> b3[Visualize 51Dataset]
      b2[Export 51Dataset] --> b3[Visualize 51Dataset]
    end       
```
## License
Refer to license https://github.com/voxel51/fiftyone/blob/develop/LICENSE
