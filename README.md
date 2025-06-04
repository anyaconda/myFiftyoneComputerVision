# myFiftyOne and Computer Vision
Voxel51 Computer Vision - embeddings, dimensionality reduction, clustering, vis and more

## Description
Visualizing Data with Dimensionality Reduction Techniques

Using 51 walkthrough, run dimensionality reduction techniques (PCA, t-SNE, UMAP) on shipping docs in FiftyOne!

MLP: get images from a dir -> create 51dataset w/ images, embeddings, and PCA dim reductions -> export 51dataset

## License
For open source projects, say how it is licensed:  
Refer to license https://github.com/voxel51/fiftyone/blob/develop/LICENSE

## Project status
POC stage, running compute intensive tasks on GPU in Azure Machine Learning Studio and then viewing the results locally in FiftyOne tool.

## Solution design
Flowchart  
![alt Flowchart](vis_mermaid_51Flowchart.PNG)

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
### Updated: Mermaid Diagram Flowchart for `Sample Invoices 250`

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

