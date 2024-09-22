# **Scientific Papers Analysis with Embeddings and Clustering**

## **Project Overview**

This project analyzes scientific papers by extracting their abstracts, generating embeddings using `SentenceTransformer`, reducing the dimensionality using `UMAP`, and clustering the papers using `KMeans`. It also computes distances between authors based on their papers' embeddings and generates author recommendations. Furthermore, keyword extraction using `KeyBERT` is applied to cluster abstracts, and the results are visualized using word clouds.

### **Key Features:**
- **Paper Embeddings:** Generates embeddings for paper abstracts using `SentenceTransformer`.
- **Dimensionality Reduction:** Applies UMAP to reduce the embeddings to two dimensions for visualization.
- **Clustering:** Clusters papers using `KMeans` to group similar papers together.
- **Author Recommendations:** Recommends authors based on the Euclidean distances between their embeddings.
- **Keyword Extraction:** Uses `KeyBERT` to extract keywords from each cluster.
- **Visualization:** Visualizes the UMAP projection and displays word clouds for each cluster.

---

## **Table of Contents**
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Examples and Visualizations](#examples-and-visualizations)
    - UMAP Projections
    - Word Clouds
    - Author Recommendations
5. [Dependencies](#dependencies)

---

## **Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/Tornadosky/semantic-search.git
    cd semantic-search
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the `scientific-papers.json` file and place it in the project `data` directory. This file should contain the scientific papers data in JSON format.

4. Optionally, set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

---

## **Usage**

1. **Run the notebook:**
   Open the notebook in Jupyter or your preferred environment and execute the cells in order. The notebook performs the following steps:
   - Loads the scientific papers JSON data.
   - Extracts abstracts and co-authors from the data.
   - Generates sentence embeddings using `SentenceTransformer`.
   - Reduces dimensionality with UMAP and visualizes it using a scatter plot.
   - Clusters the papers using KMeans.
   - Recommends authors based on the distance between their paper embeddings.
   - Extracts keywords using `KeyBERT` and generates word clouds for each cluster.

2. **Visualizing the Results:**
   - The UMAP scatter plot of paper embeddings will be displayed.
   - Word clouds for each cluster will be generated.
   - Author recommendations will be printed in the terminal.
