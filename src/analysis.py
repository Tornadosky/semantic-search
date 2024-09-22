import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from wordcloud import WordCloud
from keybert import KeyBERT
from collections import defaultdict
import numpy as np

def plot_umap(embeddings):
    """Plot UMAP for paper embeddings."""
    reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.3, metric='cosine')
    umap_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1])
    plt.title("Paper Embeddings in 2D")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()

def perform_clustering(embeddings, papers, num_clusters=6):
    """Perform KMeans clustering on embeddings and add cluster labels to papers."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    papers['cluster'] = kmeans.fit_predict(embeddings)
    return papers['cluster']

def suggest_authors(papers):
    """Suggest similar authors based on Euclidean distances between their embeddings."""
    author_embeddings = defaultdict(list)
    for _, row in papers.iterrows():
        for author in row['co_authors']:
            author_embeddings[author].append(row['embedding'])

    author_mean_embeddings = {author: np.mean(embeds, axis=0) for author, embeds in author_embeddings.items()}
    author_names = list(author_mean_embeddings.keys())
    author_distance_matrix = euclidean_distances([author_mean_embeddings[author] for author in author_names])

    author_recommendations = defaultdict(list)
    for i, author in enumerate(author_names):
        distances = list(enumerate(author_distance_matrix[i]))
        distances = sorted(distances, key=lambda x: x[1])

        closest_authors = [author_names[idx] for idx, dist in distances if author_names[idx] != author]
        author_recommendations[author] = closest_authors[:5]

    return author_recommendations

def create_word_clouds(clusters, papers):
    """Generate and display word clouds for each cluster."""
    for cluster_num in range(clusters.max() + 1):
        cluster_abstracts = papers[papers['cluster'] == cluster_num]['abstract']
        cluster_text = ' '.join(cluster_abstracts)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Cluster {cluster_num + 1} Keywords')
        plt.axis('off')
        plt.show()

def extract_keywords_per_cluster(clusters, papers):
    """Extract and display keywords for each cluster."""
    kw_model = KeyBERT()

    for cluster_num in range(clusters.max() + 1):
        cluster_abstracts = papers[papers['cluster'] == cluster_num]['abstract']
        cluster_text = ' '.join(cluster_abstracts)
        keywords = kw_model.extract_keywords(cluster_text, keyphrase_ngram_range=(1, 2), stop_words='english')
        keyword_list = [kw[0] for kw in keywords]
        print(f"Cluster {cluster_num + 1} Keywords: {', '.join(keyword_list)}")
