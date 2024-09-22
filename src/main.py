from src.utils import load_papers, generate_embeddings
from src.analysis import plot_umap, perform_clustering, suggest_authors, create_word_clouds, extract_keywords_per_cluster

def main():
    # Define the local path and the URL for the JSON file
    json_path = 'data/scientific-papers.json'
    download_url = 'https://github.com/Tornadosky/semantic-search/releases/download/v1/scientific-papers.json'

    # Load data (will download if not present)
    papers = load_papers(json_path, download_url)

    # Generate embeddings for the abstracts
    embeddings, papers = generate_embeddings(papers)

    # Perform UMAP visualization
    plot_umap(embeddings)

    # Perform clustering
    clusters = perform_clustering(embeddings, papers)

    # Suggest authors based on embeddings
    author_recommendations = suggest_authors(papers)

    # Extract keywords and generate word clouds per cluster
    create_word_clouds(clusters, papers)
    extract_keywords_per_cluster(clusters, papers)

    # Print author recommendations
    for author, recommended in author_recommendations.items():
        print(f"Author: {author}\nRecommended: {', '.join(recommended)}\n")

if __name__ == "__main__":
    main()
