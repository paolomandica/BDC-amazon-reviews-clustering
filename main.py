import argparse
import sys
from functions import *


def main(argv):

    parser = argparse.ArgumentParser(
        description="Perform clustering on your data.")
    parser.add_argument(
        "-c", "--corpus", required=True, metavar='corpus',
        help="Path of the .txt file containing the corpus.")
    parser.add_argument(
        "-l", "--labels", required=True, metavar='labels',
        help="Path of the .txt file containing the labels.")

    # python.exe .\main.py -c "corpus.txt" -l "labels.txt"
    args = vars(parser.parse_args())
    corpus_path = args['corpus']
    labels_path = args['labels']

    print("Loading data...")
    corpus = load_corpus(corpus_path)
    labels = load_labels(labels_path)
    true_k = len(np.unique(labels))

    # Preprocessing
    print("Pre-processing corpus...")
    print("This could take a while!")
    processed_corpus = process_corpus(corpus)

    # Vectorization and SVD
    print("Computing Count Vectorization...")
    X, vectorizer = get_vectorized_matrix(processed_corpus)
    print("Computing SVD...")
    tsvd, Y = TSVD(X, n_components=2, normalize=True)

    # Clustering
    print("\nStart clustering...")
    km, preds = cluster(Y, true_k)

    print("Here are some statistics from the clustering:")
    print_statistics(km, preds, labels, Y)

    print("\nLet's visualize the clusters using WordCloud!")
    visualize_clusters(vectorizer, tsvd, km)

    print("Completed!")


if __name__ == "__main__":
    main(sys.argv[1:])
