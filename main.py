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
    parser.add_argument(
        "--sc", action='store_true',
        help="Use a single cpu core.")

    # python.exe .\main.py -c "data/corpus.txt" -l "data/labels.txt"
    args = vars(parser.parse_args())
    corpus_path = args['corpus']
    labels_path = args['labels']
    singlecore = args['sc']

    print("Loading data...")
    corpus = load_corpus(corpus_path)
    labels = load_labels(labels_path)
    true_k = len(np.unique(labels))

    # Preprocessing
    print("Pre-processing corpus...")
    print("This could take a while!")
    start = time.time()
    if singlecore:
        processed_corpus = process_corpus(corpus)
    else:
        print("Using {} cores".format(cpu_count()))
        processed_corpus = multiprocess_corpus(corpus)
    end = time.time()
    print("Elapsed time: {} sec".format(int(end-start)))

    # Vectorization and SVD
    print("Computing Vectorization...")
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

    print("Completed!\n")


if __name__ == "__main__":
    main(sys.argv[1:])
