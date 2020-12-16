import re
import numpy as np
import time
import matplotlib.pyplot as plt

from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import Normalizer, scale
from sklearn.pipeline import make_pipeline
from collections import defaultdict, Counter, OrderedDict
from scipy.sparse import csc_matrix, diags
from wordcloud import WordCloud, ImageColorGenerator
from multiprocessing import Pool, cpu_count
from functools import partial


def load_corpus(filepath):
    corpus = []
    with open(filepath, 'r') as fp:
        for line in fp:
            corpus.append(line.rstrip('\n'))
    return corpus


def load_labels(filepath):
    labels = []
    with open(filepath, 'r') as fp:
        for line in fp:
            labels.append(int(line))
    return labels


def process_string(s, stemmer, lemmatizer=None):
    proc_s = str(s).lower()
    proc_s = re.sub('[^a-z0-9 ]+', ' ', proc_s)
    tokens = word_tokenize(proc_s)
    for i, word in enumerate(tokens):
        if stemmer != None:
            word = stemmer.stem(word)
        if lemmatizer != None:
            word = lemmatizer.lemmatize(word)
        tokens[i] = word
    proc_s = " ".join(tokens)
    return proc_s


def process_corpus(corpus):
    stemmer = SnowballStemmer("english")
    processed_corpus = []
    for s in tqdm(corpus):
        proc_s = process_string(s, stemmer, lemmatizer=None)
        processed_corpus.append(proc_s)
    return processed_corpus


def multiprocess_corpus(corpus):
    n_proc = cpu_count()
    stemmer = SnowballStemmer("english")

    with Pool(processes=n_proc) as pool:
        fun = partial(process_string, stemmer=stemmer)
        res = pool.map(fun, corpus)

    return res


def get_vectorized_matrix(processed_corpus):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(processed_corpus)
    return X, vectorizer


def TSVD(X, n_components, normalize=False):
    tsvd = TruncatedSVD(n_components=n_components, random_state=0)
    if normalize:
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(tsvd, normalizer)
    else:
        lsa = tsvd
    Y = lsa.fit_transform(X)
    exp_var = tsvd.explained_variance_ratio_.sum()
    print("Explained variance:", round(exp_var, 5))
    return tsvd, Y


def cluster(Y, true_k):
    km = KMeans(n_clusters=true_k, init='k-means++',
                max_iter=1000, n_init=30,
                random_state=0, verbose=0)
    km.fit(Y)

    preds = km.predict(Y)

    return km, preds


def print_statistics(km, preds, labels, Y):
    acc = metrics.accuracy_score(labels, preds)
    if acc < 0.5:
        preds = np.logical_not(preds).astype(int)
        acc = metrics.accuracy_score(labels, preds)

    print("Clustering accuracy with k=2: %0.3f" % acc)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" %
          metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(Y, km.labels_, sample_size=1000))


def visualize_clusters(vectorizer, tsvd, km):
    terms = vectorizer.get_feature_names()

    original_centroids = tsvd.inverse_transform(km.cluster_centers_)
    for i in range(original_centroids.shape[0]):
        original_centroids[i] = np.array([x for x in original_centroids[i]])
    svd_centroids = original_centroids.argsort()[:, ::-1]

    centroids_dict = {}
    for i, centroid in enumerate(svd_centroids):
        l = [terms[term_id] for term_id in centroid]
        centroids_dict[i] = l

    for i, centroid in centroids_dict.items():
        text = " ".join(centroid)
        wc = WordCloud(background_color='white', width=600,
                       height=400).generate(text)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        figname = "wordcloud_" + str(i+1) + ".png"
        plt.savefig(figname)
        plt.show()
