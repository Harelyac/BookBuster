from BookBuster import plotAnalyzer as pa
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np
from sklearn import cluster
from sklearn import metrics
import plotly.express as px
import pandas as pd

books = pa.load_pickle("/cs/usr/harelyac/PycharmProjects/needle/BookBuster/all_books_emb.pickle")
# training data

BOOK_EMBEDDING = list(books.values())[:2500]

NUM_CLUSTERS = 8
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25,
                             avoid_empty_clusters=True)
assigned_clusters = kclusterer.cluster(BOOK_EMBEDDING, assign_clusters=True)
print(assigned_clusters)

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(BOOK_EMBEDDING)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster id labels for inputted data")
print(labels)
print("Centroids data")
print(centroids)

print(
    "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print(kmeans.score(BOOK_EMBEDDING))

silhouette_score = metrics.silhouette_score(BOOK_EMBEDDING, labels, metric='euclidean')

print("Silhouette_score: ")
print(silhouette_score)

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

model = TSNE(n_components=3, random_state=0)
np.set_printoptions(suppress=True)

Y = model.fit_transform(BOOK_EMBEDDING)
Y = Y * 10
print(len(Y))
df = pd.DataFrame(Y, columns=["x", "y", "z"])
df["color"] = assigned_clusters
df["book-name"] = list(books.keys())[:2500]
fig = px.scatter_3d(df, x='x', y='y', z='z',
                    color='color', hover_name="book-name", opacity=0.5, log_x=False, log_y=False, log_z=False)
fig.show()

# plt.scatter3d(Y[:, 0], Y[:, 1], Y[:, 2], c=assigned_clusters, s=290,alpha=.5)


# for j in range(len(BOOK_EMBEDDING)):
#   plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')


plt.show()
