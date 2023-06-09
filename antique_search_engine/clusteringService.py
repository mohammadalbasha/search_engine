from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# elbow
#   Sum_of_squared_distances = []
#     K = range(2,10)
#     for k in K:
#         km = KMeans(n_clusters=k, max_iter=200, n_init=10)
#         km = km.fit(docs_tfidf)
#         Sum_of_squared_distances.append(km.inertia_)
#         plt.plot(K, Sum_of_squared_distances, 'bx-')
#         plt.xlabel('k')
#     plt.ylabel('Sum_of_squared_distances')
#     plt.title('Elbow Method For Optimal k')
#     plt.show()
  

num_clusters = 200
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

def documentClustering(docs_tfidf):
    
    kmeans.fit(docs_tfidf)
    doc_clusters = kmeans.labels_
    return doc_clusters




def queryCluserting(query_tfidf):
    query_cluster = kmeans.predict(query_tfidf)[0]
    return query_cluster

