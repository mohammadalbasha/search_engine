from sklearn.cluster import KMeans

num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

def documentClustering(docs_tfidf):
    kmeans.fit(docs_tfidf)
    doc_clusters = kmeans.labels_
    return doc_clusters




def queryCluserting(query_tfidf):
    query_cluster = kmeans.predict(query_tfidf)[0]
    return query_cluster

