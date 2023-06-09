import clusteringService as clustering
import invertedIndexServiceAfterClustering as  indexing
import textProcessingService as tp
import tfidfService as tfidf

# process dataset, normalizing, tokenizing, lemmatizing, stemmatizing, remove stop words, dates processing, remove punctions 
#documents_counts = tp.processFile("antique_search_engine/files/collection.tsv", "antique_search_engine/files/processed-collection2", 403000)
#documents_counts = tp.processFile("antique_search_engine/files/collection.tsv", "antique_search_engine/files/processed-collection3", 20000)

# get the processed dataset as a dictionary
dic = tp.fileToDict('antique_search_engine/files/processed-collection2', 387383)

tfidf_matrix = tfidf.tfidfVectorMatrix(dic)

documents_clusters = clustering.documentClustering(tfidf_matrix)

inverted_index = indexing.docIndexing(documents_clusters)

indexing.offlineWriteIndex(inverted_index, 'antique_search_engine/files/index2.json')

tfidf.offlineWriteMatrix(tfidf_matrix, 'antique_search_engine/files/matrix2.json')

