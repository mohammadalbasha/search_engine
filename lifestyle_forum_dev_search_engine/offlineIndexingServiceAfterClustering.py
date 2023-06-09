import clusteringService as clustering
import invertedIndexServiceAfterClustering as  indexing
import textProcessingService as tp
import tfidfService as tfidf

# process dataset, normalizing, tokenizing, lemmatizing, stemmatizing, remove stop words, dates processing, remove punctions 
#documents_counts = tp.processFile("lifestyle_forum_dev_search_engine/files/collection.tsv", "lifestyle_forum_dev_search_engine/files/processed-collection2", 368893)
#documents_counts = tp.processFile("lifestyle_forum_dev_search_engine/files/collection.tsv", "lifestyle_forum_dev_search_engine/files/processed-collection3", 10000)

# get the processed dataset as a dictionary
dic = tp.fileToDict('lifestyle_forum_dev_search_engine/files/processed-collection2', 268892)

tfidf_matrix = tfidf.tfidfVectorMatrix(dic)

documents_clusters = clustering.documentClustering(tfidf_matrix)

inverted_index = indexing.docIndexing(documents_clusters)

indexing.offlineWriteIndex(inverted_index, 'lifestyle_forum_dev_search_engine/files/index2.json')

tfidf.offlineWriteMatrix(tfidf_matrix, 'lifestyle_forum_dev_search_engine/files/matrix2.json')

