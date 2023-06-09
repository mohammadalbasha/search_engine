import textProcessingService as tp
import tfidfService as tfidf
import invertedIndexService as indexing

# process dataset, normalizing, tokenizing, lemmatizing, stemmatizing, remove stop words, dates processing, remove punctions 
documents_counts = tp.processFile("antique_search_engine/files/collection.tsv", "antique_search_engine/files/processed-collection", 20000)

# get the processed dataset as a dictionary
dic = tp.fileToDict('antique_search_engine/files/processed-collection', documents_counts)

tfidf_matrix = tfidf.tfidfVectorMatrix(dic)

inverted_index = indexing.createInvertedIndex(tfidf_matrix)

indexing.offlineWriteIndex(inverted_index, 'antique_search_engine/files/index.json')
tfidf.offlineWriteMatrix(tfidf_matrix, 'antique_search_engine/files/matrix.json')

