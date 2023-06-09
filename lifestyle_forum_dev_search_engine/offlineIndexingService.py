import textProcessingService as tp
import tfidfService as tfidf
import invertedIndexService as indexing

# process dataset, normalizing, tokenizing, lemmatizing, stemmatizing, remove stop words, dates processing, remove punctions 
#documents_counts = tp.processFile("lifestyle_forum_dev_search_engine/files/collection.tsv", "lifestyle_forum_dev_search_engine/files/processed-collection", 10000)

# get the processed dataset as a dictionary
dic = tp.fileToDict('lifestyle_forum_dev_search_engine/files/processed-collection', 10000)

tfidf_matrix = tfidf.tfidfVectorMatrix(dic)

inverted_index = indexing.createInvertedIndex(tfidf_matrix)

indexing.offlineWriteIndex(inverted_index, 'lifestyle_forum_dev_search_engine/files/index.json')
tfidf.offlineWriteMatrix(tfidf_matrix, 'lifestyle_forum_dev_search_engine/files/matrix.json')

