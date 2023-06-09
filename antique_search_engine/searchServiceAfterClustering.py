from flask import Flask, request, jsonify
from flask_cors import CORS
import textProcessingService as tp
import tfidfService as tfidf
from scipy.sparse import csr_matrix
import ir_datasets
import clusteringService as clustering
import numpy as np
import invertedIndexServiceAfterClustering as  indexing


#from nltk.corpus import wordnet
# def expand_query(query):
#     synonyms = set()
#     for syn in wordnet.synsets(query):
#         for lemma in syn.lemmas():
#             synonyms.add(lemma.name())
#     print(query, synonyms)
#     return list(synonyms)

# query = "neurla networks artificial inteillegence"
# expanded_query = expand_query(query)
# print(expanded_query)




app = Flask(__name__)
CORS(app)

documents_as_dict = dict()
antique_dataset = ir_datasets.load("antique")
for doc in antique_dataset.docs_iter():
    documents_as_dict[doc.doc_id] = doc.text # namedtuple<doc_id, text>



processed_docs_as_dict = tp.fileToDict('antique_search_engine/files/processed-collection2', 387383)
#processed_docs_as_dict = tp.fileToDict('antique_search_engine/files/processed-collection2', 20000)

processed_documents = list(processed_docs_as_dict.values())
processed_documents_keys = list(processed_docs_as_dict.keys())


inverted_index = indexing.offlineReadIndex('antique_search_engine/files/index2.json')
tfidf_matrix = tfidf.offlineReadMatrix('antique_search_engine/files/matrix2.json')
tfidf.vectorizer.fit(processed_documents)
clustering.documentClustering(tfidf_matrix)



def search(query):

    # process query
    processed_query = tp.processQuery(query)
    processed_query_as_text = ' '.join(processed_query)
    query_tfidf = tfidf.tfidfQuery(processed_query_as_text)
    query_cluster = clustering.queryCluserting(query_tfidf)

    relevant_docs = inverted_index[query_cluster]

    doc_scores = tfidf.matchQuery(query_tfidf, tfidf_matrix[relevant_docs])

    # sorted_doc_ids = np.array(relevant_docs)[doc_scores.argsort()[::-1]]

    # result = []
    # i = 0
    # for idx in sorted_doc_ids:
    #     if i > 10:
    #         break
    #     result.append(documents_as_dict[processed_documents_keys[idx]])
    #     i+=1
    # return result
    
    result_with_idx = []
    i = 0
    for value in doc_scores:
        if value == 0:
            continue
        result_with_idx.append((relevant_docs[i], value))
        i+=1
    result_with_idx.sort(key=lambda a: a[1], reverse=True)
    result = []
    result_idxs = []
    for j in range(0, len(result_with_idx) - 1):
        idx = result_with_idx[j][0]
        rank = result_with_idx[j][1]
        result.append({"text":documents_as_dict[processed_documents_keys[idx]], "rank": rank})    

    return result



def docsIdsSearch(query):

    # process query
    processed_query = tp.processQuery(query)
    processed_query_as_text = ' '.join(processed_query)
    query_tfidf = tfidf.tfidfQuery(processed_query_as_text)
    query_cluster = clustering.queryCluserting(query_tfidf)

    relevant_docs = inverted_index[query_cluster]

    doc_scores = tfidf.matchQuery(query_tfidf, tfidf_matrix[relevant_docs])

    # sorted_doc_ids = np.array(relevant_docs)[doc_scores.argsort()[::-1]]
    # result = []
    # i = 0
    # for idx in sorted_doc_ids:
    #     if i > 10:
    #         break
    #     result.append(processed_documents_keys[idx])
    #     i+=1
    # return result
    result_with_idx = []
    i = 0
    for value in doc_scores:
        if value == 0:
            continue
        result_with_idx.append((relevant_docs[i], value))
        i+=1
    result_with_idx.sort(key=lambda a: a[1], reverse=True)
    result = []
    for j in range(0, len(result_with_idx) - 1):
        idx = result_with_idx[j][0]
        result.append(processed_documents_keys[idx])    
    return result



@app.route("/search", methods=['POST','GET'])
def processData():
    
    if (request.method == 'POST'):
        #query = request.form['query']
        query = request.json["query"]
        return search(query)

if (__name__) == "__main__":
    app.run()
