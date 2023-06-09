from flask import Flask, request
import textProcessingService as tp
import tfidfService as tfidf
import invertedIndexService as indexing
from scipy.sparse import csr_matrix
import ir_datasets

app = Flask(__name__)




documents_as_dict = dict()
lifestyleForum_dataset = ir_datasets.load("lotte/lifestyle/dev/forum")
for doc in lifestyleForum_dataset.docs_iter():
    documents_as_dict[doc.doc_id] = doc.text # namedtuple<doc_id, text>



processed_docs_as_dict = tp.fileToDict('lifestyle_forum_dev_search_engine/files/processed-collection', 10000)
processed_documents = list(processed_docs_as_dict.values())
processed_documents_keys = list(processed_docs_as_dict.keys())


inverted_index = indexing.offlineReadIndex('lifestyle_forum_dev_search_engine/files/index.json')
tfidf_matrix = tfidf.offlineReadMatrix('lifestyle_forum_dev_search_engine/files/matrix.json')
tfidf.vectorizer.fit(processed_documents)
#tfidf_matrix = tfidf.tfidfVectorMatrix(processed_docs_as_dict)

def search(query):
    # process query
    processed_query = tp.processQuery(query)
    
    # get related document from inverted index
    result_set = set()
    for token in processed_query:
        if (token in inverted_index):
            docs_idxs = inverted_index[token]
            for doc_idx in docs_idxs:
                result_set.add(doc_idx)

    # get tfidf vector for each docuemnt
    matched_matrix = csr_matrix((tfidf_matrix.shape[0], tfidf_matrix.shape[1]))

    i = 0
    for doc_idx in result_set:
        row = tfidf_matrix.getrow(doc_idx)
        matched_matrix[i,:] = row
        i += 1



    # get tfidf vector for the query
    processed_query_as_text = ' '.join(processed_query)
    tfidf_query = tfidf.tfidfQuery(processed_query_as_text)

   # get related document with cousine similarity
    result = tfidf.matchQuery(tfidf_query, matched_matrix)
 
    # result_set = [0, 4, 3, 5]
    # result = [0.4, 0.2, 0.8, 1]
    # [(0, 0.4), (4, 0.2) ... (5,1)]
    # [(5,1), (3, 0.8), (0, 0.4), (4, 0.2)]

    # sort results based on cousine similarity
    result_with_idx = []
    i = 0
    for value in result:
        if value == 0:
            continue
        result_with_idx.append((result_set.pop(), value))
        i+=1
    result_with_idx.sort(key=lambda a: a[1], reverse=True)
    result = []
    result_idxs = []
    #for j in range(min (10, len(result_with_idx) - 1)):
    for j in range(0, min (10000000, len(result_with_idx) - 1)):
        result_idxs.append(result_with_idx[j][0])
    for idx in result_idxs:
        result.append(documents_as_dict[processed_documents_keys[idx]])    
    return result


def docsIdsSearch(query):
    # process query
    processed_query = tp.processQuery(query)
    
    # get related document from inverted index
    result_set = set()
    for token in processed_query:
        if (token in inverted_index):
            docs_idxs = inverted_index[token]
            for doc_idx in docs_idxs:
                result_set.add(doc_idx)

    # get tfidf vector for each docuemnt
    matched_matrix = csr_matrix((tfidf_matrix.shape[0], tfidf_matrix.shape[1]))

    i = 0
    for doc_idx in result_set:
        row = tfidf_matrix.getrow(doc_idx)
        matched_matrix[i,:] = row
        i += 1



    # get tfidf vector for the query
    processed_query_as_text = ' '.join(processed_query)
    tfidf_query = tfidf.tfidfQuery(processed_query_as_text)

   # get related document with cousine similarity
    result = tfidf.matchQuery(tfidf_query, matched_matrix)
 
    # result_set = [0, 4, 3, 5]
    # result = [0.4, 0.2, 0.8, 1]
    # [(0, 0.4), (4, 0.2) ... (5,1)]
    # [(5,1), (3, 0.8), (0, 0.4), (4, 0.2)]

    # sort results based on cousine similarity
    result_with_idx = []
    i = 0
    for value in result:
        if value == 0:
            continue
        result_with_idx.append((result_set.pop(), value))
        i+=1
    result_with_idx.sort(key=lambda a: a[1], reverse=True)
    result = []
    result_idxs = []
    #for j in range(min (10, len(result_with_idx) - 1)):
    for j in range(0, min (1000000, len(result_with_idx) - 1)):
        result_idxs.append(result_with_idx[j][0])
    for idx in result_idxs:
        result.append(processed_documents_keys[idx])    
    return result






# query = "mileag hybrid better citi spend time use electr motor instead ga engin hybrid batteri pack charg call gener break instead slow tradit brake car store energi brake batteri pack light chang car move energi store batteri use move car assist engin move car citi effect pronounc frequent stop start"
# result = search(query)
# print(result)


@app.route("/search", methods=['POST','GET'])
def processData():
    if (request.method == 'POST'):
        query = request.form['query']
        return search(query)

if (__name__) == "__main__":
    app.run(port=5001)
