#  in tfidfService
import tfidfService as tfidf
import json

def createInvertedIndex(tfidf_matrix):
    inverted_index = {}

    # loop overthe non-zero entries in the tfidf matrix
    for doc_index, word_index in zip(*tfidf_matrix.nonzero()):
        # retrieve the corresponding term from the vocabulary
        term = tfidf.vectorizer.get_feature_names_out()[word_index]
        # add the document ID to the list of document IDs associated with the term in the inverted index
        if term not in inverted_index:
            inverted_index[term] = []
        inverted_index[term].append(int(doc_index))
    return inverted_index


def offlineWriteIndex(inverted_index, filename):
    jsonIndex = json.dumps(inverted_index)
    indexJsonFile = open(filename, "w")
    indexJsonFile.write(jsonIndex)
    indexJsonFile.close()



def offlineReadIndex(filename):
    #indexJsonFile.close()
    # Opening JSON file
    with open(filename, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        return json_object
