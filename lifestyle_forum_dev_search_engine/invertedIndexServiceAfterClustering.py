import pickle

def docIndexing(doc_clusters):
    doc_index = {}
    for doc_id, cluster_id in enumerate(doc_clusters):
        if cluster_id not in doc_index:
            doc_index[cluster_id] = []
        doc_index[cluster_id].append(doc_id)
    return doc_index


def offlineWriteIndex(inverted_index, filename):
    
    invertedIndexJsonFile = open(filename, "wb")
    pickle.dump(inverted_index, invertedIndexJsonFile)

    invertedIndexJsonFile.close()




def offlineReadIndex(filename):
 with open(filename, 'rb') as openfile:
        index = pickle.load(openfile)
        return index

