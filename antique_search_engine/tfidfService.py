
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


vectorizer = TfidfVectorizer()

def tfidfVectorMatrix(corpus):
        
    # Create a list of documents
    documents = list(corpus.values()) #{"doc_1": "hello world ", "doc2": "halamadrid"} => ['hello world', 'halamadrid'] 

    # Create a TfidfVectorizer object
    # Fit the vectorizer to the documents
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix


def offlineWriteMatrix(tfidf_matrix, filename):
    
    #jsonMatrix = json.dump(tfidf_matrix, 4)
    # jsonMatrix = jsonpickle.encode(tfidf_matrix)
    # matrixJsonFile = open("matrix.json", "w")
    # matrixJsonFile.write(jsonMatrix)
    # matrixJsonFile.close()
    matrixJsonFile = open(filename, "wb")
    pickle.dump(tfidf_matrix, matrixJsonFile)

    #matrixJsonFile.write(jsonMatrix)
    matrixJsonFile.close()




def offlineReadMatrix(filename):
 with open(filename, 'rb') as openfile:
        # Reading from json file

        # json_object = json.load(openfile)
        # matrix = jsonpickle.decode(json_object)
        # print(matrix)
        matrix = pickle.load(openfile)
        return matrix



def tfidfQuery(query):
    query_tfidf = vectorizer.transform([query])
    return query_tfidf


def matchQuery(query_tfidf, tfidf_matrix):
    result = cosine_similarity(tfidf_matrix, query_tfidf).flatten()
    return result


    