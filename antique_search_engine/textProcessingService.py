# import required packages
from flask import Flask, request
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from spacy.lang.en import English
import spacy
import datefinder
from spellchecker import SpellChecker

spell = SpellChecker()
sp = spacy.load('en_core_web_lg')
sp.add_pipe("merge_entities")


app = Flask(__name__)



# this functions to normalize text (convert to lowercase)
# input => text
# output => normalized text
def normalizeText(text):
    # # Tokenize the text into words
    # words = nltk.word_tokenize(text)

    # # Perform part-of-speech tagging
    # pos_tags = nltk.pos_tag(words)

    # # Perform named entity recognition
    # ne_tags = nltk.ne_chunk(pos_tags)

    # # Find acronyms
    # acronyms = []
    # for chunk in ne_tags:
    #     if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'NE':
    #         if len(chunk.leaves()) == 1 and chunk.leaves()[0][0].isupper():
    #             acronyms.append(chunk.leaves()[0][0])

    return text.lower().replace('.', '').replace("'", '')


def correctTerms(text):
    if not text:
        return ""
    terms = text.split(" ")
    corrected_terms = []
    for term in terms:
        # Check if the term is misspelled
        if not spell.correction(term) == term and spell.correction(term) != None:
            # If the term is misspelled, suggest a correction
            corrected_term = spell.correction(term)
        else:
            # If the term is not misspelled, leave it as is
            corrected_term = term
        corrected_terms.append(corrected_term)

    corrected_text = " ".join(corrected_terms)
    return corrected_text


def processDates(text):
    z = [jj for jj in sp(text).doc if jj.ent_type_ == "DATE"]
    for zz in z:
        tmp = list(datefinder.find_dates("default " + zz.text + " default"))
        if len(tmp) > 0:
            text += " " + sp(tmp[0].date().strftime("%Y/%m/%d")).text

    return text    


#######################################

# input => text
# ouptut => array of tokens
# this function to get the tokens of a text
def textTokenizer(text):
    return word_tokenize(text)

#######################################

# input => array of tokens
# outpout => array of tokens filtered from stop words
# this function to remove stop words 
def removeStopWords(words): 
    stop_words = set(stopwords.words('english'))
    stopWords_filtered_words = []
    # todo
    # remove each word with length <=2 or ==3 and conatins , . '
    
    # remove stop words
    for w in words:
        #if w not in stop_words and  len(w) > 2 and not w.__contains__('.') and not w.__contains__("'"):
        if not w in stop_words and len(w) > 2: 
            stopWords_filtered_words.append(w)

    # remove punction marks    
    punctionMark_filtered_words = [word for word in stopWords_filtered_words if word.isalnum()]
  
    return punctionMark_filtered_words

##################################


# input => array of words
# output => array of stemmed words
# this function to stem words
def wordsStemmer(words):
    stemmed_words = []
    ps = PorterStemmer()
    for w in words:
        stemmed_word = ps.stem(w)
        stemmed_words.append(stemmed_word)
    return stemmed_words

##################################


# input => array of words
# output => array of lemmatized words
# this function to lemmatize words
def wordsLemmatizer(words, pos_tags):
    lemmatized_words = []
    lemmatizer = WordNetLemmatizer()
    # todo
    # check if the word is noun, verb, or adjective
    #print(lemmatizer.lemmatize("run"))
    #print(lemmatizer.lemmatize("run",'v'))
    i = 0
    for w in words:
        #lemmatized_word = lemmatizer.lemmatize(w, pos_tags[i][1])
        lemmatized_word = lemmatizer.lemmatize(w)
        lemmatized_words.append(lemmatized_word)
        i+=1
    return lemmatized_words

####################################


def processFile(inputPath, outputPath, linesNumber):
    #file = open(inputPath, 'r')
    #for line in file.readlines():
    count = 0 
    writeFile = open(outputPath,'w')
    with open(inputPath, "r") as file:  
        for i, line in enumerate(file):
            # we have a pipeline to process data
            # the output from each stage is the input to the next stage
            # text => noramlized_text => tokens => filtered_tokens => stemmed_tokens => lemmetized_tokens
            #line = next(file).strip()
            if i > linesNumber: 
                break
            line = line.strip()
            #line = correctTerms(line)
            line = processDates(line)
            normalized_text = normalizeText(line)
            tokens = textTokenizer(normalized_text)
            pos_tags = nltk.pos_tag(tokens)
            filtered_tokens  = removeStopWords(tokens)
            stemmed_tokens = wordsStemmer(filtered_tokens)
            lemmatized_tokens = wordsLemmatizer(stemmed_tokens, pos_tags)
            
            if len(lemmatized_tokens) < 2:
                continue
          
            # count how many lines stored in output file
            count += 1

            # todo
            # filter dates
            
            # write doc_id to output file
            writeFile.write(tokens[0])
            writeFile.write(' ')
            
            # remove doc_id from tokens
            lemmatized_tokens.pop(0)
            
            # write document content to output file as tokens 
            text = str.join(' ', lemmatized_tokens)
            writeFile.write(text.strip() + '\n')
            # for token in lemmatized_tokens:
            #     #writeFile.write(' '.join(lemmatized_tokens))
            #     # writeFile.write(token)
            #     # writeFile.write(' ')
            #writeFile.write(str(lemmatized_tokens))

        writeFile.close()

    #return 'data file processed successfully \ndoc_id array of tokens'
    return count
#######################################

def processQuery(query):
    # we have a pipeline to process data
    # the output from each stage is the input to the next stage
    # text => noramlized_text => tokens => filtered_tokens => stemmed_tokens => lemmetized_tokens
    query = query.strip()
    query = correctTerms(query)
    query = processDates(query)
    normalized_text = normalizeText(query)
    tokens = textTokenizer(normalized_text)
    filtered_tokens  = removeStopWords(tokens)
    stemmed_tokens = wordsStemmer(filtered_tokens)
    lemmatized_tokens = wordsLemmatizer(stemmed_tokens, [])
    return lemmatized_tokens

#######################################

def fileToDict (filePath, linesNumber):
     #dic = defaultdict()
     dic = {}
     with open(filePath, "r") as file: 
        for i, line in enumerate(file):
            #line = next(file)
            if i > linesNumber:
                break      
            line = line.strip()
            line = line.split(' ')
            doc_id = line[0]
            line.pop(0)
            line = str.join(' ', line)
            dic[doc_id] = line
     return dic


    

@app.route("/", methods=['POST','GET'])
def processData():
    if (request.method == 'POST'):
        inputFilename = request.form['inputFilename']
        outputFilename = request.form['outputFilename']
        docsNumber = request.form['docsNumber']
        return processFile(inputFilename, outputFilename, docsNumber)

if (__name__) == "__main__":
    app.run()