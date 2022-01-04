"""
Nate Peters
Hannah Smith
11/8/19
ISTA 331 HW5

This module creates a matrix of similarities between contents of text files
using the cosine similarity formula.
"""

import string, pandas as pd, numpy as np, math
from sklearn.feature_extraction import stop_words 
from nltk.stem import SnowballStemmer 

def dot_product(v1, v2):
    '''
    Gets dot product of two vectors
    ---------------------------------------------
    PARAMETERS:
        v1(dict) - first vector 
        v2(dict) - second vector
    RETURNS:
        (int) - dot product
    '''
    similar_keys = [key for key in v1.keys() if key in v2.keys()]
    return(sum([v1[key]*v2[key] for key in similar_keys]))

def magnitude(v):
    '''
    Gets magnitude of a vector
    ---------------------------------------------
    PARAMETERS:
        v(dict) - vector
    RETURNS:
        (int) - magnitude
    '''
    return math.sqrt(sum(v[i]**2 for i in v.keys()))

def cosine_similarity(v1, v2):
    '''
    Gets cosine similarity of two vectors
    ---------------------------------------------
    PARAMETERS:
        v1(dict) - first vector 
        v2(dict) - second vector
    RETURNS:
        (int) - cosine similarity
    '''
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2))

def get_text(filename):
    '''
    Creates cleaned up version of contents of a file
    ---------------------------------------------
    PARAMETERS:
        filename(str) - name of file
    RETURNS:
        text(str) - cleaned up version of file
    '''
    text = open(filename).read()
    text = text.lower()
    text = text.replace("n't", "")
    for digit in "01234567879":
        text = text.replace(digit, "")
    for punc in string.punctuation:
        text = text.replace(punc, "")
    return text

def vectorize(filename, stop_words, stemmer):
    '''
    Creates a word count vector mapping tokens from
    file to word counts
    ---------------------------------------------
    PARAMETERS:
        filename(str) - name of file
        stop_words(set) - words not included as tokens
        stemmer(obj) - SnowballStemmer object
    RETURNS:
        results(dict) - dictionary mapping tokens to word counts
    '''
    results = {}
    tokens = get_text(filename)
    tokens = tokens.split()
    for token in tokens:
        word = stemmer.stem(token)
        if word not in stop_words and word != "":
            if word not in results:
                results[word] = 1
            else:
                results[word] += 1
    return results

def get_doc_freqs(vectors):
    '''
    Creates dictionary that maps each key from all of the vectors 
    to the number of files that contain the given key.
    ---------------------------------------------
    PARAMETERS:
        vectors(list) - list of all vectors
    RETURNS:
        doc_freqs(dict) - dictionary of keys mapped to frequencies of key in files
    '''
    doc_freqs = {}
    for vector in vectors:
        for key in vector.keys():
            if key not in doc_freqs:
                doc_freqs[key] = 1
            else:
                doc_freqs[key] += 1
    return doc_freqs

def tfidf(vectors):
    '''
    Replaces all the values in all the vectors with their respective
    TFIDF values (token count in document * rarity of token in corpus of documents)
    ---------------------------------------------
    PARAMETERS:
        vectors(list) - list of vectors represented as dictionaries 
    RETURNS:
        N/A
    '''
    if(len(vectors) >= 100):
        scale = 1
    else:
        scale = 100 / len(vectors)

    doc_freqs = get_doc_freqs(vectors)
    for vector in vectors:
        for key in vector.keys():
             vector[key] *= (1 + math.log2(scale * (len(vectors) / doc_freqs[key])))

def get_similarity_matrix(filenames, stop_words, stemmer):
    '''
    Creates a matrix of similarities between files.
    ---------------------------------------------
    PARAMETERS:
        filenames(list) - list of files
        stop_words(set) - words not included as tokens
        stemmer(obj) - SnowballStemmer object
    RETURNS:
        frame(DataFrame) - cosine similarities between files
    '''
    frame = pd.DataFrame(index=filenames, columns=filenames)
    vectors = []
    for file in filenames:
        text = get_text(file)
        vector = vectorize(file, stop_words, stemmer)
        vectors.append(vector)
    tfidf(vectors)
    for i in range(len(vectors)-1):
        frame.loc[filenames[i], filenames[i]] = 1
        for j in range(i+1, len(vectors)):
            c = cosine_similarity(vectors[j], vectors[i])
            frame.loc[filenames[i], filenames[j]] = c
            frame.loc[filenames[j], filenames[i]] = c
    frame.iloc[-1, -1] = 1
    return frame

def matrix_pretty_string(sim_matrix):
    '''
    Creates 'pretty' string representing similarity matrix.
    ---------------------------------------------
    PARAMETERS:
        sim_matrix(DataFrame) - frame of cosine similarities between files
    RETURNS:
        matrix_string(str) - string representing similarity matrix
    '''
    matrix_string = ""
    matrix_string += matrix_string.rjust(7)
    filenames = list(sim_matrix.columns)
    for i in range(len(filenames)):
        filenames[i] = filenames[i].strip(".txt")
        if len(filenames[i]) > 7:
            filenames[i] = filenames[i][:7]
        matrix_string += "|" + filenames[i].ljust(7)
    matrix_string += "\n" + ("-" * 7) + "|" 
    matrix_string += ("-" * 51)
    for i in range(len(sim_matrix.index)):
        matrix_string += "\n" + filenames[i].ljust(7)
        for j in range(len(sim_matrix.columns)):
            matrix_string += "|" + str(round(sim_matrix.iloc[i,j],3)).rjust(7)
    matrix_string += "\n"
    return matrix_string


def main():
    stops = {k for k in list(stop_words.ENGLISH_STOP_WORDS) + ['did', 'gone', 'ca']} 
    stemmer = SnowballStemmer('english')
    files = ['gotg2a.txt', 'gotg2b.txt', 'gotg1.txt', 'aaou.txt', 'gw.txt', 'saguaro.txt']
    similarity_matrix = get_similarity_matrix(files, stops, stemmer)
    print(matrix_pretty_string(similarity_matrix))

if __name__ == "__main__":
    main()

