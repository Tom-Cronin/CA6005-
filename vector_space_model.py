'''
Thomas Cronin
Student ID: 23105260
'''

import invertedIndex # imports my own inverted index code
import math # uses pythons in built math


def calc_inverted_document_frequency(inverted_index, number_of_documents):
    inverse_document_frequency = {} # creates a dictionary for the term frequencies
    for term, term_documents in inverted_index.items(): # loops through the documents and terms
        inverse_document_frequency[term] = math.log(number_of_documents/len(term_documents)) # calculates the inverse document frequency
    return inverse_document_frequency


def generateVSM(documents, inverted_index):
    idf = calc_inverted_document_frequency(inverted_index, len(documents)) # gets idf for document
    vector_space_model = {}
    for doc in documents: # loops through each document
        full = documents[doc]['full_text']
        tokens = invertedIndex.tokenizer(full)
        tokens = invertedIndex.remove_stop_words(tokens)  # removes stop words
        tokens = invertedIndex.normalisation(tokens)  # normalises text

        tokens_term_frequency = {} # used to store the term frequencys
        for token in tokens:
            tokens_term_frequency[token] = tokens_term_frequency.get(token, 0) + 1 # increments the token count for a given token

        tf_idf = {}  # stores the tf_idf values
        for token, frequency in tokens_term_frequency.items():
            term_frequency_normalised = frequency / len(tokens) # normalised to reduce bias of short and large documents
            inverse_document_frequency_value = idf.get(token, 0) # if token does not exist returns 0
            tf_idf_value = term_frequency_normalised * inverse_document_frequency_value # calculates tf_idf
            tf_idf[token] = tf_idf_value # sets the tf_idf score
        vector_space_model[doc] = tf_idf

    return vector_space_model, idf # returns the vector space model and the inverted_document_frequency


def query_preprocess(query, idf):
    tokens = invertedIndex.tokenizer(query)
    tokens = invertedIndex.remove_stop_words(tokens)  # removes stop words
    tokens = invertedIndex.normalisation(tokens)  # normalises text
    query_tf = {} # stores the query term frequency's
    for token in tokens:
        query_tf[token] = query_tf.get(token, 0) + 1  # increments the token count for a given token

    query_vector = {}
    for token, frequency in query_tf.items(): # loops through each token in query
        if token in idf: # if the token exist in the corpus
            query_vector[token] = (frequency/len(tokens)) * idf[token] # calculates tf_idf for term

    return query_vector  # returns the query vector to be used in cosine simalarity


def cosine_simalarity(vector_1, vector_2):
    dot_product =0 # stores the dot product
    for token in vector_1:
        dot_product += vector_1.get(token, 0) * vector_2.get(token, 0) # calculates the dot product per token

    sum_squared_vector_1 = 0
    sum_squared_vector_2 = 0

    for value in vector_1:
        sum_squared_vector_1 += vector_1[value] ** 2 # squares the value

    for value in vector_2:
        sum_squared_vector_2 += vector_2[value] ** 2 # squares the value

    normalized_vector_1 = math.sqrt(sum_squared_vector_1)  # gets the square root of overall vector
    normalized_vector_2 = math.sqrt(sum_squared_vector_2)  # gets the square root of overall vector

    if normalized_vector_1 == 0 or normalized_vector_2 == 0:
        return 0 # stops division by 0 error
    else:
        return dot_product / (normalized_vector_1 * normalized_vector_2) # returns the cosine similarity value


def get_document_score(document):
    return document[1] # gets the score to sort the list


def vsm_document_ranking(query, vsm, idf):
    query_vector = query_preprocess(query, idf)
    scores = {} # stores the document scores

    for doc_id, document_vector in vsm.items(): # loops through vector space model documents
        scores[doc_id] = cosine_simalarity(query_vector, document_vector) # calculates query vs all doc vectors

    return sorted(scores.items(), key=get_document_score, reverse=True)


def run_vsm(query):
    inverted_index, documents = invertedIndex.get_inverted_index_and_docs(
        'cran.all.1400.xml')  # gets inverted index and document data

    vector_space_model, idf = generateVSM(documents, inverted_index)
    rankings = vsm_document_ranking(query, vector_space_model, idf)
    return rankings
