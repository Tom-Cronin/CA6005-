import invertedIndex
import math


def query_preprocess(query):
    tokens = invertedIndex.tokenizer(query)
    tokens = invertedIndex.remove_stop_words(tokens)
    tokens = invertedIndex.normalisation(tokens)
    return tokens

def get_document_score(document):
    return document[1] # gets the score to sort the list

def bm25_score(term_frequency, document_length, avg_doc_length, n, N, k1, b):
    idf = math.log((N - n + 0.5) / (n + 0.5) + 1) # code explination found from https://medium.com/@azunan3/understanding-tf-idf-term-frequency-inverse-document-frequency-in-python-373070acb895
    B = (1 - b) + b * (document_length / avg_doc_length) # normalization factor for term frequencies
    tf_prime = term_frequency / B
    score = idf * (tf_prime / (tf_prime + k1)) # Calculate BM25 score
    return score
def calculate_bm25_scores(inverted_index, total_docs, documents, avg_doc_length, query_tokenss, k1, b):
    bm25_scores = {} # stores the bm25 scores
    for token in query_tokenss:  # loops through query tokens
        if token in inverted_index:
            doc_list = inverted_index[token] # gets the ids of the documents for this token
            number_docs_with_term = len(doc_list) # gets the number of relevant documents

            for doc_id in doc_list:
                tf = documents[doc_id]['full_text'].count(token)
                document_length = documents[doc_id]['doc_length']
                bm25_term_score = bm25_score(tf, document_length, avg_doc_length, number_docs_with_term, total_docs, k1, b)

                if doc_id in bm25_scores:
                    bm25_scores[doc_id] += bm25_term_score
                else:
                    # Otherwise, start a new entry for this document
                    bm25_scores[doc_id] = bm25_term_score
    return bm25_scores

def write_scores_to_file(scores):
    with open('outputs/bm25_results.txt', 'w') as file:
        for score in scores:
            file.write(score)

def run_bm25(querys):
    run_id = "BM25"
    inverted_index, documents = invertedIndex.get_inverted_index_and_docs(
        'cran.all.1400.xml')  # gets inverted index and document data
    avg_doc_length = 0
    for doc in documents:
        avg_doc_length += documents[doc]['doc_length']
    avg_doc_length = avg_doc_length / len(documents)
    query_scores = []
    for query_id in querys:
        query_tokens = query_preprocess(querys[query_id]['title'])
        scores = calculate_bm25_scores(inverted_index, len(documents), documents, avg_doc_length, query_tokens, 1.5, 0.75)

        sorted_scores = sorted(scores.items(), key=get_document_score, reverse=True)
        rank = 1  # initiates the rank:
        for doc_id, score in sorted_scores[:100]:
            query_scores.append(f"{query_id} 0 {doc_id} {rank} {score:.4f} {run_id}\n")
            rank += 1  # Increment rank for the next document

    write_scores_to_file(query_scores)



