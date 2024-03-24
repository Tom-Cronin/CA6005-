import math
from collections import Counter
from invertedIndex import tokenizer, remove_stop_words, normalisation, get_inverted_index_and_docs


def train_collection(documents):
    collection_counts = Counter()
    for doc_id  in documents:
        text = documents[doc_id]['full_text']
        tokens = tokenizer(text)
        tokens = remove_stop_words(tokens)  # removes stop words
        tokens = normalisation(tokens)  # normalises text

        collection_counts.update(tokens) # counts the tokens apperance in all documents
    total_collection_counts = sum(collection_counts.values()) # sums the counter to get all values
    return collection_counts, total_collection_counts

def jelinek_mercer_smoothing(token, document_counts, total_document_counts, collection_counts,
                             total_collection_counts, jelinek_mercer_smoothing_lambda):


    if total_document_counts == 0:
        total_document_counts = 1 # gets rid of divide by 0

    token_document_frequency = document_counts.get(token, 0) # gets the count of the specific token
    token_collection_probability = collection_counts.get(token,0) / total_collection_counts
    if token_collection_probability == 0:
        token_collection_probability = 0.000000000001  # gets rid of divide by 0
    return ((1 - jelinek_mercer_smoothing_lambda) * (token_document_frequency/total_document_counts)) /\
        (jelinek_mercer_smoothing_lambda * token_collection_probability)

def get_document_score(document):
    return document[1] # gets the score to sort the list

def score_documents(querys, documents, colleciton_counts, total_collection_count, jelinek_mercer_smoothing_lambda):
    query_scores = {}
    for query_id in querys:
        query_tokes = tokenizer(querys[query_id]['title'])
        query_tokes = remove_stop_words(query_tokes)  # removes stop words
        query_tokes = normalisation(query_tokes)  # normalises text
        scores = {} # stores the scores of the 1-gram lm model
        for doc_id in documents:
            text = documents[doc_id]['full_text']
            document_tokens = tokenizer(text)
            document_tokens = remove_stop_words(document_tokens)  # removes stop words
            document_tokens = normalisation(document_tokens)  # normalises text

            document_counts = Counter(document_tokens)
            total_document_length = len(document_tokens)
            score =0
            for token in query_tokes:
                token_prob = jelinek_mercer_smoothing(token, document_counts, total_document_length, colleciton_counts, total_collection_count, jelinek_mercer_smoothing_lambda)
                if token_prob > 0:
                    score += math.log(token_prob)
            scores[doc_id] =score
        query_scores[query_id] = scores
    return query_scores


def write_scores_to_file(scores, run_id):
    with open('outputs/LM_1ngram_results.txt', 'w') as file:
        for query_id, q_scores in scores.items():
            sorted_docs = sorted(q_scores.items(), key=get_document_score, reverse=True)
            rank = 1 # initiates the rank:
            for doc_id, score in sorted_docs[:100]:
                file.write(f"{query_id} 0 {doc_id} {rank} {score:.4f} {run_id}\n")
                rank += 1  # Increment rank for the next document


def run_LM_1ngram(querys, jelinek_mercer_smoothing_lambda ):
    inverted_index, documents = get_inverted_index_and_docs(
        'cran.all.1400.xml')  # gets inverted index and document data
    collection_counts, total_colleciton_count = train_collection(documents)
    scores = score_documents(querys, documents, collection_counts, total_colleciton_count, jelinek_mercer_smoothing_lambda)
    run_id = "run_LM_1ngram"
    write_scores_to_file(scores, run_id)
