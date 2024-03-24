'''
Thomas Cronin
Student ID: 23105260
'''

# Stop words taken from Zipf's law with the english language: https://www.cs.cmu.edu/~cburch/words/top.html
stop_words = {"the", "of", "and", "to", "a", "in", "is", "that", "was", "it", "for", "on", "with", "he", "be", "I",
              "by", "as", "at", "you", "are", "his", "had", "not", "this", "have", "from", "but", "which", "she",
              "they", "or", "an", "her", "were", "there", "we", "their", "been", "has", "will", "one", "all", "would",
              "can", "if", "who", "more", "when", "said", "do", "what", "about", "its", "so", "up", "into", "no", "him",
              "some", "could", "them", "only", "time", "out", "my", "two", "other", "then", "may", "over", "also",
              "new", "like", "these", "me", "after", "first", "your", "did", "now", "any", "people", "than", "should",
              "very", "most", "see", "where", "just", "made", "between", "back", "way", "many", "years", "being", "our",
              "how", "work"}


def tokenizer(text):
    text = text.split()  # removes the white spaces and converts to tokens e.g 'word','word2'
    tokens = []
    for token in text:
        tokens.append(token.strip(
            '.,?!;:'))  # most common punctiation for english from: https://www.yourdictionary.com/articles/english-punctuation-marks
        # note i kept apostrophes , hiphens and brackets of all sorts which was suggested in the notes such as O'Connel being one unit
    return tokens


# uses 'stop_words' to remove the stop words from the tokens
def remove_stop_words(tokens):
    non_stopword_tokens = []  # creates a new list for the tokens that arent stop words
    for token in tokens:  # loops through the tokens
        if token not in stop_words:  # checks if token is not a stopword
            non_stopword_tokens.append(token)  # appends the token to the non stop word list
    return non_stopword_tokens


def normalisation(tokens):
    normalised_tokens = []
    for token in tokens:
        normalised_tokens.append(token.lower())  # converts token to lowercase
        # ToDo: add more normalisation steps if time allows
    return normalised_tokens


def get_documents(file_path):
    documents = {} # stores the documents
    document_id = None # store the document ids
    doc_parts = { # stores the different parts of the document
        'title': '',
        'author': '',
        'bib': '',
        'text': ''}
    current_part = '' # stores the current part for multi-line e.g <text>

    with open(file_path, 'r') as file:
        for line in file:
            if '<docno>' in line:
                document_id = int(line[line.find('<docno>') + 7:line.find('</docno>')].strip()) # gets id of document
            # getss the current xml part that the line is in note: this will break if multiple tags in the line
            elif '<title>' in line:
                current_part = 'title'
            elif '<author>' in line:
                current_part = 'author'
            elif '<bib>' in line:
                current_part = 'bib'
            elif '<text>' in line:
                current_part = 'text'
            elif line.startswith('</doc>'): # end case
                documents[document_id] = doc_parts.copy() # copy the document details
                # Resets the docs
                document_id = None
                doc_parts = {  # stores the different parts of the document
                    'title': '',
                    'author': '',
                    'bib': '',
                    'text': '',
                    'full_text': '',
                    'doc_length': 0}
            else:
                if current_part != '' and document_id is not None:
                    doc_parts[current_part] += line + ' ' # adds the line to the document part
        for doc_id, parts in documents.items():
            parts['full_text'] = parts['title'] + " " + parts['author'] + " " + parts['bib'] + " " + parts[
                'text']  # aggregates the text
            query_tokes = tokenizer(parts['full_text'])
            query_tokes = remove_stop_words(query_tokes)  # removes stop words
            query_tokes = normalisation(query_tokes)
            parts['doc_length'] = len(query_tokes)
    return documents # the documents sorted by id's


def inverted_index(file_path):
    inverted_index = {}

    documents = get_documents(file_path) # parses the xml and returns the documents
    for doc_id, parts in documents.items(): # loop through the document items:
        full = parts['full_text']
        tokens = tokenizer(full)
        tokens = remove_stop_words(tokens) # removes stop words
        tokens = normalisation(tokens) # normalises text

        # set inverted index
        for token in tokens: # loop through the tokens
            if token not in inverted_index: # if the token doesnt exist
                inverted_index[token] = set() # set to stop duplicates
            inverted_index[token].add(doc_id) # adds the id of the document to the token key aka inverted index
            # 'token': {1,3,5}
    return inverted_index, documents

def get_inverted_index(file_path):
    ii, docs_formatted = inverted_index(file_path)
    return ii # returns the inverted index

def get_inverted_index_and_docs(file_path):
    return inverted_index(file_path)
