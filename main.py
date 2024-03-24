import pytrec_eval
import json
import vector_space_model
from BM25 import run_bm25
from LM_1ngram import run_LM_1ngram

def get_querys():
    querys = {}  # stores the documents
    query_id = None  # store the document ids
    query_parts = {  # stores the different parts of the document
        'title': ''}
    current_part = ''  # stores the current part for multi-line e.g <text>
    with open('cran.qry.xml', 'r') as file:
        for line in file:
            if '<num>' in line:
                query_id = int(
                    line[line.find('<num>') + 5:line.find('</num>')].strip()
                )  # gets id of query
            # gets the current xml part that the line is in note: this will break if multiple tags in the line
            elif '<title>' in line:
                current_part = 'title'
            elif line.startswith('</top>'):  # end case
                querys[query_id] = query_parts.copy()  # copy the document details
                # Resets the docs
                query_id = None
                query_parts = {  # stores the different parts of the document
                    'title': ''
                }
            else:
                if current_part != '' and query_id is not None:
                    line = line.strip("</title>")
                    line = line.strip("\n")
                    query_parts[current_part] += line + ' '  # adds the line to the document part
    return querys




def vsm(querys):
    results = {}
    for q_id in querys:
        query = querys[q_id]['title']
        results[q_id] = vector_space_model.run_vsm(query)

    with open('outputs/vsm_output.txt', 'w') as output_file:
        for query_id, docs_scores in results.items():

            rank = 1  # initiates the rank:
            for doc_id, score in docs_scores[:100]:
                output_line = f"{query_id} 0 {doc_id} {rank} {score:.4f}\tvsm_result\n"
                rank += 1  # Increment rank for the next document
                output_file.write(output_line)



def LM_1ngram_run(querys):
    run_LM_1ngram(querys, 0.1)

def bm25(querys):
    run_bm25(querys)

if __name__=='__main__':
    queries = get_querys()
    vsm(queries)
    LM_1ngram_run(queries)
    bm25(queries)
