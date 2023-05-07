This is a test program to evaluate an implementation of Hypothetical Documents 
(https://arxiv.org/abs/2212.10496) using a simple faiss vector search and a Large 
Language Model (LLM). 

The datasets directory (src/hyde_search/datasets) has a datasets.py module that has 
methods for retrieving sentences from datasets. The method `get_sentences` takes 
an argument with the name of the dataset. The only dataset coded for now is the 
Sentence Text Similarity dataset. (More info found here: https://github.com/brmson/dataset-sts.)

The main package is query_response (src/hyde_search/query_response). This is used to 
do the query against the sentences using a vector search to get the sentences relevant 
for the query, and passing these sentences along to the query to the LLM to generate the 
response. The query_response package has two sub-packages: doc_search and llm_model. 

The doc_search subpackage (src/hyde_search/query_response/doc_search) is used to generate 
sentence embeddings for sentences, to build the vector database using the embeddings, and 
to find matching sentences. The llm_model subpackage (src/hyde_search/query_response/llm_model) 
is used to pass prompts to the LLM and get a response.
