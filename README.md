This is a test program to evaluate an implementation of Hypothetical Documents 
(https://arxiv.org/abs/2212.10496) using a simple faiss vector search and a Large 
Language Model (LLM). 

The datasets directory (hyde_search/datasets) has a datasets.py module that has 
methods for retrieving sentences from datasets. The method `get_sentences` takes 
an argument with the name of the dataset. The only dataset coded for now is the 
Sentence Text Similarity dataset. (More info found here: https://github.com/brmson/dataset-sts.)

