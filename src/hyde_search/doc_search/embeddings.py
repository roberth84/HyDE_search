from .documents import DocumentCollection, Document

import os
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple
from tqdm import tqdm
import pickle


class EmbeddingGenerator:
    """
    Generates sentence embeddings using sentence-transformers
    """
    def __init__(self, sentences: List[str], embedding_type='multi-qa-MiniLM-L6-cos-v1'):

        self.model = SentenceTransformer(embedding_type)
        self.sentences = sentences       
        self.pickle_file = 'embeddings.pkl'
        
        repickle = True
        if os.path.isfile(self.pickle_file):
            print("Evaluating pickle for sentence embeddings")
            with open(self.pickle_file, "rb") as f:
                sentences, embeddings = pickle.load(f)
                if self.sentences == sentences:
                    print("Sentences in embeddings pickle match provided sentences, using pickled embeddings")
                    self.embeddings = embeddings
                    repickle = False
                else:
                    print("Sentences in embeddings pickle don't match provided sentences")
        if repickle:
            # get embeddings
            print("getting sentence embeddings")
            self.embeddings = self.get_embeddings()
            print("creating embeddings pickle")
            with open(self.pickle_file, "wb") as f:
                pickle.dump((self.sentences, self.embeddings), f)
                
    def get_embeddings(self):
        embeddings = []
        print('Calculating sentence embeddings')
        for sentence in tqdm(self.sentences):
            embedding = self.model.encode(sentence)
            embeddings.append(embedding)        
        return np.array(embeddings)
    
    def embed_sentence(self, sentence: str) -> np.ndarray:
        vector = self.model.encode(sentence)
        np_vector = np.array([vector])
        return np_vector


class VectorDatabase:
    def __init__(self, sentences: List[str]):
        self.sentences = sentences
        self.eg = EmbeddingGenerator(sentences)
        embeddings = self.eg.embeddings
        vector_dimension = embeddings.shape[1]
        self.pickle_file = 'index.pkl'

        repickle = True
        if os.path.isfile(self.pickle_file):
            print("Evaluating pickle for faiss sentences")
            with open(self.pickle_file, "rb") as f:
                sentences, serialized_index = pickle.load(f)
                if self.sentences == sentences:
                    print("sentences in faiss pickle match provided sentences, using pickled index")
                    self.index = faiss.deserialize_index(serialized_index) 
                    repickle = False
                else:
                    print("sentences in faiss pickle don't match provided sentences")
        if repickle:
            
            self.index = faiss.IndexFlatL2(vector_dimension)
            embedding_length = embeddings[0].shape[0]
            
            print("Indexing sentences for faiss")
            for embedding in tqdm(embeddings):
                self.index.add(embedding.reshape((1, embedding_length)))

            print("creating new pickle file for faiss sentences and index")
            chunk = faiss.serialize_index(self.index)
            with open(self.pickle_file, "wb") as f:
                pickle.dump((self.sentences, chunk), f)
                
    def get_nearest_neighbors(self, num_matches: int, 
                              search_sentence: List[str]) -> Tuple[List[np.float32], str]:
        
        # the vectors from multi-qa-MiniLM-L6-cos-v1 are normalized, so we probably 
        # don't need to do this
        _vector = self._embed_sentence(search_sentence)
        faiss.normalize_L2(_vector)

        # do search to get distances and num_matches nearest neighbors
        distances, ann = self.index.search(_vector, k=num_matches)
        sentences = [self.eg.sentences[i] for i in ann[0]]
        return distances[0], sentences
        
    def _embed_sentence(self, sentence: str) -> np.ndarray:
        return self.eg.embed_sentence(sentence)