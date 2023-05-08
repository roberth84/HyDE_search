import os
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
        """
        :param: sentences: The sentences to use for the vector database
        :param: embedding_type: The embedding type, must be a sentence-transformers embedding
        """
        self.model = SentenceTransformer(embedding_type)
        self.sentences = sentences       
        self.pickle_file = 'embeddings.pkl'
        
        repickle = True
        if os.path.isfile(self.pickle_file):
            with open(self.pickle_file, "rb") as f:
                sentences, embeddings = pickle.load(f)
                if self.sentences == sentences:
                    self.embeddings = embeddings
                    repickle = False
        if repickle:
            # get embeddings
            print("getting sentence embeddings")
            self.embeddings = self._get_embeddings()
            print("creating embeddings pickle")
            with open(self.pickle_file, "wb") as f:
                pickle.dump((self.sentences, self.embeddings), f)
                
    def _get_embeddings(self) -> np.ndarray:
        """
        Internal method to embed self.sentences
        :return: A numpy array of embeddings with shape (nsentences, embedding_length)
        """
        embeddings = []
        print('Calculating sentence embeddings')
        for sentence in tqdm(self.sentences):
            embedding = self.model.encode(sentence)
            embeddings.append(embedding)        
        return np.array(embeddings)
    
    def embed_sentence(self, sentence: str) -> np.ndarray:
        """
        Method that takes a sentence and returns the embedding.

        :param: sentence: The sentence to embed
        :return: The embedding as a numpy array
        """

        vector = self.model.encode(sentence)
        np_vector = np.array([vector])
        return np_vector
