import os
import numpy as np
import faiss
from query_response.doc_search.embedding_generator import EmbeddingGenerator
from typing import List, Tuple
from tqdm import tqdm
import pickle


class VectorDatabase:
    """
    This class performs vector searches against a dataset of sentences.
    """
    def __init__(self, sentences: List[str]):
        """
        :param: sentences: The list of sentences to use as the basis of the vector database.
        """
        self.sentences = sentences
        self.eg = EmbeddingGenerator(sentences)
        embeddings = self.eg.embeddings
        vector_dimension = embeddings.shape[1]
        self.pickle_file = 'index.pkl'

        repickle = True
        if os.path.isfile(self.pickle_file):
            with open(self.pickle_file, "rb") as f:
                sentences, serialized_index = pickle.load(f)
                if self.sentences == sentences:
                    self.index = faiss.deserialize_index(serialized_index)
                    repickle = False
        if repickle:
            
            self.index = faiss.IndexFlatL2(vector_dimension)
            embedding_length = embeddings[0].shape[0]
            
            for embedding in tqdm(embeddings):
                self.index.add(embedding.reshape((1, embedding_length)))

            chunk = faiss.serialize_index(self.index)
            with open(self.pickle_file, "wb") as f:
                pickle.dump((self.sentences, chunk), f)
                
    def get_nearest_neighbors(self, num_matches: int, 
                              search_sentence: List[str]) -> Tuple[List[np.float32], str]:
        """
        Returns the sentences closest to the search sentence.

        :param: num_matches: The number of matches to return.
        :param: search_sentence: The sentence to find matches to
        :return: A tuple of distances (one for each match) and matches sentences
        """
        
        # the vectors from multi-qa-MiniLM-L6-cos-v1 are normalized, so we probably 
        # don't need to do this
        _vector = self._embed_sentence(search_sentence)
        faiss.normalize_L2(_vector)

        # do search to get distances and num_matches nearest neighbors
        distances, ann = self.index.search(_vector, k=num_matches)
        sentences = [self.eg.sentences[i] for i in ann[0]]
        return distances[0], sentences
        
    def _embed_sentence(self, sentence: str) -> np.ndarray:
        """
        Internal method that embeds a single sentence.
        :param: sentence: The input sentence
        :return: A numpy array with the embeddings.
        """
        return self.eg.embed_sentence(sentence)