from unittest import TestCase
from query_response.doc_search import EmbeddingGenerator, VectorDatabase


class TestEmbeddingGenerator(TestCase):
    
    def get_sentences(self):
        sentences = ['A group of kids is playing in a yard and an old man is standing in the background',
                     'A group of children is playing in the house and there is no man standing in the background',
                     'The young boys are playing outdoors and the man is smiling nearby',
                     'The kids are playing outdoors near a man with a smile',
                     'The young boys are playing outdoors and the man is smiling nearby']
        return sentences
 
    def test_embedding_generator(self):
        sentences = self.get_sentences()
        eg = EmbeddingGenerator(sentences)
        assert eg.embeddings.shape == (5, 384)

    def test_vector_database(self):
        vecdb = VectorDatabase(self.get_sentences()) 
        search_text = "A group of children is playing in a yard and an older man is standing in the background"
       
        distances, sentences = vecdb.get_nearest_neighbors(1, search_text)
        assert sentences[0] == "A group of kids is playing in a yard and an old man is standing in the background"
            
            
        