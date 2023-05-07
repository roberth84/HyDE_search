from unittest import TestCase

from .test_utils import get_example_sentences
from query_response.doc_search.embedding_generator import EmbeddingGenerator


class TestEmbeddingGenerator(TestCase):

    def get_example_embedding_generator(self):
        sentences = get_example_sentences()
        return EmbeddingGenerator(sentences)

    def test_embedding_generator(self):
        eg = self.get_example_embedding_generator()
        assert eg.embeddings.shape == (5, 384)

    def test_embed_sentence(self):
        eg = self.get_example_embedding_generator()
        embedding = eg.embed_sentence("This is an example sentence.")
        embedding