from unittest import TestCase
import pytest

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
        assert embedding.shape == (1, 384)
        assert embedding[0][0] == pytest.approx(0.051501602, 1e-5)
        assert embedding[0][37] == pytest.approx(0.079380006, 1e-5)