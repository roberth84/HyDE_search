from unittest import TestCase
from .test_utils import get_example_sentences
from query_response.doc_search.vector_database import VectorDatabase


class TestVectorDatabase(TestCase):

    def test_get_nearest_neighbors(self):
        vecdb = VectorDatabase(get_example_sentences())
        search_text = "A group of children is playing in a yard and an older man is standing in the background"

        distances, sentences = vecdb.get_nearest_neighbors(2, search_text)
        assert sentences[0] == "A group of kids is playing in a yard and an old man is standing in the background"

