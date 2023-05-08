from unittest import TestCase
import pytest

from query_response.doc_search.search_engine import DocumentSearchEngine
from query_response.doc_search.vector_database import VectorDatabase


class TestDocumentSearchEngine(TestCase):

    def test_search(self):
        query = 'What did the cat do?'
        sentences = [
            'He smiled and waved goodbye.',
            'The cat jumped off the couch and ran to the window.',
            'She sipped her coffee and checked her emails.',
            'The phone rang but nobody answered.',
            'The flowers bloomed in the warm sunshine.'
        ]
        vecdb = VectorDatabase(sentences)
        ds = DocumentSearchEngine(vecdb)
        response_dict = ds.response_using_sentences(query, k=2)
        response = response_dict['response']
        distances = response_dict['distances']
        sentences = response_dict['sentences']

        assert response == "The cat jumped off the couch and ran to the window."

        assert distances[0] == pytest.approx(0.5437092, 1e-5)
        assert distances[1] == pytest.approx(1.3271233, 1e-5)
        assert sentences == ['The cat jumped off the couch and ran to the window.',
                             'She sipped her coffee and checked her emails.']
