from unittest import TestCase
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
        response = ds.response_using_sentences(query, k=2)
        assert response == "The cat jumped off the couch and ran to the window."
