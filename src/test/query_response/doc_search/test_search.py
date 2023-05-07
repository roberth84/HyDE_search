from unittest import TestCase
from query_response.doc_search import DocumentSearch
from query_response.doc_search import VectorDatabase


class TestDocumentSearch(TestCase):

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
        ds = DocumentSearch(query, vecdb)
        response = ds.search(k=2)
        response
