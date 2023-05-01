from unittest import TestCase
from doc_search.documents import Document, DocumentCollection


class TestDocuments(TestCase):
    def test_document_text(self):
        doc = Document('This is a test document.')
        assert doc.text == 'This is a test document.'

    def test_document_collection_str(self):
        doc1 = Document('This is document 1.')
        doc2 = Document('This is document 2.')
        coll = DocumentCollection([doc1, doc2])
        assert coll.documents == [doc1, doc2]
        
    def test_document_split_sentences(self):
        sent1 = 'This is a test document.'
        sent2 = 'This is the second sentence.'
        doc = Document(sent1 + ' ' + sent2)
        sentences = doc.split_sentences()
        assert sentences == [sent1, sent2]
        
if __name__ == '__main__':
    pytest.main(['--capture=no'])