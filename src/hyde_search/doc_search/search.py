import requests

from doc_search.vector_database import VectorDatabase


class DocumentSearch:
    def __init__(self, query: str, vecdb: VectorDatabase):
        self.query = query
        self.vecdb = vecdb

    def search(self, k=5):
        query_embedding = self.vecdb._embed_sentence(self.query)
        distances, indices = self.vecdb.index.search(query_embedding, k)
        top_k_docs = [self.vecdb.documents[i] for i in indices[0]]
        summaries = self.summarize_context(top_k_docs)
        return summaries

    def summarize_context(self, docs):
        summaries = []
        for doc in docs:
            response = requests.post('https://api.openai.com/v1/text-davinci-003/completions',
                                     headers={'Content-Type': 'application/json',
                                              'Authorization': 'Bearer <YOUR_API_KEY>'},
                                     json={'prompt': doc,
                                           'max_tokens': 50,
                                           'temperature': 0.5})
            summary = response.json()['choices'][0]['text']
            summaries.append(summary)
        return summaries