from query_response.doc_search.vector_database import VectorDatabase
from query_response.llm_model.llm_model import LLM_Model
from typing import List, Tuple
import random

class DocumentSearchEngine:
    def __init__(self, vecdb: VectorDatabase, llm: LLM_Model = None):
        """
        DocumentSearchEngine is used to get answers from a LLM to questions using
        relevant sentences from a database
        :param: vecdb: The vector database
        :param: llm: The large language model
        """
        self.vecdb = vecdb
        self.llm = llm if llm is not None else LLM_Model()

    def response_using_sentences(self, query: str, k: int = 5) -> Tuple[str, Tuple[List[float], List[str]]]:
        """
        Gets response for a query from a LLM using the top k sentences from the vector database.

        :param: query: The query to pass to the LLM along with the k sentences
        :param: k: The number of sentences to select from the vector database
        :return: A dict with the answer from the LLM, a list of distances for the selected
         sentences from the vector search, and the list of selected sentences
        """

        query_embedding = self.vecdb.eg.embed_sentence(query)
        distances, indices = self.vecdb.index.search(query_embedding, k)
        top_k_sents = [self.vecdb.sentences[i] for i in indices[0]]
        prompt = self._create_prompt(query, top_k_sents)
        response = {
            'response': self.llm.get_response(prompt),
            'distances': distances[0].tolist(),
            'sentences': top_k_sents
        }
        return response

    def get_random_sentences(self, num_samples=10):
        indices = random.sample(range(len(self.vecdb.eg.sentences)), k=num_samples)
        return [self.vecdb.sentences[i] for i in indices]

    @staticmethod
    def _create_prompt(query: str, context: List[str]) -> str:
        """
        Internal method to create a prompt using the query and the list of sentences.

        :param: query: The question to ask using the provided sentences.
        :param: context: The list of sentences that will be passed to the LLM to answer the question
        :return: The prompt to pass to the LLM.
        """

        context_str = '\n'.join(context)
        prompt = f"""
        You are an expert student. Answer the question using only the background information provided.
        Background: {context_str}
        
        Question: {query}
        Answer:
        """
        return prompt