from unittest import TestCase
from mock import patch
from query_response.llm_model.llm_model import LLM_Model


class TestLLM_Model(TestCase):
    @patch("openai.Completion")
    def test_get_response(self, openai_completion):
        llm = LLM_Model()
        openai_completion.create.return_value = {'choices': [{'text': 'foo'}]}
        response = llm.get_response("Is it foo?")
        assert response == 'foo'
