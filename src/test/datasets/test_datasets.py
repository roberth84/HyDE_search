from unittest import TestCase
import pytest
from datasets.datasets import get_sentences


class TestDatasets(TestCase):

    def test_get_sentences(self):
        # this doesn't really test anything since the sentences are pickled
        sentences = get_sentences('dataset-sts')
        assert len(sentences) == 14506
        assert sentences[0] == "Four young men are standing still and a car is exploding behind them"

        with pytest.raises(ValueError, match='Unknown dataset!'):
            get_sentences('foo')
