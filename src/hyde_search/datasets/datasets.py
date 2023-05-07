import requests
import pandas as pd
import os
from io import StringIO
import pickle
from typing import List


def get_sentences(dataset: str = 'dataset-sts') -> List[str]:
    """
    Retrieves sentences for predefined datasets

    :param: dataset: The name of the desired dataset.
    :return: list of sentences
    :rtype: List[str]
    :raises Exception: if the input dataset is unknown
    """

    if dataset == 'dataset-sts':
        pickle_file = "dataset-sts-sentences.pkl"
        urls = [
            'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt',
            'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/MSRpar.train.tsv',
            'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/MSRpar.test.tsv',
            'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/OnWN.test.tsv',
            'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2013/OnWN.test.tsv',
            'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2014/OnWN.test.tsv',
            'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2014/images.test.tsv',
            'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2015/images.test.tsv'
        ]
    else:
        raise ValueError("Unknown dataset!")

    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as f:
            sentences = pickle.load(f)
    else:
        sentences = []
        for url in urls:
            res = requests.get(url)
            # extract to dataframe
            data = pd.read_csv(StringIO(res.text), sep='\t', header=None, on_bad_lines='skip')
            # add to columns 1 and 2 to sentences list
            sentences.extend(data[1].tolist())
            sentences.extend(data[2].tolist())

        # remove duplicates and NaN
        sentences = [
            sentence.replace('\n', '') for sentence in list(set(sentences)) if type(sentence) is str
        ]

        with open(pickle_file.replace('pkl', 'txt'), 'w') as f:
            f.write('\n'.join(sentences))

        with open(pickle_file, 'wb') as f:
            pickle.dump(sentences, f)

    return sentences
