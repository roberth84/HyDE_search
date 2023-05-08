import argparse

from datasets.datasets import get_sentences
from query_response.doc_search.vector_database import VectorDatabase
from query_response.doc_search.search_engine import DocumentSearchEngine


parser = argparse.ArgumentParser(
    prog='HyDE Search',
    description='Takes a question and answers based on information found in a dataset of sentences'
)
parser.add_argument('--dataset', dest='dataset', type=str, default='dataset-sts')
parser.add_argument('--num_matches', dest='num_matches', type=int, default=20)


def get_document_search_engine(dataset: str) -> DocumentSearchEngine:
    sentences = get_sentences(dataset)
    vecdb = VectorDatabase(sentences=sentences)
    search_engine = DocumentSearchEngine(vecdb)
    return search_engine


def main():
    args = parser.parse_args()

    while True:
        search_text = input("\n\nPlease enter query\n> ")
        search_engine = get_document_search_engine(args.dataset)
        search_response = search_engine.response_using_sentences(query=search_text, k=args.num_matches)

        print(f"\nAnswer: {search_response['response']}")
        print("\nCitations:")
        for i, sentence in enumerate(search_response['sentences']):
            print(f"{i}. {sentence}")


if __name__ == '__main__':
    main()