import streamlit as st
import sys
import pandas as pd

from datasets.datasets import get_sentences
from query_response.doc_search.vector_database import VectorDatabase
from query_response.doc_search.search_engine import DocumentSearchEngine


st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="HyDE Search"
)

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .css=18e3th9 {padding-top: 1rem;}
    </style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)


def get_document_search_engine():
    sentences = get_sentences()
    vecdb = VectorDatabase(sentences=sentences)
    search_engine = DocumentSearchEngine(vecdb)
    return search_engine


search_engine = get_document_search_engine()

search_text = st.text_input('HyDE Search', value='What happened in Syria?')

if search_text in ['random', 'Random']:
    search_response = {'response': '',
                       'distances': [None]*20,
                       'sentences': search_engine.get_random_sentences(num_samples=20)}
else:
    search_response = search_engine.response_using_sentences(query=search_text, k=10)

df = pd.DataFrame.from_dict({key: search_response[key] for key in ['distances', 'sentences']})
st.write(f"{search_response['response']}")

st.table(df)


