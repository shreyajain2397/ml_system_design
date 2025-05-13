import streamlit as st
import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    if not corpus or not query:
        raise ValueError("Corpus and query cannot be empty")

    doc_lengths = [len(doc) for doc in corpus]
    avg_doc_length = np.mean(doc_lengths)
    doc_term_counts = [Counter(doc) for doc in corpus]
    doc_freq = Counter()

    for doc in corpus:
        doc_freq.update(set(doc))

    scores = np.zeros(len(corpus))
    N = len(corpus)

    for term in query:
        df = doc_freq.get(term, 0) + 1
        idf = np.log((N + 1) / df)

        for idx, term_counts in enumerate(doc_term_counts):
            if term not in term_counts:
                continue
            tf = term_counts[term]
            doc_len_norm = 1 - b + b * (doc_lengths[idx] / avg_doc_length)
            term_score = (tf * (k1 + 1)) / (tf + k1 * doc_len_norm)
            scores[idx] += idf * term_score

    return np.round(scores, 3)

def highlight_terms(doc, query):
    highlighted_doc = ' '.join([
        f"<span style='background-color: yellow'>{word}</span>" if word in query else word 
        for word in doc
    ])
    return highlighted_doc

st.title("BM25 Document Ranking")

st.markdown("### Enter documents (one per line):")
corpus_input = st.text_area("Corpus", "the quick brown fox\nthe lazy dog\nthe fox jumped over the lazy dog")

st.markdown("### Enter query (space-separated):")
query_input = st.text_input("Query", "fox lazy")

if st.button("Compute BM25 Scores"):
    try:
        corpus = [line.strip().split() for line in corpus_input.strip().split("\n")]
        query = query_input.strip().split()

        scores = calculate_bm25_scores(corpus, query)

        ranked_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        st.subheader("Ranking of Documents based on BM25 Score:")
        for rank, (idx, score) in enumerate(ranked_docs, 1):
            st.markdown(f"**Rank {rank}: Document {idx + 1}** - BM25 Score: {score}")
            
            # Highlight terms in the document
            highlighted_doc = highlight_terms(corpus[idx], query)
            st.markdown(f"<div>{highlighted_doc}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
