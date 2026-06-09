import streamlit as st
import os
import shutil

from create_db import create_db
from rag_pipeline import (
    load_vector_db,
    retrieve_context,
    generate_answer
)

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 PDF RAG Chatbot")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    # Remove old PDFs
    if os.path.exists("data"):
        shutil.rmtree("data")

    os.makedirs("data", exist_ok=True)

    # Save uploaded PDFs
    for file in uploaded_files:
        with open(
            os.path.join("data", file.name),
            "wb"
        ) as f:
            f.write(file.getbuffer())

    st.success("PDFs uploaded successfully!")

    if st.button("Create Knowledge Base"):

     with st.spinner("Creating Vector Database..."):

        # Release old DB if loaded
        if "vector_store" in st.session_state:
            del st.session_state["vector_store"]


        create_db()

     st.success("Knowledge Base Ready!")

st.divider()

question = st.text_input(
    "Ask a question about the uploaded PDFs"
)

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching..."):
            if "vector_store" not in st.session_state:
                st.session_state["vector_store"] = load_vector_db()

            context = retrieve_context(
            st.session_state["vector_store"],
            question
            )

            answer = generate_answer(
                question,
                context
            )

        st.subheader("Answer")
        st.write(answer)