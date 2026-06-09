
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os


def create_db():

    db_path = "vector_db"

    all_docs = []
    pdf_folder = "data"

    if not os.path.exists(pdf_folder):
        print(" PDF folder not found!")
        return

    for file in os.listdir(pdf_folder):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(pdf_folder, file)

            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
        
            all_docs.extend(docs)

            print(f"Loaded: {file}")

    print(f"\nStep 1 Done: Total pages loaded = {len(all_docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(all_docs)

    print(f"Step 2 Done: Total chunks = {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Step 3 Done: Embedding model loaded")

    print("Step 4: Loading / Creating Chroma DB...")

    vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=db_path,
    collection_name="pdf_chatbot"
    )

    print(" Chroma DB updated successfully!")


if __name__ == "__main__":
    create_db()

