
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai
from dotenv import load_dotenv
import os


def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings,
    collection_name="pdf_chatbot"
    )

    return vector_store



def retrieve_context(vector_store, query):
    results = vector_store.similarity_search(query, k=30)
    # print("\n===== RETRIEVED DOCUMENTS =====\n")

    # for i, doc in enumerate(results):
    #   print(f"\nDOCUMENT {i+1}")
    #   print(doc.metadata)
    #   print(doc.page_content[:500])
    #   print("-" * 50)

    context = ""

    for i, doc in enumerate(results):
       

        source = os.path.basename(
          doc.metadata.get("source", "Unknown file")
         )
        page = doc.metadata.get("page", 0) + 1

        context += f"""
[Source {i+1}]
File: {source}
Page: {page}

Content:
{doc.page_content}
---
"""

    return context



def generate_answer(query, context):

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an assistant for answering questions based ONLY on the provided context.

Rules:
- Use only the information present in the context.
- Do not use outside knowledge.
- If answer is not found, say:
  "I could not find the answer in the provided documents."
- Give complete answers from the context.
- Do not summarize unless the user asks for a summary.
- If the context contains a list, include all important points.
-  At the end of the answer write:
  File: <filename>
  Page: <page number>

- Never write "Source 1" or "Source 2".
- Always write the actual filename.
- if you have multiple answers mention sourse file and page number in
  the next line of every answer 
Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text



def main():

    vector_store = load_vector_db()

    query = input("Ask a question: ")

    context = retrieve_context(vector_store, query)
   
    answer = generate_answer(query, context)
    print("\n========== ANSWER ==========\n")
    print(answer)


if __name__ == "__main__":
    main()