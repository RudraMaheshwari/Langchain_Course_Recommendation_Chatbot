import os
import pickle
from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

def build_vector_store(docs: list[Document], persist_path: str = "./faiss_store") -> FAISS:
    os.makedirs(persist_path, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    faiss_index_path = os.path.join(persist_path, "index")
    doc_store_path = os.path.join(persist_path, "doc_store.pkl")

    if os.path.exists(faiss_index_path) and os.path.exists(doc_store_path):
        try:
            print("[INFO] Loading existing FAISS index and document store...")
            with open(doc_store_path, "rb") as f:
                _ = pickle.load(f)
            return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"[WARN] Failed to load FAISS index. Rebuilding. Reason: {e}")

    print("[INFO] Creating new FAISS index from documents...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(faiss_index_path)

    with open(doc_store_path, "wb") as f:
        pickle.dump(split_docs, f)

    print("[INFO] FAISS index and document store created and saved.")
    return vectorstore
