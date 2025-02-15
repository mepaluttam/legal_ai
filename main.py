import streamlit as st
import pdfplumber
import re
import numpy as np
import faiss
import os
import torch
from sentence_transformers import SentenceTransformer
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model with GPU acceleration if available
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/msmarco-distilbert-base-tas-b", device=device)

model = load_embedding_model()

# Predefined PDFs
pdf_files = [
    "constitution.pdf",
    "IMPORTANT JUDGEMENTS_OF_SUPREME_COURT_COMPLETE_BOOK.pdf",
    "ipc_act.pdf",
    "it_act_2000_updated.pdf",
    "motor_vehicle_act.pdf",
    "the_code_of_criminal_procedure_1973.pdf"
]

# Function to extract text from PDFs (caches result)
@st.cache_data
def extract_text_from_pdfs(pdf_paths):
    all_texts = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            all_texts.append(text)
    return all_texts

legal_texts = extract_text_from_pdfs(pdf_files)

# Text cleaning function
def clean_text(text):
    if text is None:  # Handle empty or None text
        return ""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s.,;:()\-]', '', text)  # Remove special characters
    return text.strip()

cleaned_texts = [clean_text(text) for text in legal_texts]

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
docs = [text_splitter.create_documents([text]) for text in cleaned_texts]
all_docs = [doc for sublist in docs for doc in sublist]

# Cache FAISS index to avoid recomputation
faiss_index_path = "data/faiss_index.pkl"

@st.cache_data
def generate_embeddings():
    if os.path.exists(faiss_index_path):
        with open(faiss_index_path, "rb") as f:
            return pickle.load(f)

    embeddings = np.array([model.encode(doc.page_content, convert_to_numpy=True) for doc in all_docs])

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index for future use
    with open(faiss_index_path, "wb") as f:
        pickle.dump(index, f)

    return index

index = generate_embeddings()

# Retrieve top K similar texts
def retrieve_top_k(query, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [all_docs[i].page_content for i in indices[0]]

# Streamlit UI
st.title("Legal AI Assistant 🇮🇳")
st.write("Ask questions about Indian laws and get AI-generated answers.")

user_query = st.text_input("Enter your legal question:")

if st.button("Get Answer"):
    if user_query:
        retrieved_text = retrieve_top_k(user_query)
        context = "\n".join(retrieved_text)

        prompt = f"Referencing Indian Laws:\n\n{context}\n\nQuestion: {user_query}\nAnswer:"

        # Generate response using Ollama (legal-llm)
        response = ollama.chat(model="legal-llm", messages=[{"role": "user", "content": prompt}])

        st.subheader("Answer:")
        st.write(response["message"]["content"])
