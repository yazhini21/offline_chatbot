import streamlit as st
import ollama
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer

# ---- SETUP ----
model_name = "llama3"  # model loaded via Ollama
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create or load a local Chroma vector store
chroma_client = chromadb.PersistentClient(path="embeddings_db")
collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

# ---- PDF LOADER ----
def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# ---- EMBEDDING AND STORAGE ----
def embed_and_store(file):
    text = extract_text_from_pdf(file)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"id_{i}" for i in range(len(chunks))]
    )

# ---- QUERY KNOWLEDGE BASE ----
def query_knowledge_base(question):
    q_embed = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embed, n_results=3)
    context = " ".join(results['documents'][0]) if results['documents'] else "No data yet."
    prompt = f"Answer the question based on the context below:\nContext: {context}\nQuestion: {question}\nAnswer:"

    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# ---- STREAMLIT UI ----
st.title("📘 Offline Research Summarizer Chatbot")
st.write("Upload PDF files to build a local knowledge base and ask questions — no API keys needed!")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    embed_and_store(uploaded_file)
    st.success("✅ PDF content embedded and stored successfully!")

question = st.text_input("💬 Ask a question about your PDFs:")
if st.button("Ask"):
    if question.strip() != "":
        answer = query_knowledge_base(question)
        st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("Please enter a question.")
