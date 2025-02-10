import streamlit as st
import os
import dotenv
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from io import BytesIO
import fitz  # PyMuPDF
from langchain.schema import Document  

# Load environment variables
dotenv.load_dotenv()
openai.api_key = os.environ.get("OPENAI_KEY")

# Load and process the uploaded file
def load_and_process_file(uploaded_file):
    # Read the file content into a BytesIO object
    file_bytes = BytesIO(uploaded_file.read())

    # For PDF files, use PyMuPDF (fitz) to extract text
    if uploaded_file.type == "application/pdf":
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        documents = []
        for page in pdf_document:
            text = page.get_text("text")
            # Wrap the text inside a Langchain Document object
            documents.append(Document(page_content=text.strip()))
    else:
        # For other file types (like txt or csv), handle accordingly
        documents = [Document(page_content=uploaded_file.read().decode("utf-8"))]
    
    return documents

# Embed documents and store them in FAISS for retrieval
def create_retriever(documents):
    embeddings = OpenAIEmbeddings()
    # Create a FAISS vector store for efficient similarity search
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# Initialize the QA chain with a ChatGPT model
def initialize_qa_chain(retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=retriever
    )
    return qa_chain


# streamlit frontend
st.title("TLGPT with Langchain")

uploaded_file = st.file_uploader("Upload a text file", type=["txt", "pdf"])
if uploaded_file:
    documents = load_and_process_file(uploaded_file)
    retriever = create_retriever(documents)
    qa_chain = initialize_qa_chain(retriever)

    query = st.text_input("Ask a question about your data:")
    if query:
        # Get the answer from the model
        result = qa_chain.run(query)
        st.write("Answer:", result)
