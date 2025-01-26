import requests
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader  # Use this for PDF parsing
import json

load_dotenv()

BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = "ac1e5228-87cb-48c4-8bac-9a8e5a00c1dd"
FLOW_ID = "316f48e5-bf3c-4b30-8e68-d708f3e3de8a"
APPLICATION_TOKEN = os.environ.get("langflow_token")
ENDPOINT = "vdev0"



import requests
import os
import json

def insert_document(document: dict):
    """Insert a document into the Astra DB collection."""
    base_url = f"https://{os.environ['ASTRA_DB_ID']}-{os.environ['ASTRA_DB_REGION']}.apps.astra.datastax.com"
    endpoint = f"/api/rest/v2/namespaces/{os.environ['ASTRA_DB_KEYSPACE']}/collections/your_collection_name"
    url = base_url + endpoint

    headers = {
        "X-Cassandra-Token": os.environ['ASTRA_DB_APPLICATION_TOKEN'],
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=json.dumps(document))

    if response.status_code == 201:
        print("Document inserted successfully.")
    else:
        print(f"Failed to insert document: {response.json()}")


def run_flow(message: str) -> dict:
    """Send a message to Langflow API and return the response."""
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{ENDPOINT}"

    payload = {
        "input_value": message,
        "output_type": "chat",
        "input_type": "chat",
    }

    headers = {"Authorization": "Bearer " + APPLICATION_TOKEN, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    print(response)
    return response.json()

def parse_pdf(file) -> str:
    """
    Parse the content of a PDF file and return the extracted text.
    """
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def main():
    st.title("TLGPT")

    # Chat input
    message = st.text_area("Message", placeholder="Ask a question")
    if st.button("Submit"):
        if not message:
            st.error("Please enter a message")
            return
        try:
            with st.spinner("Running the flow"):
                result = run_flow(message)
            response = result["outputs"][0]["outputs"][0]["results"]["message"]["text"]
            st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return 

    # Sidebar file upload
    with st.sidebar:
        pdf_uploader = st.file_uploader("Upload a PDF file", type=["pdf"])
        if pdf_uploader:
            try:
                with st.spinner("Processing the PDF..."):
                    file_content = parse_pdf(pdf_uploader)
                    st.success("File uploaded and parsed successfully!")

                    # Prepare the document to insert
                    document = {
                        "file_name": pdf_uploader.name,
                        "content": file_content,
                        # Add any other metadata or fields as needed
                    }

                    # Insert the document into Astra DB
                    insert_document(document)
                    st.success("Document inserted into Astra DB successfully!")

                    # Optionally, send parsed content to Langflow
                    with st.spinner("Sending the file content to Langflow..."):
                        result = run_flow(file_content[:500])  # Send a snippet to Langflow
                        response = result["outputs"][0]["outputs"][0]["results"]["message"]["text"]
                        st.text_area("Langflow Response", value=response, height=300)
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()
