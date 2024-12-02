import pandas as pd
import yaml
import json
import tempfile
import os
from pathlib import Path
from langchain.schema import Document
from langchain.document_loaders import  PyPDFLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, Docx2txtLoader


def process_logs(uploaded_log, file_type, file_name):
    """Process logs in CSV, JSON, or YAML format."""
    documents = []
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_log.read())
        temp_file_path = temp_file.name

    try:
        if file_type in ["application/vnd.ms-excel", "text/csv"]:
            # Using LangChain's CSVLoader
            loader = CSVLoader(file_path=temp_file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = "csv"
                doc.metadata["file_name"] = file_name

        elif file_type == "application/json":
            # Using temporary file path for JSON
            data = json.loads(Path(temp_file_path).read_text())
            documents = [
                Document(
                    page_content=json.dumps(entry, indent=2),
                    metadata={"source": "json", "file_name": file_name}
                )
                for entry in (data if isinstance(data, list) else [data])
            ]

        elif file_type == "application/x-yaml":
            # Using temporary file path for YAML
            data = yaml.safe_load(Path(temp_file_path).read_text())
            documents = [
                Document(
                    page_content=yaml.dump(entry),
                    metadata={"source": "yaml", "file_name": file_name}
                )
                for entry in (data if isinstance(data, list) else [data])
            ]
    finally:
        os.remove(temp_file_path)

    return documents


def load_text_documents(uploaded_file):
    """Load text documents as LangChain documents."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = "text"
            doc.metadata["file_name"] = uploaded_file.name
    finally:
        os.remove(temp_file_path)

    return documents


def load_word_documents(uploaded_file):
    """Load Word documents as LangChain documents."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        loader = UnstructuredWordDocumentLoader(temp_file_path)
        # loader = Docx2txtLoader(temp_file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = "word"
            doc.metadata["file_name"] = uploaded_file.name
    finally:
        os.remove(temp_file_path)

    return documents


def load_pdf_documents(uploaded_file):
    """Load PDF documents as LangChain documents."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = "pdf"
            doc.metadata["file_name"] = uploaded_file.name
    finally:
        os.remove(temp_file_path)

    return documents
