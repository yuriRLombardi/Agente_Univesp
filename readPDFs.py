from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def carregar_pdfs():
    loader = PyPDFLoader("pdfs/ManualAluno.pdf")
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    return pages


