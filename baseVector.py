"""
Módulo responsável por criar, persistir, carregar e reconstruir a base
vetorial (ChromaDB) usada pelo agente RAG, com embeddings do Google Gemini.
"""

from pydantic import SecretStr
import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from my_keys import GEMINI_API_KEY
from readPDFs import carregar_pdfs, dividir_em_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "univesp"
EMBEDDING_MODEL = "gemini-embedding-001"


def _obter_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Instancia o modelo de embeddings do Gemini usando a chave configurada em my_keys.py."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY não configurada. Verifique se o arquivo .env "
            "existe na raiz do projeto e contém a variável GEMINI_API_KEY."
        )
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=SecretStr(GEMINI_API_KEY))


def _base_existe(persist_directory: Path = CHROMA_DIR) -> bool:
    """Verifica se já existe uma base persistida em disco (pasta presente e não vazia)."""
    return persist_directory.exists() and any(persist_directory.iterdir())


def criar_ou_carregar_base(persist_directory: Path = CHROMA_DIR) -> Chroma:
    """
    Carrega a base vetorial existente em disco, ou cria uma nova a partir
    dos PDFs em 'pdfs/' caso ainda não exista.

    Esta é a função que main.py deve chamar ao iniciar a aplicação —
    evita reprocessar os PDFs (e gastar chamadas de API de embeddings)
    toda vez que o Streamlit é aberto.
    """
    embeddings = _obter_embeddings()

    if _base_existe(persist_directory):
        logger.info("Base vetorial encontrada em '%s' — carregando do disco.", persist_directory)
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )

    logger.info("Nenhuma base encontrada em '%s' — criando a partir dos PDFs.", persist_directory)
    return _construir_base(embeddings, persist_directory)


def reconstruir_base(persist_directory: Path = CHROMA_DIR) -> Chroma:
    """
    Força a reconstrução completa da base vetorial: apaga a base existente
    (se houver) e reprocessa todos os PDFs do zero.

    Usada pelo botão "Reconstruir base" do Streamlit (main.py), por exemplo
    após trocar um PDF ou ajustar o chunking em readPDFs.py.
    """
    if persist_directory.exists():
        logger.info("Removendo base existente em '%s' antes de reconstruir.", persist_directory)
        shutil.rmtree(persist_directory)

    embeddings = _obter_embeddings()
    return _construir_base(embeddings, persist_directory)


def _construir_base(embeddings: GoogleGenerativeAIEmbeddings, persist_directory: Path) -> Chroma:
    """Lê os PDFs, divide em chunks, gera embeddings e persiste no ChromaDB."""
    paginas = carregar_pdfs()
    chunks = dividir_em_chunks(paginas)

    if not chunks:
        raise RuntimeError(
            "Nenhum chunk gerado a partir dos PDFs — verifique a pasta 'pdfs/' "
            "e o mapeamento em CATEGORIAS_POR_ARQUIVO (readPDFs.py)."
        )

    logger.info(
        "Gerando embeddings e persistindo %d chunks em '%s'... "
        "(isso faz chamadas à API do Gemini, pode levar alguns segundos)",
        len(chunks),
        persist_directory,
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )
    logger.info("Base vetorial criada com sucesso.")
    return vector_store


if __name__ == "__main__":
    base = criar_ou_carregar_base()
    resultados = base.similarity_search("Como funciona o vestibular da Univesp?", k=2)

    print(f"\n{len(resultados)} resultado(s) de teste para a busca de exemplo:\n")
    for doc in resultados:
        print(f"[{doc.metadata.get('categoria')}] fonte: {doc.metadata.get('fonte')}")
        print(doc.page_content[:200])
        print()