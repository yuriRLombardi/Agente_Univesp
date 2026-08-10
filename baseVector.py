"""
Módulo responsável por criar, persistir, carregar e reconstruir a base
vetorial (ChromaDB) usada pelo agente RAG, com embeddings do Google Gemini.
"""

from pydantic import SecretStr
import logging
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from my_keys import GEMINI_API_KEY
from readPDFs import carregar_pdfs, dividir_em_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "univesp"
EMBEDDING_MODEL = "gemini-embedding-001"

# Controle de taxa para respeitar o limite do tier gratuito do Gemini
# (100 requisições de embedding por minuto). Lotes pequenos + pausa entre
# eles mantêm a taxa efetiva em ~90 req/min, com margem de segurança.
BATCH_SIZE = 15
BATCH_DELAY_SECONDS = 10
MAX_RETRIES = 5


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
    Força a reconstrução completa da base vetorial: apaga a coleção existente
    (se houver) e reprocessa todos os PDFs do zero.

    A remoção é feita via delete_collection() da API do Chroma, não apagando
    arquivos do disco diretamente — isso evita PermissionError no Windows,
    já que o SQLite usado internamente pelo Chroma mantém um arquivo aberto
    enquanto houver uma conexão ativa (ex: a base já carregada em
    st.session_state no main.py).

    Usada pelo botão "Reconstruir base" do Streamlit (main.py), por exemplo
    após trocar um PDF ou ajustar o chunking em readPDFs.py.
    """
    embeddings = _obter_embeddings()

    if _base_existe(persist_directory):
        logger.info("Removendo coleção '%s' existente antes de reconstruir.", COLLECTION_NAME)
        base_existente = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )
        base_existente.delete_collection()

    return _construir_base(embeddings, persist_directory)


def _construir_base(embeddings: GoogleGenerativeAIEmbeddings, persist_directory: Path) -> Chroma:
    """Lê os PDFs, divide em chunks e persiste os embeddings no ChromaDB em lotes."""
    paginas = carregar_pdfs()
    chunks = dividir_em_chunks(paginas)

    if not chunks:
        raise RuntimeError(
            "Nenhum chunk gerado a partir dos PDFs — verifique a pasta 'pdfs/' "
            "e o mapeamento em CATEGORIAS_POR_ARQUIVO (readPDFs.py)."
        )

    logger.info(
        "Criando base vazia e adicionando %d chunks em lotes de %d "
        "(pausa de %ds entre lotes, para respeitar o limite de requisições da API)...",
        len(chunks),
        BATCH_SIZE,
        BATCH_DELAY_SECONDS,
    )

    # Cria a coleção vazia primeiro (não faz chamadas de embedding ainda).
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    _adicionar_em_lotes(vector_store, chunks)

    logger.info("Base vetorial criada com sucesso.")
    return vector_store


def _adicionar_em_lotes(vector_store: Chroma, chunks: list[Document]) -> None:
    """Adiciona os chunks ao Chroma em lotes pequenos, com pausa entre eles."""
    total = len(chunks)
    for inicio in range(0, total, BATCH_SIZE):
        lote = chunks[inicio : inicio + BATCH_SIZE]
        _adicionar_lote_com_retry(vector_store, lote)
        logger.info("Lote %d–%d de %d processado.", inicio + 1, inicio + len(lote), total)
        if inicio + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY_SECONDS)


def _adicionar_lote_com_retry(vector_store: Chroma, lote: list[Document], tentativa: int = 1) -> None:
    """
    Tenta adicionar um lote de chunks à base. Se a API retornar erro de quota
    excedida (RESOURCE_EXHAUSTED / 429), espera progressivamente mais e tenta
    novamente, até MAX_RETRIES vezes.
    """
    try:
        vector_store.add_documents(lote)
    except Exception as erro:
        if "RESOURCE_EXHAUSTED" in str(erro) and tentativa <= MAX_RETRIES:
            espera = 20 * tentativa
            logger.warning(
                "Quota da API excedida (tentativa %d/%d). Aguardando %ds antes de tentar novamente...",
                tentativa,
                MAX_RETRIES,
                espera,
            )
            time.sleep(espera)
            _adicionar_lote_com_retry(vector_store, lote, tentativa + 1)
        else:
            raise


if __name__ == "__main__":
    base = criar_ou_carregar_base()
    resultados = base.similarity_search("Como funciona o vestibular da Univesp?", k=2)

    print(f"\n{len(resultados)} resultado(s) de teste para a busca de exemplo:\n")
    for doc in resultados:
        print(f"[{doc.metadata.get('categoria')}] fonte: {doc.metadata.get('fonte')}")
        print(doc.page_content[:200])
        print()