"""
Módulo responsável pela pipeline de recuperação: busca por similaridade na
base vetorial, filtrada por categoria, e formatação do contexto para o LLM.
"""

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from readPDFs import CATEGORIAS_POR_ARQUIVO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fonte única da verdade sobre quais categorias existem — vem de readPDFs.py,
# evitando duplicar a lista aqui e ela ficar desatualizada.
CATEGORIAS_VALIDAS = set(CATEGORIAS_POR_ARQUIVO.values())
K_PADRAO = 4


def buscar_contexto(
    vector_store: Chroma,
    pergunta: str,
    categoria: str,
    k: int = K_PADRAO,
) -> list[Document]:
    """
    Busca os chunks mais relevantes para a pergunta, restritos à categoria
    informada (filtro de metadata no ChromaDB).

    Args:
        vector_store: base vetorial já criada/carregada (ver baseVector.py).
        pergunta: pergunta do usuário (ou reformulação feita pela ferramenta).
        categoria: categoria a filtrar — ex: "faq", "vestibular", "institucional_aluno".
        k: número de chunks a retornar.

    Returns:
        Lista de Document mais relevantes dentro da categoria informada.
    """
    if categoria not in CATEGORIAS_VALIDAS:
        logger.warning(
            "Categoria '%s' não reconhecida (esperado uma de: %s). "
            "A busca provavelmente não retornará resultados.",
            categoria,
            sorted(CATEGORIAS_VALIDAS),
        )

    logger.info("Buscando contexto | categoria='%s' | k=%d | pergunta='%s'", categoria, k, pergunta)

    resultados = vector_store.similarity_search(
        query=pergunta,
        k=k,
        filter={"categoria": categoria},
    )

    if not resultados:
        logger.warning("Nenhum resultado encontrado para categoria='%s'.", categoria)

    return resultados


def formatar_contexto(documentos: list[Document]) -> str:
    """
    Formata os documentos recuperados em um bloco de texto único, pronto para
    ser inserido no prompt do LLM (ou exibido no modo Debug do Streamlit),
    indicando a fonte de cada trecho.
    """
    if not documentos:
        return "Nenhum trecho relevante foi encontrado nos documentos."

    blocos = []
    for indice, doc in enumerate(documentos, start=1):
        fonte = doc.metadata.get("fonte", "desconhecida")
        pagina = doc.metadata.get("page")
        pagina_info = f", página {pagina + 1}" if isinstance(pagina, int) else ""
        blocos.append(f"[Trecho {indice} — fonte: {fonte}{pagina_info}]\n{doc.page_content}")

    return "\n\n".join(blocos)


if __name__ == "__main__":
    from baseVector import criar_ou_carregar_base

    base = criar_ou_carregar_base()

    pergunta_teste = "Quais são as datas importantes do vestibular 2026?"
    categoria_teste = "vestibular"

    chunks_encontrados = buscar_contexto(base, pergunta_teste, categoria_teste)
    print(f"\n{len(chunks_encontrados)} chunk(s) recuperado(s):\n")
    print(formatar_contexto(chunks_encontrados))