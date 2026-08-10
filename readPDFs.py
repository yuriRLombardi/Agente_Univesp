"""
Módulo responsável por carregar os PDFs institucionais da UNIVESP e prepará-los
para indexação na base vetorial: leitura, atribuição de metadados (fonte e
categoria) e divisão em chunks.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDFS_DIR = Path("pdfs")

# Mapeamento explícito: nome do arquivo -> categoria usada pelas ferramentas do agente
# (ver ferramentas.py na Fase 4). Mantido como dict fixo, em vez de inferir
# automaticamente, para deixar claro e fácil de auditar qual PDF alimenta qual
# ferramenta. Para adicionar um novo PDF no futuro, basta acrescentar uma linha aqui.
CATEGORIAS_POR_ARQUIVO: dict[str, str] = {
    "FAQUnivesp.pdf": "faq",
    "Manual_do_Candidado2026.pdf": "vestibular",
    "ManualAluno.pdf": "institucional_aluno",
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 180


def carregar_pdfs(pdfs_dir: Path = PDFS_DIR) -> list[Document]:
    """
    Carrega todos os PDFs da pasta informada e atribui metadados de
    'fonte' (nome do arquivo) e 'categoria' (conforme CATEGORIAS_POR_ARQUIVO)
    a cada página carregada.

    Args:
        pdfs_dir: caminho da pasta contendo os PDFs. Padrão: 'pdfs/' na raiz.

    Returns:
        Lista de Document (uma entrada por página) com metadata enriquecida.

    Raises:
        FileNotFoundError: se a pasta de PDFs não existir.
    """
    if not pdfs_dir.exists():
        raise FileNotFoundError(f"Pasta de PDFs não encontrada: {pdfs_dir.resolve()}")

    documentos: list[Document] = []
    arquivos_pdf = sorted(pdfs_dir.glob("*.pdf"))

    if not arquivos_pdf:
        logger.warning("Nenhum PDF encontrado em %s", pdfs_dir.resolve())
        return documentos

    for caminho_pdf in arquivos_pdf:
        categoria = CATEGORIAS_POR_ARQUIVO.get(caminho_pdf.name)
        if categoria is None:
            logger.warning(
                "PDF '%s' não está mapeado em CATEGORIAS_POR_ARQUIVO — "
                "será ignorado. Adicione-o ao dicionário para incluí-lo.",
                caminho_pdf.name,
            )
            continue

        logger.info("Carregando '%s' (categoria: %s)", caminho_pdf.name, categoria)
        try:
            loader = PyPDFLoader(str(caminho_pdf))
            paginas = loader.load()
        except Exception:
            logger.exception("Falha ao carregar '%s' — arquivo será ignorado", caminho_pdf.name)
            continue

        for pagina in paginas:
            pagina.metadata["fonte"] = caminho_pdf.name
            pagina.metadata["categoria"] = categoria
            documentos.append(pagina)

    logger.info("Total de páginas carregadas: %d", len(documentos))
    return documentos


def dividir_em_chunks(
    documentos: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Divide os documentos (páginas) em chunks menores, preservando os
    metadados originais (fonte, categoria, página) em cada chunk gerado.

    Args:
        documentos: páginas carregadas por carregar_pdfs().
        chunk_size: tamanho alvo de cada chunk, em caracteres.
        chunk_overlap: sobreposição entre chunks consecutivos, em caracteres.

    Returns:
        Lista de Document já divididos em chunks, prontos para embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documentos)
    logger.info(
        "Documentos divididos em %d chunks (chunk_size=%d, overlap=%d)",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


if __name__ == "__main__":
    paginas = carregar_pdfs()
    chunks = dividir_em_chunks(paginas)

    resumo: dict[str, int] = {}
    for chunk in chunks:
        categoria = chunk.metadata.get("categoria", "desconhecida")
        resumo[categoria] = resumo.get(categoria, 0) + 1

    print("\nResumo de chunks por categoria:")
    for categoria, total in sorted(resumo.items()):
        print(f"  {categoria}: {total}")

    if chunks:
        primeiro = chunks[0]
        print(
            f"\nExemplo de chunk (fonte: {primeiro.metadata['fonte']}, "
            f"categoria: {primeiro.metadata['categoria']}):\n"
        )
        print(primeiro.page_content[:300])


