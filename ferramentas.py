"""
Módulo responsável pelas ferramentas (tools) que o agente pode escolher para
consultar cada documento da UNIVESP.
"""

import logging

from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool

from pipelineRAG import K_PADRAO, buscar_contexto, formatar_contexto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def criar_ferramentas(vector_store: Chroma, k: int = K_PADRAO) -> list[BaseTool]:
    """
    Cria as ferramentas de consulta do agente, vinculadas à base vetorial
    informada. Cada ferramenta é fixada a uma categoria/documento específico
    (ver CATEGORIAS_POR_ARQUIVO em readPDFs.py).

    Args:
        vector_store: base vetorial já criada/carregada (ver baseVector.py).
        k: número de chunks a recuperar por consulta.

    Returns:
        Lista de ferramentas (@tool) prontas para serem passadas ao agente
        (ver agentRAG.py, Fase 5).
    """

    @tool
    def consultar_faq(pergunta: str) -> str:
        """
        Consulta o documento de Perguntas Frequentes (FAQ) da UNIVESP.
        Use esta ferramenta para dúvidas gerais e rápidas sobre o que é a
        UNIVESP, como ela funciona, e outras perguntas comuns já
        respondidas em formato de FAQ oficial.
        """
        logger.info("Ferramenta 'consultar_faq' chamada com: %s", pergunta)
        documentos = buscar_contexto(vector_store, pergunta, categoria="faq", k=k)
        return formatar_contexto(documentos)

    @tool
    def consultar_vestibular(pergunta: str) -> str:
        """
        Consulta o Manual do Candidato do Vestibular 2026 da UNIVESP.
        Use esta ferramenta para perguntas sobre o processo seletivo de
        ingresso: inscrição, datas e cronograma, provas, critérios de
        seleção, documentação exigida e regras do vestibular.
        """
        logger.info("Ferramenta 'consultar_vestibular' chamada com: %s", pergunta)
        documentos = buscar_contexto(vector_store, pergunta, categoria="vestibular", k=k)
        return formatar_contexto(documentos)

    @tool
    def consultar_manual_aluno(pergunta: str) -> str:
        """
        Consulta o Manual do Aluno da UNIVESP.
        Use esta ferramenta para perguntas sobre a vida acadêmica de quem
        já é aluno matriculado: cursos, disciplinas, estágios, regras de
        comportamento, recursos disponíveis e demais informações
        institucionais gerais. Não use para perguntas sobre o processo de
        ingresso (vestibular) — para isso, use consultar_vestibular.
        """
        logger.info("Ferramenta 'consultar_manual_aluno' chamada com: %s", pergunta)
        documentos = buscar_contexto(vector_store, pergunta, categoria="institucional_aluno", k=k)
        return formatar_contexto(documentos)

    return [consultar_faq, consultar_vestibular, consultar_manual_aluno]


if __name__ == "__main__":
    from baseVector import criar_ou_carregar_base

    base = criar_ou_carregar_base()
    ferramentas = criar_ferramentas(base)

    print(f"\n{len(ferramentas)} ferramenta(s) criada(s):\n")
    for ferramenta in ferramentas:
        primeira_linha_descricao = ferramenta.description.strip().splitlines()[0]
        print(f"- {ferramenta.name}: {primeira_linha_descricao}")

    print("\nTestando chamada direta de 'consultar_vestibular'...\n")
    resultado = ferramentas[1].invoke({"pergunta": "Quando são as provas do vestibular 2026?"})
    print(resultado)