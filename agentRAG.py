"""
Módulo responsável pelo agente RAG: conecta o Gemini às ferramentas de
consulta (ferramentas.py) via tool calling, e orquestra a geração da
resposta final com base no contexto recuperado.
"""

import logging
import re

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, ToolMessage

from ferramentas import criar_ferramentas
from my_keys import GEMINI_API_KEY
from my_models import GEMINI_FLASH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELO_AGENTE = f"google_genai:{GEMINI_FLASH}"

SYSTEM_PROMPT = """\
Você é o assistente virtual da UNIVESP (Universidade Virtual do Estado de São Paulo).

Responda exclusivamente com base nas informações retornadas pelas ferramentas
de consulta disponíveis (FAQ, Manual do Candidato do Vestibular e Manual do
Aluno). Nunca invente informações que não estejam nos documentos.

Regras:
- Sempre use uma ferramenta antes de responder perguntas sobre a UNIVESP.
- Se a pergunta abranger mais de um tema, use mais de uma ferramenta.
- Se as ferramentas não retornarem informação suficiente para responder com
  segurança, diga claramente que não encontrou evidências suficientes nos
  documentos disponíveis. Não tente adivinhar ou completar com conhecimento
  próprio.
- Seja direto e, quando fizer sentido, cite o nome do documento consultado.
- Responda sempre em português do Brasil.
"""


def criar_agente(vector_store: Chroma):
    """
    Cria o agente RAG, conectando o modelo Gemini às ferramentas de consulta
    vinculadas à base vetorial informada.

    Args:
        vector_store: base vetorial já criada/carregada (ver baseVector.py).

    Returns:
        Agente pronto para ser executado via responder().
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY não configurada. Verifique se o arquivo .env "
            "existe na raiz do projeto e contém a variável GEMINI_API_KEY."
        )

    ferramentas = criar_ferramentas(vector_store)

    agente = create_agent(
        model=MODELO_AGENTE,
        tools=ferramentas,
        system_prompt=SYSTEM_PROMPT,
    )
    logger.info("Agente criado com modelo '%s' e %d ferramenta(s).", MODELO_AGENTE, len(ferramentas))
    return agente


def responder(agente, pergunta: str, historico: list[dict] | None = None) -> dict:
    """
    Executa uma pergunta no agente, incluindo o histórico de conversa opcional.

    Args:
        agente: agente criado por criar_agente().
        pergunta: pergunta atual do usuário.
        historico: lista de mensagens anteriores no formato
            [{"role": "user"|"assistant", "content": str}, ...]. Opcional.

    Returns:
        Dicionário com:
            - "resposta": texto final gerado pelo agente.
            - "ferramentas_usadas": nomes das ferramentas chamadas nesta rodada.
            - "fontes": nomes dos documentos PDF citados nos trechos recuperados.
    """
    mensagens = list(historico or [])
    mensagens.append({"role": "user", "content": pergunta})

    logger.info("Executando agente para a pergunta: %s", pergunta)
    resultado = agente.invoke({"messages": mensagens})

    mensagens_resultado = resultado["messages"]

    return {
        "resposta": _extrair_resposta_final(mensagens_resultado),
        "ferramentas_usadas": _extrair_ferramentas_usadas(mensagens_resultado),
        "fontes": _extrair_fontes(mensagens_resultado),
        "contexto_debug": _extrair_contexto_debug(mensagens_resultado),
    }


def _extrair_texto(conteudo) -> str:
    """
    Normaliza o campo `content` de uma AIMessage, que pode vir como string
    simples ou como lista de blocos estruturados — comportamento observado em
    modelos Gemini mais novos ao usar tool calling, que incluem blocos de
    texto junto de metadados extras (ex: assinatura da chamada de ferramenta).
    """
    if isinstance(conteudo, str):
        return conteudo

    if isinstance(conteudo, list):
        partes = [
            bloco.get("text", "")
            for bloco in conteudo
            if isinstance(bloco, dict) and bloco.get("type") == "text"
        ]
        return "".join(partes).strip()

    return str(conteudo)


def _extrair_resposta_final(mensagens: list) -> str:
    """Retorna o texto da última mensagem do assistente com conteúdo (resposta final)."""
    for mensagem in reversed(mensagens):
        if isinstance(mensagem, AIMessage) and mensagem.content:
            texto = _extrair_texto(mensagem.content)
            if texto:
                return texto
    return "Não foi possível gerar uma resposta."


def _extrair_ferramentas_usadas(mensagens: list) -> list[str]:
    """Lista os nomes das ferramentas chamadas ao longo da execução, sem duplicatas, na ordem."""
    nomes: list[str] = []
    for mensagem in mensagens:
        if isinstance(mensagem, AIMessage):
            for chamada in getattr(mensagem, "tool_calls", None) or []:
                nome = chamada.get("name")
                if nome and nome not in nomes:
                    nomes.append(nome)
    return nomes


def _extrair_fontes(mensagens: list) -> list[str]:
    """Extrai os nomes de arquivo PDF citados nos resultados das ferramentas (via regex)."""
    fontes: list[str] = []
    for mensagem in mensagens:
        if isinstance(mensagem, ToolMessage):
            for encontrada in re.findall(r"fonte:\s*([^,\n]+)", str(mensagem.content)):
                nome = encontrada.strip()
                if nome and nome not in fontes:
                    fontes.append(nome)
    return fontes


def _extrair_contexto_debug(mensagens: list) -> list[dict]:
    """
    Retorna o conteúdo bruto de cada chamada de ferramenta (texto exatamente
    como foi injetado no contexto do modelo), para exibição no modo Debug
    do Streamlit (main.py).
    """
    registros: list[dict] = []
    for mensagem in mensagens:
        if isinstance(mensagem, ToolMessage):
            registros.append({"ferramenta": mensagem.name, "conteudo": str(mensagem.content)})
    return registros


if __name__ == "__main__":
    from baseVector import criar_ou_carregar_base

    base = criar_ou_carregar_base()
    agente = criar_agente(base)

    perguntas_teste = [
        "Quando são as provas do vestibular 2026?",
        "Qual é o horário de funcionamento da lanchonete mais próxima da minha casa?",
    ]

    for pergunta in perguntas_teste:
        print(f"\n{'=' * 60}\nPergunta: {pergunta}\n{'=' * 60}")
        resultado = responder(agente, pergunta)
        print(f"\nResposta:\n{resultado['resposta']}")
        print(f"\nFerramentas usadas: {resultado['ferramentas_usadas']}")
        print(f"Fontes: {resultado['fontes']}")