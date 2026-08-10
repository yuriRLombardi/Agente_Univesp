"""
Interface Streamlit do agente RAG da UNIVESP: chat com histórico, botão de
limpar conversa, botão de reconstruir a base vetorial, exibição das fontes
utilizadas em cada resposta e modo Debug com o contexto bruto recuperado.
"""

import logging

import streamlit as st

from agentRAG import criar_agente, responder
from baseVector import criar_ou_carregar_base, reconstruir_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Agente UNIVESP", page_icon="🎓")


def _inicializar_estado() -> None:
    """Garante que as chaves usadas em session_state existam, sem sobrescrever valores já presentes."""
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("agente", None)
    st.session_state.setdefault("chat", [])  # cada item: {"role", "content", "fontes"?, "debug"?}
    st.session_state.setdefault("modo_debug", False)


def _garantir_agente() -> None:
    """Cria a base vetorial e o agente na primeira execução, reaproveitando entre reruns do Streamlit."""
    if st.session_state.agente is not None:
        return

    with st.spinner("Carregando base de conhecimento..."):
        st.session_state.vector_store = criar_ou_carregar_base()
        st.session_state.agente = criar_agente(st.session_state.vector_store)


def _mensagens_para_agente() -> list[dict]:
    """Converte o histórico exibido na tela para o formato simples {role, content} esperado pelo agente."""
    return [{"role": item["role"], "content": item["content"]} for item in st.session_state.chat]


def _renderizar_historico() -> None:
    """Reexibe todas as mensagens já trocadas na conversa, incluindo fontes/debug de respostas anteriores."""
    for item in st.session_state.chat:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                _renderizar_extras(item)


def _renderizar_extras(item: dict) -> None:
    """Mostra o expander de fontes utilizadas e, se o modo Debug estiver ativo, o contexto bruto recuperado."""
    fontes = item.get("fontes") or []
    if fontes:
        with st.expander("📄 Fontes utilizadas"):
            for fonte in fontes:
                st.markdown(f"- {fonte}")

    if st.session_state.modo_debug and item.get("debug"):
        with st.expander("🔍 Debug: contexto recuperado"):
            for registro in item["debug"]:
                st.markdown(f"**Ferramenta:** `{registro['ferramenta']}`")
                st.code(registro["conteudo"], language="text")


def _renderizar_sidebar() -> None:
    """Controles laterais: modo debug, limpar conversa e reconstruir base vetorial."""
    with st.sidebar:
        st.header("Configurações")

        st.session_state.modo_debug = st.toggle("Modo Debug", value=st.session_state.modo_debug)

        if st.button("🗑️ Limpar conversa"):
            st.session_state.chat = []
            st.rerun()

        st.divider()

        if st.button("🔄 Reconstruir base vetorial"):
            with st.spinner("Reconstruindo base a partir dos PDFs (pode levar alguns minutos)..."):
                st.session_state.vector_store = reconstruir_base()
                st.session_state.agente = criar_agente(st.session_state.vector_store)
            st.success("Base reconstruída com sucesso.")


def main() -> None:
    st.title("🎓 Agente UNIVESP")
    st.caption("Tire dúvidas sobre a FAQ, o Vestibular 2026 e o Manual do Aluno da UNIVESP.")

    _inicializar_estado()
    _renderizar_sidebar()

    try:
        _garantir_agente()
    except RuntimeError as erro:
        st.error(str(erro))
        st.stop()

    _renderizar_historico()

    pergunta = st.chat_input("Digite sua pergunta...")
    if not pergunta:
        return

    st.session_state.chat.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando os documentos..."):
            try:
                resultado = responder(
                    st.session_state.agente,
                    pergunta,
                    historico=_mensagens_para_agente()[:-1],  # exclui a pergunta atual (já anexada acima)
                )
            except Exception as erro:
                logger.exception("Erro ao processar a pergunta.")
                st.error(f"Ocorreu um erro ao consultar o agente: {erro}")
                return

        st.markdown(resultado["resposta"])

        item_assistente = {
            "role": "assistant",
            "content": resultado["resposta"],
            "fontes": resultado["fontes"],
            "debug": resultado.get("contexto_debug", []),
        }
        _renderizar_extras(item_assistente)
        st.session_state.chat.append(item_assistente)


if __name__ == "__main__":
    main()