# Copilot Instructions — Agente UNIVESP (RAG)

## Project Overview
Agente conversacional RAG que responde perguntas sobre a UNIVESP (cursos, vestibulares, polos, calendário, editais, FAQ) com base **exclusivamente** em PDFs oficiais indexados localmente. Stack: Python, LangChain, Google Gemini, ChromaDB, Streamlit, PyPDFLoader.

Regra inegociável: o agente nunca responde com conhecimento geral do modelo. Se o contexto recuperado do ChromaDB for insuficiente, a resposta deve ser explícita: "não encontrei essa informação nos documentos disponíveis". Nunca gere resposta que pareça vir dos documentos sem ter sido de fato recuperada deles.

## Architecture
Três camadas, sem pular etapas:

1. **Ingestão** — `core/readPDFs.py` → `core/baseVector.py`. Lê PDFs, faz chunking, gera embeddings, persiste no ChromaDB.
2. **Raciocínio** — `core/pipelineRAG.py` → `core/ferramentas.py` → `core/agentRAG.py`. Recupera contexto, monta prompt, o agente escolhe a tool certa e chama o Gemini.
3. **Apresentação** — `main.py`. Interface Streamlit; único ponto de entrada do usuário.

Regras de dependência (não violar):
- `main.py` nunca chama `baseVector.py` ou `pipelineRAG.py` diretamente para responder perguntas — sempre passa por `agentRAG.py`. Exceção: botão "reconstruir base vetorial" chama `readPDFs.py` → `baseVector.py` direto.
- `ferramentas.py` depende de `pipelineRAG.py`, nunca o inverso.
- `pipelineRAG.py` depende só de `baseVector.py` e de `prompts/`.
- `config/` e `utils/` são transversais: podem ser importados por qualquer módulo, mas nunca importam módulos de domínio.

## File Responsibilities (uma responsabilidade por arquivo)
- `core/readPDFs.py`: leitura, limpeza e chunking de PDFs. Classes: `PDFLoaderService`, `DocumentChunker`, `DocumentCategorizer`.
- `core/baseVector.py`: criação/atualização/persistência do índice ChromaDB. Classes: `EmbeddingProvider`, `VectorStoreManager`.
- `core/pipelineRAG.py`: retrievers, prompt templates, cadeia de resposta. Classes: `RetrieverFactory`, `PromptTemplateManager`, `RAGChain`.
- `core/ferramentas.py`: uma Tool por domínio temático (Vestibular, Cursos, Editais, Institucional, FAQ, Calendário), cada uma filtrando apenas seu subconjunto de documentos. Classe agregadora: `ToolRegistry`.
- `core/agentRAG.py`: seleção de tool, chamada ao LLM, resposta final. Classes: `UnivespAgent`, `ResponseFormatter`.
- `main.py`: chat, histórico, botão de limpar conversa, exibição de fontes, modo debug opcional, reconstrução da base.

Ao gerar código para um destes arquivos, não adicione responsabilidades de outro módulo nele — se a lógica pertence a outro arquivo, sinalize isso em vez de implementar ali.

## Code Style
- Type hints obrigatórios em toda função/método, incluindo retorno.
- Docstrings em todas as classes e métodos públicos (padrão Google ou NumPy).
- Logging via `utils/logger.py`, nunca `print()`. `INFO` para fluxo normal, `WARNING` para contexto insuficiente, `ERROR` para falha de leitura/API.
- Tratamento de exceções explícito em I/O (leitura de PDF, chamadas ao Gemini, acesso ao ChromaDB) — nunca `except Exception` silencioso.
- Configuração (chaves de API, parâmetros de chunk, nome da coleção) sempre em `config/settings.py` ou `.env`, nunca hardcoded no código.
- Prompts vivem em arquivos dentro de `prompts/`, nunca como strings soltas no meio do código Python.

## Patterns
- Toda Tool em `ferramentas.py` segue o mesmo contrato: método `run(query: str) -> str` e atributo `description` claro (o LLM decide qual tool usar com base nesse texto — não deixe descrições vagas ou parecidas entre tools).
- Toda resposta do agente deve retornar junto as fontes (nome do arquivo/página) usadas para gerá-la.
- Indexação deve ser idempotente: `VectorStoreManager.update()` não deve duplicar chunks já indexados (usar hash do conteúdo como parte do ID).
- Nomeie a coleção ChromaDB com versão (ex.: `univesp_v1`) para permitir rollback em caso de regressão após reindexação.
- Novas integrações (API externa, banco SQL/NoSQL) entram como novas classes em `ferramentas.py` seguindo o mesmo contrato de Tool — não abra novos caminos de comunicação entre módulos para isso.

## Testing
- Cobrir no mínimo: chunking (nenhum documento perdido), contrato de cada Tool (aceita e devolve o formato esperado), e um teste end-to-end do pipeline RAG.
- Testes ficam em `tests/`, espelhando a estrutura de `core/`.
- Após qualquer mudança em `readPDFs.py` ou `baseVector.py`, rodar os testes de chunking/indexação antes de aceitar a sugestão.

## Things to Avoid
- Não gerar resposta do agente sem contexto recuperado do ChromaDB por trás.
- Não colocar lógica de negócio (RAG, seleção de tool) dentro de `main.py`.
- Não criar dependência circular entre `ferramentas.py` e `pipelineRAG.py`.
- Não hardcodar chaves de API, nomes de coleção ou caminhos de arquivo.
- Não usar `except Exception` genérico sem logar ou re-lançar.
- Não misturar prompts como strings inline quando já existe `prompts/` para isso.