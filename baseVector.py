from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from readPDFs import carregar_pdfs

embed_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

vector_store = InMemoryVectorStore.from_documents(carregar_pdfs(), embed_model)

docs = vector_store.similarity_search("Sobre os cursos", k=2)
     
for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')