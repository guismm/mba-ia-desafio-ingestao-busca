import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# Ative com DEBUG_PROMPT=1 para imprimir o prompt final (contexto + pergunta) no stderr
DEBUG_PROMPT = os.getenv("DEBUG_PROMPT", "").strip().lower() in ("1", "true", "yes")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS DENTRO DO CONTEXTO:
Pergunta: "Qual foi o faturamento da empresa X?"
Resposta: "O faturamento da empresa X foi de R$ 1000,00."

Pergunta: "Qual é o ano de fundação da empresa Y?"
Resposta: "O ano de fundação da empresa Y foi 2005."

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _format_docs_with_score(docs_with_score):
    """Formata documentos retornados por similarity_search_with_score."""
    return "\n\n".join(doc.page_content for doc, _ in docs_with_score)


def _log_prompt_if_debug(prompt_text):
    """Imprime o prompt final no stderr quando DEBUG_PROMPT=1."""
    if DEBUG_PROMPT:
        err = sys.stderr
        print("--- DEBUG PROMPT (após replace de {contexto} e {pergunta}) ---", file=err)
        print(prompt_text, file=err)
        print("--- FIM DEBUG PROMPT ---", file=err)
        err.flush()


def create_rag_components():
    """Cria e retorna os componentes do RAG (vector_store, prompt, llm)."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        )
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=os.getenv("PGVECTOR_COLLECTION", "rag"),
            connection=os.getenv(
                "PGVECTOR_URL", "postgresql://postgres:postgres@localhost:5432/rag"
            ),
            use_jsonb=True,
        )

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
            temperature=0.4,
        )

        return vector_store, prompt, llm
    except Exception as e:
        print(f"Erro ao inicializar componentes: {e}")
        return None, None, None


def search(query: str, vector_store, prompt, llm) -> str:
    """Executa a busca e retorna a resposta."""
    # Parte 1: Busca por similaridade
    docs_with_score = vector_store.similarity_search_with_score(query, k=10)
    contexto = _format_docs_with_score(docs_with_score)

    # Parte 2: Gera resposta com LLM
    prompt_text = prompt.format(contexto=contexto, pergunta=query)
    _log_prompt_if_debug(prompt_text)

    response = llm.invoke(prompt_text)
    return response.content
