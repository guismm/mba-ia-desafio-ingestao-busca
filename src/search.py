import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

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


def _get_embeddings():
    """Return embeddings instance based on LLM_PROVIDER."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)
    raise ValueError(
        f"LLM_PROVIDER must be 'openai' or 'gemini'. Got: {LLM_PROVIDER!r}"
    )


def _get_llm():
    """Return chat LLM instance based on LLM_PROVIDER."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    raise ValueError(
        f"LLM_PROVIDER must be 'openai' or 'gemini'. Got: {LLM_PROVIDER!r}"
    )


def search_prompt(question: str) -> str:
    if not question or not question.strip():
        return "Não tenho informações necessárias para responder sua pergunta."

    if not DATABASE_URL or not DATABASE_URL.strip():
        raise ValueError("DATABASE_URL is not set. Set it in .env or environment.")

    embeddings = _get_embeddings()
    vector_store = PGVector(
        embeddings=embeddings,
        connection=DATABASE_URL,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        use_jsonb=True,
    )
    results = vector_store.similarity_search_with_score(query=question, k=10)
    contexto = "\n\n".join(doc.page_content for doc, _ in results)
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question.strip())
    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content if hasattr(response, "content") else str(response)
