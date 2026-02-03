import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

load_dotenv()

# Batch size and delays to avoid API rate limits (429)
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "10"))
INGEST_BATCH_DELAY_SEC = float(os.getenv("INGEST_BATCH_DELAY_SEC", "1.0"))
INGEST_MAX_RETRIES = int(os.getenv("INGEST_MAX_RETRIES", "3"))
INGEST_RETRY_BACKOFF_SEC = float(os.getenv("INGEST_RETRY_BACKOFF_SEC", "60"))

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()


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


def ingest_pdf():
    if not PDF_PATH or not PDF_PATH.strip():
        raise ValueError("PDF_PATH is not set. Set it in .env or environment.")
    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")
    if not DATABASE_URL or not DATABASE_URL.strip():
        raise ValueError("DATABASE_URL is not set. Set it in .env or environment.")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    embeddings = _get_embeddings()
    vector_store = PGVector(
        embeddings=embeddings,
        connection=DATABASE_URL,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        use_jsonb=True,
    )

    total = 0
    for i in range(0, len(chunks), INGEST_BATCH_SIZE):
        batch = chunks[i : i + INGEST_BATCH_SIZE]
        for attempt in range(INGEST_MAX_RETRIES):
            try:
                vector_store.add_documents(batch)
                total += len(batch)
                break
            except Exception as e:
                err_name = type(e).__module__ + "." + type(e).__qualname__
                is_rate_limit = "429" in str(e) or "RateLimit" in err_name or "insufficient_quota" in str(e).lower()
                if is_rate_limit and attempt < INGEST_MAX_RETRIES - 1:
                    wait = INGEST_RETRY_BACKOFF_SEC * (attempt + 1)
                    print(f"Rate limit (429). Waiting {wait:.0f}s before retry {attempt + 2}/{INGEST_MAX_RETRIES}...")
                    time.sleep(wait)
                else:
                    if "insufficient_quota" in str(e).lower():
                        print(
                            "Erro: cota da API OpenAI excedida. Opções: (1) Use Gemini "
                            "definindo LLM_PROVIDER=gemini e GOOGLE_API_KEY no .env; "
                            "(2) Verifique billing em https://platform.openai.com/account/billing"
                        )
                    raise
        if i + INGEST_BATCH_SIZE < len(chunks):
            time.sleep(INGEST_BATCH_DELAY_SEC)
    return total


if __name__ == "__main__":
    n = ingest_pdf()
    print(f"Ingested {n} chunks.")
