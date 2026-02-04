import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

def ingest_document(pdf_path: str) -> int:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    print(f"📄 Lendo PDF: {pdf_path}")
    docs = PyPDFLoader(str(pdf_path), extraction_mode="layout").load()
    
    # --- LÓGICA DE CABEÇALHO ---
    # Assumimos que a primeira linha do PDF é o cabeçalho da tabela
    full_text = "\n".join([doc.page_content for doc in docs])
    lines = full_text.split('\n')
    header = lines[0] if lines else ""
    headers = re.split(r'\s{2,}', header.strip())
    # ---------------------------

    chunker = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, 
        add_start_index=False
    )

    chunks = chunker.split_documents(docs)

    enriched = []
    for i, doc in enumerate(chunks):
        # Agora vamos transformar cada linha no formato de chave-valor
        content = doc.page_content
        lines = content.split('\n')
        structured_lines = []
        for j, line in enumerate(lines):
            # Pula a linha do cabeçalho original
            if i == 0 and j == 0:
                continue 
            values = re.split(r'\s{2,}', line.strip())
            if len(values) == len(headers):
                structured_line = ";".join([f"{h}: {v}" for h, v in zip(headers, values)])
                structured_lines.append(structured_line)
            else:
                # Linha sem estrutura clara, manter como está
                structured_lines.append(line)  

        
        enriched.append(
            Document(
                page_content="\n".join([line for line in structured_lines]),
                metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)}
            )
        )

    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"))
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION", "rag"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )

    vector_store.add_documents(documents=enriched, ids=ids)
    return len(ids)

if __name__ == "__main__":
    count = ingest_document(os.getenv("PDF_PATH", "document.pdf"))
    print(f"✅ Ingestão completa! {count} chunks com cabeçalho injetado.")