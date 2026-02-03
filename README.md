# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão de PDF em PostgreSQL com pgVector e chat via CLI que responde com base apenas no conteúdo ingerido.

## Como executar

1. **Variáveis de ambiente**

   Copie o exemplo e preencha com suas chaves e caminhos:

   ```bash
   cp .env.example .env
   ```

   Edite `.env` e defina:

   - `LLM_PROVIDER`: `openai` ou `gemini` (deve bater com a API key usada).
   - Para OpenAI: `OPENAI_API_KEY`, e opcionalmente `OPENAI_EMBEDDING_MODEL` e `OPENAI_LLM_MODEL`.
   - Para Gemini: `GOOGLE_API_KEY`, e opcionalmente `GOOGLE_EMBEDDING_MODEL` e `GOOGLE_LLM_MODEL`.
   - `DATABASE_URL`: conexão com o Postgres (ex.: `postgresql+psycopg://postgres:postgres@localhost:5432/rag`).
   - `PG_VECTOR_COLLECTION_NAME`: nome da coleção (padrão: `documents`).
   - `PDF_PATH`: caminho do PDF a ser ingerido (ex.: `document.pdf`).

   Os nomes de modelo sugeridos no desafio (ex.: gpt-5-nano, gemini-2.5-flash-lite) podem precisar ser ajustados para IDs válidos atuais (ex.: `gpt-4o-mini`, `gemini-2.0-flash`). Consulte a documentação do provedor.

2. **Banco de dados**

   Suba o Postgres com pgVector:

   ```bash
   docker compose up -d
   ```

   Aguarde o healthcheck do Postgres e a execução do serviço que cria a extensão `vector`.

3. **Ingestão do PDF**

   Com o banco no ar e o `.env` configurado:

   ```bash
   python src/ingest.py
   ```

   O script carrega o PDF em `PDF_PATH`, divide em chunks (1000 caracteres, overlap 150), gera embeddings e grava no banco. Em caso de **erro 429 (quota excedida)** da OpenAI, use o Gemini: no `.env` defina `LLM_PROVIDER=gemini` e `GOOGLE_API_KEY` (e remova ou deixe em branco `OPENAI_API_KEY` se quiser). Opcionalmente use `INGEST_BATCH_SIZE`, `INGEST_BATCH_DELAY_SEC` e `INGEST_RETRY_BACKOFF_SEC` para ajustar rate limits.

4. **Chat no terminal**

   Para fazer perguntas no CLI:

   ```bash
   python src/chat.py
   ```

   Digite a pergunta quando solicitado. Para sair, use `sair` ou Enter vazio. As respostas usam apenas o conteúdo do PDF ingerido; perguntas fora do contexto recebem: *"Não tenho informações necessárias para responder sua pergunta."*

## Dependências

Instale com:

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ e Postgres com extensão pgvector (o `docker-compose.yml` já provê a imagem adequada).
