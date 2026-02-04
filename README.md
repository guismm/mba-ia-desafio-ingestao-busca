# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema RAG (Retrieval-Augmented Generation) para ingestão de documentos PDF e perguntas/respostas via CLI utilizando LangChain e modelos Google Gemini.

## Requisitos

- Python 3.12+
- Docker e Docker Compose
- API Key do Google (Gemini)

## Configuração

1. Clone o repositório e crie o ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Configure as variáveis de ambiente:

```bash
cp .env.example .env
```

Edite o arquivo `.env` e adicione sua `GOOGLE_API_KEY`.

## Execução

1. Subir o banco de dados PostgreSQL com pgVector:

```bash
docker-compose up -d
```

2. Executar a ingestão do PDF:

```bash
python src/ingest.py
```

3. Rodar o chat interativo:

```bash
python src/chat.py
```

## Exemplo de uso

```
Faça sua pergunta (digite 'quit' ou 'exit' para sair):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Estrutura do Projeto

```
├── docker-compose.yml    # PostgreSQL + pgVector
├── requirements.txt      # Dependências Python
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Módulo de busca RAG
│   └── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md
```

## Tecnologias

- **LangChain**: Framework para aplicações LLM
- **Google Gemini**: Embeddings (`models/embedding-001`) e LLM (`gemini-2.5-flash-lite`)
- **PostgreSQL + pgVector**: Armazenamento vetorial
- **PyPDF**: Leitura de documentos PDF
