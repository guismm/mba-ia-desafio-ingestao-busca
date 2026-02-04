from search import create_rag_components, search


def main():
    vector_store, prompt, llm = create_rag_components()

    if not vector_store:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'quit' ou 'exit' para sair):\n")

    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando...")
            break

        if not pergunta:
            continue
        if pergunta.lower() in ("quit", "exit"):
            print("Encerrando...")
            break

        try:
            resposta = search(pergunta, vector_store, prompt, llm)
            print(f"RESPOSTA: {resposta}\n")
        except Exception as e:
            print(f"Erro ao processar pergunta: {e}\n")


if __name__ == "__main__":
    main()
