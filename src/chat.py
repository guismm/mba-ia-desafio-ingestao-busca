from search import search_prompt


def main():
    print("Faça sua pergunta (digite 'sair' ou Enter vazio para encerrar).\n")
    while True:
        try:
            print("Faça sua pergunta:")
            pergunta = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break
        if not pergunta or pergunta.lower() == "sair":
            print("Encerrando.")
            break
        try:
            resposta = search_prompt(pergunta)
        except Exception as e:
            print(
                "Não foi possível obter a resposta. Verifique DATABASE_URL, "
                "API key e se a ingestão foi feita. Erro:",
                e,
            )
            continue
        print(f"\nPERGUNTA: {pergunta}")
        print(f"RESPOSTA: {resposta}\n\n---\n")


if __name__ == "__main__":
    main()
