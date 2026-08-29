import chromadb

#cria cliente (banco)
client = chromadb.Client()

#cria ou pega uma coleção
collection = client.get_or_create_collection(name="meu_banco")

#dado guardado
texto = "O Brasil tem como capital Brasília"

#adiciona no banco
collection.add(
    documents=[texto],
    ids=["1"]
)

print("Salvo no banco!")

#buscar no banco
resultado = collection.query(
    query_texts=["Qual é a capital da Argentina?"],
    n_results=1
)
print("\nResultado da busca:")
print(resultado["documents"])