import pandas as pd

df = pd.read_excel("disciplinas.xlsx")

for nome in ["Redes de Computadores", "Inteligência Artificial"]:
    print("\n====================")
    print(nome)
    print("====================")

    linha = df[df["Componente"].astype(str).str.contains(nome, case=False, na=False)]

    if linha.empty:
        print("Disciplina NÃO encontrada")
        continue

    for campo in [
        "Bibliografia_Basica",
        "Bibliografia_Complementar",
        "Areas_integracao",
        "Ementa",
        "Objetivos",
        "Conteudo"
    ]:
        print("\n", campo)
        print(linha.iloc[0].get(campo, "COLUNA NÃO EXISTE"))