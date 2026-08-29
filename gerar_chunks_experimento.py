import os
import re
from datetime import datetime

import pandas as pd

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================================================
# CONFIGURAÇÃO EXPERIMENTAL
# =========================================================

ESTRATEGIA = "semantic"
# opções:
# recursive
# fixed
# semantic

CHUNK_SIZE = 350
CHUNK_OVERLAP = 50

MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# =========================================================
# ARQUIVOS
# =========================================================

ARQUIVO_ENTRADA = "disciplinas.xlsx"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

PASTA = "experimentos_chunks"

os.makedirs(PASTA, exist_ok=True)

ARQUIVO_SAIDA = os.path.join(
    PASTA,
    f"chunks_{ESTRATEGIA}_{CHUNK_SIZE}_{CHUNK_OVERLAP}_{timestamp}.xlsx"
)


# =========================================================
# LIMPEZA
# =========================================================

def limpar(valor):

    if pd.isna(valor):
        return ""

    valor = str(valor).strip()

    if valor == "":
        return ""

    return valor


def limpar_ruido_pdf(texto):
    texto = str(texto)

    # Remove valores vazios comuns
    texto = texto.replace("nan", "")
    texto = texto.replace("NaN", "")

    # Normaliza quebras de linha e espaços
    texto = re.sub(r"\r\n", "\n", texto)
    texto = re.sub(r"\r", "\n", texto)

    # Remove cabeçalhos e rodapés repetidos do PDF
    padroes_remover = [
        r"Projeto\s+Pedagógico\s+do\s+Curso\s+Técnico\s+em\s+Informática\s+Integrado\s+ao\s+Ensino\s+Médio",
        r"Projeto\s+Pedagógico\s+do\s+Curso\s+T[ée]cnico\s+em\s+Inform[áa]tica\s+Integrado\s+ao\s+Ensino\s+M[ée]dio",
        r"Curso\s+T[ée]cnico\s+em\s+Inform[áa]tica\s+Integrado\s+ao\s+Ensino\s+M[ée]dio",
        r"T[ée]cnico\s+em\s+Inform[áa]tica\s+Integrado\s+ao\s+Ensino\s+M[ée]dio",
        r"T[ÉE]CNICO\s+EM\s+INFORM[ÁA]TICA\s+INTEGRADO\s+AO\s+ENSINO\s+M[ÉE]DIO",
        r"CUBAT[ÃA]O",
    ]

    for padrao in padroes_remover:
        texto = re.sub(padrao, " ", texto, flags=re.IGNORECASE)

    # Remove variações em linha única com muito espaçamento
    texto = re.sub(
        r"Projeto\s+Pedagógico.*?Ensino\s+M[ée]dio",
        " ",
        texto,
        flags=re.IGNORECASE
    )

    texto = re.sub(
        r"CUBAT[ÃA]O\s+T[ÉE]CNICO\s+EM\s+INFORM[ÁA]TICA\s+INTEGRADO\s+AO\s+ENSINO\s+M[ÉE]DIO",
        " ",
        texto,
        flags=re.IGNORECASE
    )

    # Remove possíveis números de página
    texto = re.sub(r"\bPágina\s*\d+\b", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\bP[áa]gina\s*\d+\s*de\s*\d+\b", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\b\d+\s*/\s*\d+\b", " ", texto)

    # Remove linhas que ficaram só com espaços ou traços
    linhas = []
    for linha in texto.split("\n"):
        linha_limpa = linha.strip()

        if not linha_limpa:
            continue

        if re.fullmatch(r"[-–—_\s]+", linha_limpa):
            continue

        linhas.append(linha_limpa)

    texto = "\n".join(linhas)

    # Remove espaços repetidos, preservando quebras úteis
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()


# =========================================================
# LEITURA
# =========================================================

df = pd.read_excel(ARQUIVO_ENTRADA)

for col in df.columns:

    if df[col].dtype == "object":

        df[col] = df[col].apply(
            limpar_ruido_pdf
        )


# =========================================================
# SPLITTER
# =========================================================

def criar_splitter(estrategia, chunk_size, overlap):

    if estrategia == "fixed":

        return CharacterTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    if estrategia == "recursive":

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
                ""
            ]
        )

    if estrategia == "semantic":

        embeddings = HuggingFaceEmbeddings(
            model_name=MODELO
        )

        return SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80
        )

    raise ValueError("Estratégia inválida.")


# =========================================================
# CAMPOS
# =========================================================

campos_textuais = [

    "Conhecimentos",

    "Ementa",

    "Objetivos",

    "Areas_integracao",

    "Conteudo",

    "Bibliografia_Basica",

    "Bibliografia_Complementar"
]


# =========================================================
# GERAR CHUNKS
# =========================================================

chunks = []

splitter = criar_splitter(
    ESTRATEGIA,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

for _, row in df.iterrows():

    componente = limpar(
        row.get("Componente")
    )

    componente = limpar_ruido_pdf(componente)

    ano = limpar(
        row.get("Ano")
    )

    ano = limpar_ruido_pdf(ano)

    # =====================================================
    # ESTRUTURAL
    # =====================================================

    texto_estrutural = f"""
Disciplina: {componente}

Ano: {ano}

Aulas semanais:
{limpar_ruido_pdf(limpar(row.get("Aulas_semanais")))}

Carga horária presencial:
{limpar_ruido_pdf(limpar(row.get("CH_presencial")))}

Carga horária laboratório:
{limpar_ruido_pdf(limpar(row.get("CH_laboratorio")))}
""".strip()

    texto_estrutural = limpar_ruido_pdf(
        texto_estrutural
    )

    if texto_estrutural:

        chunks.append({

            "Componente": componente,

            "Ano": ano,

            "campo": "Estrutural",

            "chunk_id": 0,

            "texto_original_campo":
                texto_estrutural,

            "texto_chunkado":
                texto_estrutural,

            "texto":
                texto_estrutural,

            "tamanho_chunk":
                len(texto_estrutural),

            "estrategia":
                ESTRATEGIA
        })

    # =====================================================
    # TEXTUAIS
    # =====================================================

    for campo in campos_textuais:

        texto_original = limpar(
            row.get(campo)
        )

        texto_original = limpar_ruido_pdf(
            texto_original
        )

        if texto_original == "":
            continue

        partes = splitter.split_text(
            texto_original
        )

        for i, parte in enumerate(partes):

            parte = limpar_ruido_pdf(parte)

            if not parte:
                continue

            texto_chunk = f"""
            Disciplina: {componente}

            Ano: {ano}

            Campo: {campo}

            Conteúdo:

            {parte}
            """.strip().strip()

            texto_chunk = limpar_ruido_pdf(
                texto_chunk
            )

            chunks.append({

                "Componente":
                    componente,

                "Ano":
                    ano,

                "campo":
                    campo,

                "chunk_id":
                    i,

                "texto_original_campo":
                    texto_original,

                "texto_chunkado":
                    parte,

                "texto":
                    texto_chunk,

                "tamanho_chunk":
                    len(texto_chunk),

                "estrategia":
                    ESTRATEGIA
            })


# =========================================================
# DATAFRAME FINAL
# =========================================================

df_chunks = pd.DataFrame(chunks)

if not df_chunks.empty:

    df_chunks = df_chunks.drop_duplicates(

        subset=[
            "Componente",
            "campo",
            "chunk_id",
            "texto_chunkado"
        ]

    ).reset_index(drop=True)


# =========================================================
# VALIDAÇÃO
# =========================================================

print("\n✅ Chunks gerados!")

print("Estratégia:", ESTRATEGIA)

print("Total chunks:", len(df_chunks))

print("\nChunks por campo:")

print(df_chunks["campo"].value_counts())

print("\nTamanho dos chunks:")

print(df_chunks["tamanho_chunk"].describe())


# =========================================================
# VERIFICAÇÃO DE RUÍDO
# =========================================================

ruidos = [
    "Projeto Pedagógico",
    "CUBATÃO",
    "TÉCNICO EM INFORMÁTICA INTEGRADO AO ENSINO MÉDIO"
]

print("\n🔎 Verificando ruídos restantes:")

for ruido in ruidos:

    qtd = df_chunks["texto"].astype(str).str.contains(
        ruido,
        case=False,
        regex=False
    ).sum()

    print(f"{ruido}: {qtd} ocorrência(s)")


# =========================================================
# SALVAR
# =========================================================

with pd.ExcelWriter(ARQUIVO_SAIDA) as writer:

    df_chunks.to_excel(
        writer,
        sheet_name="chunks",
        index=False
    )

print("\n✅ Arquivo salvo:")
print(ARQUIVO_SAIDA)