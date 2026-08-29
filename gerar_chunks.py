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
# GERAÇÃO DE CHUNKS
# =========================================================

def criar_chunk(componente, ano, campo, descricao, conteudo):

    return limpar_ruido_pdf(
        f"""
        Documento: Projeto Pedagógico do Curso Técnico em Informática Integrado ao Ensino Médio
        
Disciplina: {componente}

Ano: {ano}

Campo: {campo}

Conteúdo:
{conteudo}
""".strip()
    )

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

def gerar_chunks(df, estrategia, chunk_size, overlap):

    chunks = []

    splitter = criar_splitter(
        estrategia,
        chunk_size,
        overlap
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

        # TODO: TODO O RESTO DO CÓDIGO FICA AQUI DENTRO
        campos_estruturais = [
            (
                "Aulas semanais",
                limpar(row.get("Aulas_semanais")),
                "Quantidade de aulas semanais da disciplina."
            ),
            (
                "Carga horária presencial",
                limpar(row.get("CH_presencial")),
                "Carga horária presencial da disciplina."
            ),
            (
                "Carga horária de laboratório",
                limpar(row.get("CH_laboratorio")),
                "Carga horária destinada às atividades de laboratório."
            )

        ]

        descricoes = {

            "Conhecimentos":
                "Conhecimentos prévios recomendados para acompanhar a disciplina.",

            "Ementa":
                "Resumo oficial dos conteúdos abordados pela disciplina.",

            "Objetivos":
                "Objetivos de aprendizagem da disciplina.",

            "Areas_integracao":
                "Disciplinas e áreas do conhecimento relacionadas.",

            "Conteudo":
                "Conteúdo programático previsto para a disciplina.",

            "Bibliografia_Basica":
                "Bibliografia básica utilizada na disciplina.",

            "Bibliografia_Complementar":
                "Bibliografia complementar recomendada."
        }

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

                texto_chunk = criar_chunk(

                    componente,

                    ano,

                    campo,

                    descricoes[campo],

                    parte

                )

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

        for campo, valor, descricao in campos_estruturais:

            valor = limpar_ruido_pdf(valor)

            if valor == "":
                continue

            texto = criar_chunk(
                componente,
                ano,
                campo,
                descricao,
                valor
            )

            chunks.append({

                "Componente": componente,
                "Ano": ano,
                "campo": campo,
                "chunk_id": 0,

                "texto_original_campo": valor,

                "texto_chunkado": valor,

                "texto": texto,

                "tamanho_chunk": len(texto),

                "estrategia": ESTRATEGIA

            })

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

    return df_chunks

    # =====================================================
    # ESTRUTURAL
    # =====================================================


    # =====================================================
    # TEXTUAIS
    # =====================================================



# =========================================================
# DATAFRAME FINAL
# =========================================================

if __name__ == "__main__":

    df = pd.read_excel(ARQUIVO_ENTRADA)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(limpar_ruido_pdf)

    df_chunks = gerar_chunks(
        df,
        ESTRATEGIA,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

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