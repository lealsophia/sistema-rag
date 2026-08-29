import pandas as pd

from sentence_transformers import (
    SentenceTransformer
)

# =====================================================
# ARQUIVOS
# =====================================================

ARQUIVO_CHUNKS = "chunks.xlsx"

ARQUIVO_SAIDA = "chunks_embeddings.json"

# =====================================================
# LEITURA
# =====================================================

df = pd.read_excel(
    ARQUIVO_CHUNKS,
    sheet_name="chunks"
)

df["texto"] = (
    df["texto"]
    .fillna("")
    .astype(str)
)

df = df[
    df["texto"].str.strip() != ""
].copy()

print("Total de chunks:", len(df))

# =====================================================
# MODELO
# =====================================================

MODELO = "BAAI/bge-m3"

model = SentenceTransformer(
    MODELO
)

# =====================================================
# PREPARAÇÃO
# =====================================================

textos = [
    f"passage: {t}"
    for t in df["texto"].tolist()
]

# =====================================================
# EMBEDDINGS
# =====================================================

print("\nGerando embeddings...")

embeddings = model.encode(

    textos,

    show_progress_bar=True,

    batch_size=16,

    normalize_embeddings=True
)

df["embedding"] = embeddings.tolist()

# =====================================================
# EXPORTAÇÃO
# =====================================================

df.to_json(

    ARQUIVO_SAIDA,

    orient="records",

    force_ascii=False,

    indent=2
)

print("\n✅ Embeddings gerados!")

print(f"Arquivo salvo: {ARQUIVO_SAIDA}")