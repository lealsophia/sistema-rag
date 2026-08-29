import pandas as pd
from langchain_core.documents import Document


ARQUIVO_CHUNKS = "chunks.xlsx"
PASTA_CHROMA = "chroma_ppc"
COLLECTION_NAME = "ppc_informatica"

df = pd.read_excel(ARQUIVO_CHUNKS, sheet_name="chunks")

def limpar(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return "Não se aplica"
    return str(valor).strip()

documents = []

for _, row in df.iterrows():
    documents.append(
        Document(
            page_content=limpar(row.get("texto")),
            metadata={
                "curso": limpar(row.get("Curso")),
                "componente": limpar(row.get("Componente")),
                "tipo": limpar(row.get("Tipo")),
                "nucleo": limpar(row.get("Núcleo")),
                "ano": limpar(row.get("Ano")),
                "sigla": limpar(row.get("Sigla")),
                "aulas_semanais": limpar(row.get("Aulas_semanais")),
                "total_aulas": limpar(row.get("Total_aulas")),
                "ch_presencial": limpar(row.get("CH_presencial")),
                "ch_distancia": limpar(row.get("CH_distancia")),
                "total_horas": limpar(row.get("Total_horas")),
                "docentes": limpar(row.get("Docentes")),
                "ch_laboratorio": limpar(row.get("CH_laboratorio")),
                "campo": limpar(row.get("campo")),
                "chunk_id": limpar(row.get("chunk_id")),
                "estrategia": limpar(row.get("estrategia")),
                "chunk_size": limpar(row.get("chunk_size")),
                "chunk_overlap": limpar(row.get("chunk_overlap")),
            }
        )
    )

print("Total de documentos:", len(documents))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=PASTA_CHROMA,
    collection_name=COLLECTION_NAME
)

print("✅ ChromaDB criado com sucesso!")
print(f"Salvo na pasta: {PASTA_CHROMA}")