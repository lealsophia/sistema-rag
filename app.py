import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================
# CONFIGURAÇÃO
# =========================
PASTA_CHROMA = "chroma_ppc"
COLLECTION_NAME = "ppc_informatica"
MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# =========================
# CARREGAR BANCO
# =========================
@st.cache_resource
def carregar_db():
    embeddings = HuggingFaceEmbeddings(model_name=MODELO)

    return Chroma(
        persist_directory=PASTA_CHROMA,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

db = carregar_db()

# =========================
# FUNÇÕES
# =========================
def valor(v):
    if v is None or str(v).strip() == "":
        return "Não se aplica"
    return str(v).strip()


def detectar_campo(pergunta):
    p = pergunta.lower()

    if "ementa" in p:
        return "Ementa"

    if "objetivo" in p or "objetivos" in p:
        return "Objetivos"

    if (
        "conteúdo" in p
        or "conteudo" in p
        or "programático" in p
        or "programatico" in p
    ):
        return "Conteudo"

    if "bibliografia básica" in p or "bibliografia basica" in p:
        return "Bibliografia_Basica"

    if "bibliografia complementar" in p:
        return "Bibliografia_Complementar"

    if "conhecimentos" in p or "currículo de referência" in p:
        return "Conhecimentos"

    if "áreas de integração" in p or "areas de integracao" in p:
        return "Areas_integracao"

    return None


def montar_resposta(doc, pergunta):
    meta = doc.metadata
    campo = valor(meta.get("campo"))
    componente = valor(meta.get("componente"))

    p = pergunta.lower()

    # respostas curtas para campos numéricos
    if "total de aulas" in p or "quantas aulas" in p:
        return f"## 📘 {componente}\n\n**Total de aulas:** {valor(meta.get('total_aulas'))}"

    if "aulas semanais" in p or "por semana" in p:
        return f"## 📘 {componente}\n\n**Nº de aulas semanais:** {valor(meta.get('aulas_semanais'))}"

    if "carga horária presencial" in p or "ch presencial" in p:
        return f"## 📘 {componente}\n\n**Carga horária presencial:** {valor(meta.get('ch_presencial'))}"

    if "carga horária a distância" in p or "ch distância" in p or "ch distancia" in p:
        return f"## 📘 {componente}\n\n**Carga horária a distância:** {valor(meta.get('ch_distancia'))}"

    if "total de horas" in p or "carga horária total" in p:
        return f"## 📘 {componente}\n\n**Total de horas:** {valor(meta.get('total_horas'))}"

    if "docentes" in p or "professores" in p:
        return f"## 📘 {componente}\n\n**Quantidade de docentes:** {valor(meta.get('docentes'))}"

    if "laboratório" in p or "laboratorio" in p:
        return f"## 📘 {componente}\n\n**Carga horária em laboratório:** {valor(meta.get('ch_laboratorio'))}"

    # resposta específica textual
    campo_detectado = detectar_campo(pergunta)

    if campo_detectado:
        return f"""
## 📘 {componente}

### {campo}

{doc.page_content}
"""

    # resposta geral
    return f"""
## 📘 {componente}

### Resultado mais relevante

**Campo encontrado:** {campo}

**Ano:** {valor(meta.get("ano"))}  
**Sigla:** {valor(meta.get("sigla"))}  
**Nº de aulas semanais:** {valor(meta.get("aulas_semanais"))}  
**aulas semanais:** {valor(meta.get("aulas_semanais"))}  
**Total de aulas:** {valor(meta.get("total_aulas"))}  
**Carga horária presencial:** {valor(meta.get("ch_presencial"))}  
**Carga horária a distância:** {valor(meta.get("ch_distancia"))}  
**Total de horas:** {valor(meta.get("total_horas"))}  
**Quantidade de docentes:** {valor(meta.get("docentes"))}  
**Carga horária em laboratório:** {valor(meta.get("ch_laboratorio"))}  

---

{doc.page_content}
"""


# =========================
# INTERFACE
# =========================
st.set_page_config(
    page_title="Chatbot PPC",
    page_icon="📚",
    layout="wide"
)

st.markdown(
    """
    <div class="cabecalho-chat">
        <span>Chatbot Acadêmico</span>
    </div>
    """,
    unsafe_allow_html=True
)
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    campo_detectado = detectar_campo(pergunta)

    if campo_detectado:
        resultados = db.similarity_search_with_score(
            pergunta,
            k=5,
            filter={"campo": campo_detectado}
        )
    else:
        resultados = db.similarity_search_with_score(
            pergunta,
            k=5
        )

    if not resultados:
        st.warning("Nenhum resultado encontrado.")
    else:
        melhor_doc, score = resultados[0]

        st.subheader("Resposta")
        st.markdown(montar_resposta(melhor_doc, pergunta))

        st.subheader("Trechos encontrados")

        for doc, score in resultados:
            meta = doc.metadata

            st.markdown("---")
            st.write(f"**Disciplina:** {valor(meta.get('componente'))}")
            st.write(f"**Campo:** {valor(meta.get('campo'))}")
            st.write(f"**Score:** {score:.4f}")
            st.write(doc.page_content)