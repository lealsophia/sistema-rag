import re
import time
from datetime import datetime
from llm import gerar_resposta
import json
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import altair as alt
from firestore import listar_mensagens
import streamlit.components.v1 as components

# =========================
# CONFIGURAÇÕES
# =========================

ARQUIVO_DISCIPLINAS = "disciplinas.xlsx"

MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 10

CAMPOS_TEXTUAIS = [
    "Conhecimentos",
    "Ementa",
    "Objetivos",
    "Areas_integracao",
    "Conteudo",
    "Bibliografia_Basica",
    "Bibliografia_Complementar"
]


# =========================
# FUNÇÕES BÁSICAS
# =========================
def rodar_experimento_comparacao(
        df_perguntas,
        df_disciplinas,
        modelo,
        estrategia,
        chunk_size,
        overlap
):
    df_chunks = gerar_chunks(
        df_disciplinas,
        estrategia,
        chunk_size,
        overlap
    )

    embeddings_chunks = modelo.encode(
        [f"passage: {t}" for t in df_chunks["texto"].tolist()],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    df_resultados, df_logs = avaliar(
        df_perguntas,
        df_chunks,
        embeddings_chunks,
        modelo
    )

    recall_1 = df_resultados["recall_1"].mean()
    recall_5 = df_resultados["recall_5"].mean()
    recall_10 = df_resultados["recall_10"].mean()
    mrr = df_resultados["mrr"].mean()
    tempo_medio = df_resultados["tempo_busca_ms"].mean()

    resumo = {
        "estrategia": estrategia,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_chunks": len(df_chunks),
        "acertos_1_lugar": recall_1,
        "acertos_2_5_lugar": recall_5,
        "acertos_6_10_lugar": recall_10,
        "qualidade_media_posicao": mrr,
        "tempo_medio_busca_ms": tempo_medio
    }

    return resumo, df_resultados, df_logs, df_chunks


def botao_copiar_resposta(texto):
    texto_js = json.dumps(str(texto))

    components.html(
        f"""
        <div style="
            display: flex;
            justify-content: flex-end;
            margin-top: -8px;
        ">

            <button
                id="botao-copiar"
                style="
                    border: none;
                    background: transparent;
                    cursor: pointer;
                    font-size: 13px;
                    color: #888;
                    padding: 4px 8px;
                "
            >
                📋 Copiar
            </button>

        </div>

        <script>

            const botao = document.getElementById("botao-copiar");

            botao.addEventListener("click", async () => {{

                const texto = {texto_js};

                try {{

                    await navigator.clipboard.writeText(texto);

                }} catch (erro) {{

                    const area = document.createElement("textarea");

                    area.value = texto;
                    document.body.appendChild(area);

                    area.select();

                    document.execCommand("copy");

                    document.body.removeChild(area);

                }}

                botao.innerText = "✓ Copiado";

                setTimeout(() => {{
                    botao.innerText = "📋 Copiar";
                }}, 1500);

            }});

        </script>
        """,
        height=35
    )


def gerar_exportacao_conversa(titulo, mensagens):
    conteudo = f"# {titulo}\n\n"

    for mensagem in mensagens:

        if mensagem["role"] == "user":
            autor = "Usuário"
        else:
            autor = "Assistente"

        conteudo += f"## {autor}\n\n"
        conteudo += f"{mensagem['content']}\n\n"

    return conteudo


def nome_arquivo_conversa(titulo):
    nome = normalizar_busca(titulo)

    nome = re.sub(
        r"[^a-z0-9]+",
        "_",
        nome
    )

    nome = nome.strip("_")

    if not nome:
        nome = "conversa"

    return f"{nome}.md"


def grafico_barra_altair(
        df,
        x,
        y,
        titulo,
        eixo_y,
        eixo_x="Perguntas",
        escala_0_1=False
):
    escala_y = alt.Scale(domain=[0, 1]) if escala_0_1 else alt.Scale(zero=True)

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            x,
            sort=None,
            axis=alt.Axis(
                labelAngle=0,
                title=eixo_x
            )
        ),
        y=alt.Y(
            y,
            scale=escala_y,
            axis=alt.Axis(title=eixo_y)
        ),
        color=alt.Color(x, legend=None),
        tooltip=[x, y]
    ).properties(
        title=titulo,
        height=380
    )

    st.altair_chart(chart, use_container_width=True)


def grafico_linha_altair(df, x, y, titulo, eixo_y, escala_0_1=False):
    escala_y = alt.Scale(domain=[0, 1]) if escala_0_1 else alt.Scale(zero=True)

    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X(
            x,
            axis=alt.Axis(labelAngle=0, title=None)
        ),
        y=alt.Y(
            y,
            scale=escala_y,
            axis=alt.Axis(title=eixo_y)
        ),
        tooltip=[x, y]
    ).properties(
        title=titulo,
        height=380
    )

    st.altair_chart(chart, use_container_width=True)


def grafico_barras_multiplas_altair(df, coluna_categoria, colunas_metricas, titulo, eixo_y):
    df_long = df.melt(
        id_vars=[coluna_categoria],
        value_vars=colunas_metricas,
        var_name="Métrica",
        value_name="Valor"
    )

    chart = alt.Chart(df_long).mark_bar().encode(
        x=alt.X(
            coluna_categoria,
            sort=None,
            axis=alt.Axis(labelAngle=0, title=None)
        ),
        y=alt.Y(
            "Valor",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(title=eixo_y)
        ),
        color=alt.Color("Métrica", title="Métrica"),
        xOffset="Métrica",
        tooltip=[coluna_categoria, "Métrica", "Valor"]
    ).properties(
        title=titulo,
        height=420
    )

    st.altair_chart(chart, use_container_width=True)


def limpar(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return "Não se aplica"
    return str(valor).strip()


def limpar_ruido_pdf(texto):
    texto = str(texto)

    # Remove valores vazios comuns
    texto = texto.replace("nan", "")

    # Remove cabeçalhos e rodapés repetidos do PDF
    padroes_remover = [
        r"Projeto Pedagógico do Curso Técnico em Informática Integrado ao Ensino Médio",
        r"CUBATÃO",
        r"TÉCNICO EM INFORMÁTICA INTEGRADO AO ENSINO MÉDIO",
        r"Técnico em Informática Integrado ao Ensino Médio",
        r"Curso Técnico em Informática Integrado ao Ensino Médio",
    ]

    for padrao in padroes_remover:
        texto = re.sub(padrao, " ", texto, flags=re.IGNORECASE)

    # Remove números de página isolados
    texto = re.sub(r"\bPágina\s*\d+\b", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\b\d+\s*/\s*\d+\b", " ", texto)

    # Remove espaços repetidos, mas preserva quebras de linha
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()


def precisa_reescrever(pergunta, historico):
    if not historico.strip():
        return False

    pergunta_norm = normalizar_busca(pergunta)

    referencias = [
        "ela",
        "ele",
        "dela",
        "dele",
        "essa",
        "esse",
        "dessa",
        "desse",
        "isso",
        "aquilo",
        "e a de",
        "e o de",
        "e qual",
        "e quais",
        "e quanto",
        "e quantas",
        "e quantos"
    ]

    return any(
        termo in pergunta_norm
        for termo in referencias
    )


def normalizar(texto):
    texto = str(texto).lower().strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def precisa_reescrever(pergunta, historico):
    if not historico.strip():
        return False

    pergunta_norm = normalizar_busca(pergunta)

    referencias = [
        "ela",
        "ele",
        "dela",
        "dele",
        "essa",
        "esse",
        "dessa",
        "desse",
        "e a de",
        "e o de",
        "e qual",
        "e quais",
        "e quanto",
        "e quantas",
        "e quantos"
    ]

    return any(
        termo in pergunta_norm
        for termo in referencias
    )


import unicodedata


def normalizar_busca(texto):
    texto = str(texto).lower().strip()

    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    texto = re.sub(r"\s+", " ", texto)

    return texto


@st.cache_data
def carregar_disciplinas():
    return pd.read_excel(ARQUIVO_DISCIPLINAS)


@st.cache_resource
def carregar_modelo():
    return SentenceTransformer(MODELO)


@st.cache_resource
def preparar_base_chat(estrategia, chunk_size, overlap):
    st.write("⚙️ Gerando chunks e embeddings...")

    df_disciplinas = carregar_disciplinas()
    modelo = carregar_modelo()

    df_chunks = gerar_chunks(
        df_disciplinas,
        estrategia,
        chunk_size,
        overlap
    )

    embeddings_chunks = modelo.encode(
        [f"passage: {t}" for t in df_chunks["texto"].tolist()],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return df_chunks, embeddings_chunks


def validar_perguntas(df):
    colunas_obrigatorias = [
        "id",
        "pergunta",
        "tipo_pergunta",
        "componente_esperado",
        "campo_esperado"
    ]

    faltando = [
        col for col in colunas_obrigatorias
        if col not in df.columns
    ]

    if faltando:
        st.error(f"O CSV está sem as colunas obrigatórias: {faltando}")
        return None

    return df


# =========================
# SPLITTER
# =========================

def criar_splitter(estrategia, chunk_size, overlap):
    if estrategia == "fixo":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separator=" "
        )

    if estrategia == "recursivo":
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

    if estrategia == "semântico":
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=[
                ". ",
                "; ",
                "\n",
                " ",
                ""
            ]
        )

    raise ValueError("Estratégia inválida.")


# =========================
# GERAR CHUNKS
# =========================

def gerar_chunks(df, estrategia, chunk_size, overlap):
    chunks = []

    for _, row in df.iterrows():

        base_info = {
            "Curso": limpar(row.get("Curso")),
            "Componente": limpar(row.get("Componente")),
            "Tipo": limpar(row.get("Tipo")),
            "Núcleo": limpar(row.get("Núcleo")),
            "Ano": limpar(row.get("Ano")),
            "Sigla": limpar(row.get("Sigla")),
            "Aulas_semanais": limpar(row.get("Aulas_semanais")),
            "Total_aulas": limpar(row.get("Total_aulas")),
            "CH_presencial": limpar(row.get("CH_presencial")),
            "CH_distancia": limpar(row.get("CH_distancia")),
            "Total_horas": limpar(row.get("Total_horas")),
            "Docentes": limpar(row.get("Docentes")),
            "CH_laboratorio": limpar(row.get("CH_laboratorio")),
        }

        texto_estrutural = f"""
Curso: {base_info["Curso"]}
Disciplina: {base_info["Componente"]}
Tipo: {base_info["Tipo"]}
Núcleo: {base_info["Núcleo"]}
Ano: {base_info["Ano"]}
Sigla: {base_info["Sigla"]}

Nº de aulas semanais: {base_info["Aulas_semanais"]}
Total de aulas: {base_info["Total_aulas"]}
Carga horária presencial: {base_info["CH_presencial"]}
Carga horária a distância: {base_info["CH_distancia"]}
Total de horas: {base_info["Total_horas"]}
Quantidade de docentes: {base_info["Docentes"]}
Carga horária em laboratório: {base_info["CH_laboratorio"]}
""".strip()

        texto_estrutural = limpar_ruido_pdf(texto_estrutural)

        chunks.append({
            **base_info,
            "estrategia": estrategia,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "campo": "Estrutural",
            "chunk_id": 0,
            "texto_original_campo": texto_estrutural,
            "texto_chunkado": texto_estrutural,
            "texto": texto_estrutural,
            "tamanho_chunk": len(texto_estrutural)
        })

        for campo in CAMPOS_TEXTUAIS:
            texto_original = limpar(row.get(campo))
            texto_original = limpar_ruido_pdf(texto_original)

            if (
                    texto_original == "Não se aplica"
                    or texto_original.strip() == ""
                    or texto_original.lower() == "nan"
            ):
                continue

            splitter = criar_splitter(
                estrategia,
                chunk_size,
                overlap
            )

            partes = splitter.split_text(texto_original)

            for i, parte in enumerate(partes):
                parte = limpar_ruido_pdf(parte)

                if not parte:
                    continue

                texto_chunk = f"""
                Disciplina: {base_info["Componente"]}
                Ano: {base_info["Ano"]}
                Campo: {campo}

                {parte}
                """.strip()

                texto_chunk = limpar_ruido_pdf(texto_chunk)

                chunks.append({
                    **base_info,
                    "estrategia": estrategia,
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                    "campo": campo,
                    "chunk_id": i,
                    "texto_original_campo": texto_original,
                    "texto_chunkado": parte,
                    "texto": texto_chunk,
                    "tamanho_chunk": len(texto_chunk)
                })

    df_chunks = pd.DataFrame(chunks)

    if not df_chunks.empty:
        df_chunks = df_chunks.drop_duplicates(
            subset=["Componente", "campo", "chunk_id", "texto_chunkado"]
        ).reset_index(drop=True)

    return df_chunks


# =========================
# BUSCA
# =========================

def buscar(pergunta, df_chunks, embeddings_chunks, modelo, top_k=TOP_K):
    emb_pergunta = modelo.encode(
        [f"query: {pergunta}"],
        normalize_embeddings=True
    )

    similaridades = cosine_similarity(
        emb_pergunta,
        embeddings_chunks
    )[0]

    ordem = similaridades.argsort()[::-1][:top_k]

    resultados = df_chunks.iloc[ordem].copy()
    resultados["rank"] = range(1, len(resultados) + 1)
    resultados["similaridade"] = similaridades[ordem]
    resultados["score_final_retrieval"] = similaridades[ordem]

    return resultados


def buscar_chat(
        pergunta,
        df_chunks,
        embeddings_chunks,
        modelo,
        top_k=5,
        top_candidatos=30,
        candidatos_por_area=2
):
    """
    Busca otimizada para o chatbot.

    Fluxo:
    1. Busca semântica.
    2. Recupera candidatos gerais.
    3. Identifica componentes explícitos.
    4. Identifica campo solicitado.
    5. Identifica áreas mencionadas.
    6. Faz reranking normal.
    7. Em perguntas interdisciplinares:
       - adiciona até N candidatos relevantes por área;
       - prioriza o campo solicitado;
       - garante cobertura das áreas.
    8. Retorna Top-K final.

    IMPORTANTE:
    Esta função é exclusiva do chatbot.
    A função buscar() dos experimentos permanece inalterada.
    """

    # ==========================================
    # 1. EMBEDDING DA PERGUNTA
    # ==========================================

    emb_pergunta = modelo.encode(
        [f"query: {pergunta}"],
        normalize_embeddings=True
    )

    similaridades = cosine_similarity(
        emb_pergunta,
        embeddings_chunks
    )[0]

    pergunta_normalizada = normalizar_busca(pergunta)

    # Base completa com score semântico
    base_com_scores = df_chunks.copy()
    base_com_scores["similaridade"] = similaridades

    # ==========================================
    # 2. CANDIDATOS GERAIS
    # ==========================================

    quantidade_candidatos = min(
        top_candidatos,
        len(df_chunks)
    )

    ordem = similaridades.argsort()[::-1][
        :quantidade_candidatos
    ]

    candidatos = df_chunks.iloc[ordem].copy()
    candidatos["similaridade"] = similaridades[ordem]

    # ==========================================
    # 3. COMPONENTES EXPLICITAMENTE CITADOS
    # ==========================================

    componentes = (
        df_chunks["Componente"]
        .dropna()
        .unique()
        .tolist()
    )

    componentes = sorted(
        componentes,
        key=lambda x: len(str(x)),
        reverse=True
    )

    componentes_identificados = []

    for componente in componentes:

        componente_norm = normalizar_busca(componente)

        if componente_norm in pergunta_normalizada:
            componentes_identificados.append(componente)

    # ==========================================
    # 4. CAMPO SOLICITADO
    # ==========================================

    campos_possiveis = {
        "bibliografia complementar":
            "Bibliografia_Complementar",

        "bibliografia básica":
            "Bibliografia_Basica",

        "bibliografia basica":
            "Bibliografia_Basica",

        "áreas de integração":
            "Areas_integracao",

        "areas de integração":
            "Areas_integracao",

        "areas de integracao":
            "Areas_integracao",

        "área de integração":
            "Areas_integracao",

        "area de integracao":
            "Areas_integracao",

        "conteúdo programático":
            "Conteudo",

        "conteudo programatico":
            "Conteudo",

        "conteúdo":
            "Conteudo",

        "conteudo":
            "Conteudo",

        "conhecimentos":
            "Conhecimentos",

        "conhecimento":
            "Conhecimentos",

        "objetivos":
            "Objetivos",

        "objetivo":
            "Objetivos",

        "ementa":
            "Ementa"
    }

    campo_identificado = None

    for termo in sorted(
            campos_possiveis,
            key=len,
            reverse=True
    ):

        if (
                normalizar_busca(termo)
                in pergunta_normalizada
        ):
            campo_identificado = campos_possiveis[termo]
            break

    # ==========================================
    # 5. PERGUNTAS ESTRUTURAIS
    # ==========================================

    termos_estruturais = [
        "carga horaria",
        "quantas horas",
        "total de horas",
        "aulas semanais",
        "quantas aulas",
        "total de aulas",
        "qual ano",
        "em que ano",
        "ano e oferecida",
        "ano e oferecido",
        "sigla",
        "quantidade de docentes",
        "quantos docentes",
        "quantos professores",
        "laboratorio",
        "carga horaria presencial",
        "carga horaria a distancia"
    ]

    if any(
            termo in pergunta_normalizada
            for termo in termos_estruturais
    ):
        campo_identificado = "Estrutural"

    # ==========================================
    # 6. ÁREAS MENCIONADAS NA PERGUNTA
    # ==========================================

    areas_possiveis = {
        "matematica": "matematica",
        "informatica": "informatica",
        "geografia": "geografia",
        "historia": "historia",
        "biologia": "biologia",
        "fisica": "fisica",
        "quimica": "quimica",
        "portugues": "portugues",
        "lingua portuguesa": "portugues",
        "ingles": "ingles",
        "lingua inglesa": "ingles",
        "espanhol": "espanhol",
        "lingua espanhola": "espanhol"
    }

    areas_identificadas = []

    for termo, area in areas_possiveis.items():

        if (
                normalizar_busca(termo)
                in pergunta_normalizada
        ):

            if area not in areas_identificadas:
                areas_identificadas.append(area)

    # ==========================================
    # 7. CLASSIFICA COMPONENTES POR ÁREA
    # ==========================================

    componentes_por_area = {
        area: set()
        for area in set(areas_possiveis.values())
    }

    chunks_conhecimentos = df_chunks[
        df_chunks["campo"]
        .apply(normalizar_busca)
        == "conhecimentos"
        ]

    for _, linha in chunks_conhecimentos.iterrows():

        componente = linha["Componente"]

        componente_norm = normalizar_busca(
            componente
        )

        texto_norm = normalizar_busca(
            linha["texto"]
        )

        # --------------------------
        # Áreas propedêuticas
        # --------------------------

        if "matematica" in componente_norm:
            componentes_por_area[
                "matematica"
            ].add(componente)

        if "geografia" in componente_norm:
            componentes_por_area[
                "geografia"
            ].add(componente)

        if "historia" in componente_norm:
            componentes_por_area[
                "historia"
            ].add(componente)

        if "biologia" in componente_norm:
            componentes_por_area[
                "biologia"
            ].add(componente)

        if "fisica" in componente_norm:
            componentes_por_area[
                "fisica"
            ].add(componente)

        if "quimica" in componente_norm:
            componentes_por_area[
                "quimica"
            ].add(componente)

        if (
                "portugues" in componente_norm
                or "lingua portuguesa" in componente_norm
        ):
            componentes_por_area[
                "portugues"
            ].add(componente)

        if (
                "ingles" in componente_norm
                or "lingua inglesa" in componente_norm
        ):
            componentes_por_area[
                "ingles"
            ].add(componente)

        if (
                "espanhol" in componente_norm
                or "espanhola" in componente_norm
        ):
            componentes_por_area[
                "espanhol"
            ].add(componente)

        # --------------------------
        # Informática
        # --------------------------

        if (
                "/informatica" in texto_norm
                or "informatica/" in texto_norm
        ):
            componentes_por_area[
                "informatica"
            ].add(componente)

    # ==========================================
    # 8. RERANKING NORMAL
    # ==========================================

    candidatos["bonus_componente"] = 0.0
    candidatos["bonus_campo"] = 0.0

    if componentes_identificados:
        componentes_norm = {
            normalizar_busca(c)
            for c in componentes_identificados
        }

        mascara_componente = (
            candidatos["Componente"]
            .apply(normalizar_busca)
            .isin(componentes_norm)
        )

        candidatos.loc[
            mascara_componente,
            "bonus_componente"
        ] = 0.15

    if campo_identificado is not None:
        mascara_campo = (
                candidatos["campo"]
                .apply(normalizar_busca)
                == normalizar_busca(campo_identificado)
        )

        candidatos.loc[
            mascara_campo,
            "bonus_campo"
        ] = 0.10

    candidatos["score_final_retrieval"] = (
            candidatos["similaridade"]
            + candidatos["bonus_componente"]
            + candidatos["bonus_campo"]
    )

    candidatos = candidatos.sort_values(
        "score_final_retrieval",
        ascending=False
    )

    # ==========================================
    # 9. ADICIONA CANDIDATOS INTERDISCIPLINARES
    # ==========================================

    if len(areas_identificadas) >= 2:

        indices_extras = []

        for area in areas_identificadas:

            componentes_area = (
                componentes_por_area.get(
                    area,
                    set()
                )
            )

            if not componentes_area:
                continue

            candidatos_area = base_com_scores[
                base_com_scores["Componente"]
                .isin(componentes_area)
            ].copy()

            if candidatos_area.empty:
                continue

            # Se há campo explícito, usa primeiro
            # apenas o campo solicitado.
            if campo_identificado is not None:

                candidatos_area_campo = (
                    candidatos_area[
                        candidatos_area["campo"]
                        .apply(normalizar_busca)
                        == normalizar_busca(
                            campo_identificado
                        )
                        ]
                        .copy()
                )

                if not candidatos_area_campo.empty:
                    candidatos_area = candidatos_area_campo

            candidatos_area = (
                candidatos_area
                .sort_values(
                    "similaridade",
                    ascending=False
                )
                .head(candidatos_por_area)
            )

            indices_extras.extend(
                candidatos_area.index.tolist()
            )

        if indices_extras:

            indices_extras = list(
                dict.fromkeys(indices_extras)
            )

            extras = base_com_scores.loc[
                indices_extras
            ].copy()

            extras["bonus_componente"] = 0.0
            extras["bonus_campo"] = 0.0

            if campo_identificado is not None:
                mascara_extra_campo = (
                        extras["campo"]
                        .apply(normalizar_busca)
                        == normalizar_busca(
                    campo_identificado
                )
                )

                extras.loc[
                    mascara_extra_campo,
                    "bonus_campo"
                ] = 0.10

            extras[
                "score_final_retrieval"
            ] = (
                    extras["similaridade"]
                    + extras["bonus_componente"]
                    + extras["bonus_campo"]
            )

            candidatos = (
                pd.concat(
                    [
                        candidatos,
                        extras
                    ]
                )
                .loc[
                    lambda df:
                    ~df.index.duplicated(
                        keep="first"
                    )
                ]
            )

            candidatos = candidatos.sort_values(
                "score_final_retrieval",
                ascending=False
            )

    # ==========================================
    # 10. MONTA TOP-K COM COBERTURA
    # ==========================================

    selecionados = []
    indices_selecionados = set()

    if len(areas_identificadas) >= 2:

        # Primeiro garante pelo menos
        # um resultado de cada área.
        for area in areas_identificadas:

            componentes_area = (
                componentes_por_area.get(
                    area,
                    set()
                )
            )

            if not componentes_area:
                continue

            candidatos_area = candidatos[
                candidatos["Componente"]
                .isin(componentes_area)
            ]

            if campo_identificado is not None:

                candidatos_area_campo = (
                    candidatos_area[
                        candidatos_area["campo"]
                        .apply(normalizar_busca)
                        == normalizar_busca(
                            campo_identificado
                        )
                        ]
                )

                if not candidatos_area_campo.empty:
                    candidatos_area = (
                        candidatos_area_campo
                    )

            if not candidatos_area.empty:

                melhor = candidatos_area.iloc[0]

                if (
                        melhor.name
                        not in indices_selecionados
                ):
                    selecionados.append(melhor)

                    indices_selecionados.add(
                        melhor.name
                    )

    # ==========================================
    # 11. COMPLETA COM MELHORES SCORES
    # ==========================================

    for indice, linha in candidatos.iterrows():

        if len(selecionados) >= top_k:
            break

        if indice in indices_selecionados:
            continue

        selecionados.append(
            linha.copy()
        )

        indices_selecionados.add(
            indice
        )

    resultados = pd.DataFrame(
        selecionados[:top_k]
    )

    # ==========================================
    # 12. RANK FINAL
    # ==========================================

    resultados = resultados.copy()

    resultados["rank"] = range(
        1,
        len(resultados) + 1
    )

    return resultados


# =========================
# AVALIAÇÃO
# =========================

def avaliar(df_perguntas, df_chunks, embeddings_chunks, modelo):
    resultados_finais = []
    logs = []

    for _, pergunta_row in df_perguntas.iterrows():
        pergunta = pergunta_row["pergunta"]

        inicio = time.time()

        resultados = buscar(
            pergunta,
            df_chunks,
            embeddings_chunks,
            modelo,
            top_k=TOP_K
        )

        tempo = (time.time() - inicio) * 1000

        componente_esperado = normalizar(pergunta_row.get("componente_esperado"))
        campo_esperado = normalizar(pergunta_row.get("campo_esperado"))

        rank_correto = None

        for _, r in resultados.iterrows():
            componente_ok = normalizar(r.get("Componente")) == componente_esperado
            campo_ok = normalizar(r.get("campo")) == campo_esperado

            if componente_ok and campo_ok:
                rank_correto = int(r["rank"])
                break

        recall_1 = 1 if rank_correto == 1 else 0

        recall_5 = 1 if (
                rank_correto is not None
                and 2 <= rank_correto <= 5
        ) else 0

        recall_10 = 1 if (
                rank_correto is not None
                and 6 <= rank_correto <= 10
        ) else 0

        mrr = 1 / rank_correto if rank_correto else 0

        top1 = resultados.iloc[0]

        resultados_finais.append({
            "id": pergunta_row.get("id"),
            "pergunta": pergunta,
            "tipo_pergunta": pergunta_row.get("tipo_pergunta"),
            "componente_esperado": pergunta_row.get("componente_esperado"),
            "campo_esperado": pergunta_row.get("campo_esperado"),
            "componente_top1": top1.get("Componente"),
            "campo_top1": top1.get("campo"),
            "rank_correto": rank_correto if rank_correto else "Não encontrado",
            "recall_1": recall_1,
            "recall_5": recall_5,
            "recall_10": recall_10,
            "mrr": mrr,
            "tempo_busca_ms": tempo
        })

        for _, r in resultados.iterrows():
            logs.append({
                "id_pergunta": pergunta_row.get("id"),
                "pergunta": pergunta,
                "rank": r.get("rank"),
                "componente": r.get("Componente"),
                "campo": r.get("campo"),
                "similaridade": r.get("similaridade"),
                "score_final_retrieval": r.get("score_final_retrieval"),
                "texto_chunkado": r.get("texto_chunkado")
            })

    return pd.DataFrame(resultados_finais), pd.DataFrame(logs)


# =========================
# STREAMLIT
# =========================

st.set_page_config(
    page_title="Assistente Acadêmico",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from auth import login, register, logout, usuario_logado

# =========================
# LOGIN
# =========================

if not usuario_logado():

    st.title("Login")

    email = st.text_input("Email")
    senha = st.text_input("Senha", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Entrar"):
            ok, msg = login(email, senha)

            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with col2:
        if st.button("Criar conta"):
            ok, msg = register(email, senha)

            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.stop()

# =========================
# INTERFACE
# =========================

st.markdown("""
<style>

/* ==========================================
   SIDEBAR GERAL
   ========================================== */
[data-testid="stSidebar"] {
    background-color: #17171f;
    min-width: 360px !important;
    width: 360px !important;
}

[data-testid="stSidebarContent"] {
    width: 360px !important;
    padding-left: 16px;
    padding-right: 16px;
}

/* ==========================================
   TOPO DO ASSISTENTE
   ========================================== */
.topo-assistente {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 4px 6px 4px;
    margin-bottom: 4px;
}

.topo-assistente-icone {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
}

.topo-assistente-texto {
    display: flex;
    flex-direction: column;
    min-width: 0;
}

.topo-assistente-nome {
    font-size: 20px;
    font-weight: 600;
    line-height: 1.2;
}

.topo-assistente-subtitulo {
    margin-top: 4px;
    font-size: 12px;
    opacity: 0.55;
}

/* ==========================================
   MODO
   ========================================== */
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    margin-top: 8px;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    min-height: 40px !important;
    background: #0f1016 !important;
    border: none !important;
    padding: 0 8px !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span {
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ==========================================
   BOTÕES NORMAIS
   ========================================== */
[data-testid="stSidebar"] .stButton > button {
    border: none;
    background: transparent;
    border-radius: 7px;
    width: 100%;
    min-height: 36px;
    padding: 7px 9px;
    justify-content: flex-start !important;
    text-align: left !important;
    transition: background-color 0.15s ease;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255,255,255,0.07);
}

[data-testid="stSidebar"] .stButton > button p {
    text-align: left !important;
    width: 100% !important;
    margin: 0 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

[data-testid="stSidebar"] .stButton > button > div {
    width: 100% !important;
    justify-content: flex-start !important;
}

/* ==========================================
   NOVA CONVERSA
   ========================================== */
[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #6d46d9 !important;
    border: none !important;
    border-radius: 8px !important;
    min-height: 42px !important;
    justify-content: center !important;
}

[data-testid="stSidebar"] button[kind="primary"] p {
    text-align: center !important;
    font-weight: 500;
}

/* ==========================================
   CONVERSAS
   ========================================== */
[class*="st-key-convrow_"] {
    border-radius: 8px;
    padding: 1px 2px;
    margin-bottom: -4px;
}

[class*="st-key-convrow_"]:hover {
    background-color: rgba(255,255,255,0.05);
}

[class*="st-key-convrow_"] .stButton > button {
    justify-content: flex-start !important;
    text-align: left !important;
    padding-left: 6px !important;
}

[class*="st-key-convrow_"] .stButton > button p {
    text-align: left !important;
    width: 100% !important;
    margin: 0 !important;
}

/* ==========================================
   MENU DA CONVERSA
   ========================================== */
[class*="st-key-convrow_"] [data-testid="stPopover"] button {
    opacity: 0;
    background: transparent !important;
    border: none !important;
    padding: 4px !important;
    min-width: 24px !important;
    transition: opacity 0.15s ease;
}

[class*="st-key-convrow_"]:hover [data-testid="stPopover"] button {
    opacity: 1;
}

/* ==========================================
   USUÁRIO E SAIR FIXOS NO RODAPÉ
   ========================================== */

.usuario-fixo {
    position: fixed;
    bottom: 48px;
    left: 16px;

    width: 328px;

    display: flex;
    align-items: center;
    gap: 10px;

    padding: 8px 4px;

    background: #17171f;

    z-index: 1000;
}

.usuario-icone {
    width: 34px;
    height: 34px;

    border-radius: 50%;

    background: #2b2b35;

    display: flex;
    align-items: center;
    justify-content: center;

    flex-shrink: 0;

    font-size: 16px;
}

.usuario-email {
    font-size: 12px;

    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}


/* BOTÃO SAIR */

[data-testid="stSidebar"] [class*="st-key-botao_sair"] {
    position: fixed;

    bottom: 8px;
    left: 16px;

    width: 328px;

    z-index: 1001;
}

[data-testid="stSidebar"] [class*="st-key-botao_sair"] button {
    justify-content: flex-start !important;
    text-align: left !important;
    padding-left: 9px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOGIN
# =========================

from auth import login, register, logout, usuario_logado

if not usuario_logado():
    st.title("Login")
    email = st.text_input("Email")
    senha = st.text_input("Senha", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Entrar"):
            ok, msg = login(email, senha)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    with col2:
        if st.button("Criar conta"):
            ok, msg = register(email, senha)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    st.stop()

# =========================
# DADOS / MODELO
# =========================

df_disciplinas = carregar_disciplinas()
modelo = carregar_modelo()

# =========================
# NAVEGAÇÃO PRINCIPAL
# =========================

opcoes_modo = {
    "Chatbot Acadêmico": "Chatbot interativo",
    "Avaliação do RAG": "Experimento com arquivo de perguntas",
    "Comparar estratégias": "Comparar estratégias de chunking"
}

st.sidebar.markdown(
    """
    <div class="topo-assistente">
        <div class="topo-assistente-icone">🤖</div>
        <div class="topo-assistente-texto">
            <div class="topo-assistente-nome">Assistente Acadêmico</div>
            <div class="topo-assistente-subtitulo">IFSP</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

modo_visual = st.sidebar.selectbox(
    "Modo",
    list(opcoes_modo.keys()),
    index=0,
    label_visibility="collapsed",
    key="modo_principal"
)

modo_uso = opcoes_modo[modo_visual]

estrategia = None
chunk_size = None
overlap = None

# ==========================================
# PARÂMETROS DE CHUNKING
# ==========================================

# Só aparecem nos modos de avaliação.
# No Chatbot Acadêmico eles ficam ocultos.

if modo_uso == "Experimento com arquivo de perguntas":

    estrategia = st.sidebar.selectbox(
        "Estratégia de chunking",
        ["recursivo", "fixo", "semântico"]
    )

    chunk_size = st.sidebar.slider(
        "Tamanho do chunk",
        200,
        1500,
        700,
        100
    )

    overlap = st.sidebar.slider(
        "Sobreposição (Overlap)",
        0,
        500,
        120,
        50
    )

    with st.sidebar.expander("❓ O que significam esses parâmetros?"):
        st.write("""
        **Tamanho do chunk**

        Define o tamanho máximo de cada chunk gerado a partir do documento.
        Neste sistema, esse valor é medido em **caracteres**.

        **Sobreposição**

        Define quantos caracteres do final de um chunk serão repetidos
        no início do próximo.

        Isso ajuda a preservar o contexto quando uma informação fica
        dividida entre dois chunks.

        Exemplo:

        Se o tamanho do chunk for 700 e a sobreposição for 120,
        cada trecho terá até 700 caracteres, e 120 caracteres poderão
        se repetir entre trechos consecutivos.
        """)

# ==========================================
# RODAPÉ DA SIDEBAR
# ==========================================

email_usuario = st.session_state["user"]["email"]

html_usuario = f"""
<div class="usuario-fixo">
<div class="usuario-icone">👤</div>
<div class="usuario-email">{email_usuario}</div>
</div>
"""

st.sidebar.markdown(
    html_usuario,
    unsafe_allow_html=True
)

if st.sidebar.button(
    "🚪 Sair",
    use_container_width=True,
    key="botao_sair"
):
    logout()
    st.rerun()

st.sidebar.markdown("</div></div>", unsafe_allow_html=True)

# =========================
# MODO EXPERIMENTO
# =========================


if modo_uso == "Experimento com arquivo de perguntas":

    st.subheader("📁 Importar arquivo de perguntas")

    with st.expander("ℹ️ Como deve ser a planilha de perguntas?"):

        st.info("""
    A planilha deve conter:

    - id
    - pergunta
    - componente_esperado
    - campo_esperado

    Formatos aceitos:
    - CSV (.csv)
    - XLSX (.xlsx)
    - XLS (.xls)
    """)

        exemplo = pd.DataFrame({
            "id": [1, 2],
            "pergunta": [
                "Qual é a ementa de Algoritmos?",
                "Quais são os objetivos de Inteligência Artificial?"
            ],
            "tipo_pergunta": [
                "ementa",
                "objetivos"
            ],
            "componente_esperado": [
                "Algoritmos",
                "Inteligência Artificial"
            ],
            "campo_esperado": [
                "Ementa",
                "Objetivos"
            ]
        })

        st.dataframe(exemplo)
    arquivo_perguntas = st.file_uploader(
        "Envie um arquivo de perguntas",
        type=["csv", "xlsx", "xls"]
    )

    df_perguntas = None

    if arquivo_perguntas is not None:

        extensao = arquivo_perguntas.name.split(".")[-1].lower()

        if extensao == "csv":
            df_perguntas = pd.read_csv(arquivo_perguntas)

        elif extensao in ["xlsx", "xls"]:
            df_perguntas = pd.read_excel(arquivo_perguntas)

        else:
            st.error("Formato não suportado.")
            df_perguntas = None

        if df_perguntas is not None:
            df_perguntas = validar_perguntas(df_perguntas)

    if df_perguntas is not None:
        st.metric("Perguntas carregadas", len(df_perguntas))

        with st.expander("Ver perguntas importadas"):
            st.dataframe(df_perguntas)

        if st.button("Rodar experimento"):
            with st.spinner("Gerando chunks..."):
                df_chunks = gerar_chunks(
                    df_disciplinas,
                    estrategia,
                    chunk_size,
                    overlap
                )

            with st.spinner("Gerando embeddings..."):
                embeddings_chunks = modelo.encode(
                    [f"passage: {t}" for t in df_chunks["texto"].tolist()],
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

            with st.spinner("Avaliando perguntas..."):
                df_resultados, df_logs = avaliar(
                    df_perguntas,
                    df_chunks,
                    embeddings_chunks,
                    modelo
                )

            recall_1 = df_resultados["recall_1"].mean()
            recall_5 = df_resultados["recall_5"].mean()
            recall_10 = df_resultados["recall_10"].mean()
            mrr = df_resultados["mrr"].mean()
            tempo_medio = df_resultados["tempo_busca_ms"].mean()

            resumo = pd.DataFrame([{
                "estrategia": estrategia,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "top_k": TOP_K,
                "total_chunks": len(df_chunks),
                "recall_1": recall_1,
                "recall_5": recall_5,
                "recall_10": recall_10,
                "mrr": mrr,
                "tempo_medio_ms": tempo_medio,
                "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])

            resumo_tipo = df_resultados.groupby("tipo_pergunta")[
                ["recall_1", "recall_5", "recall_10", "mrr", "tempo_busca_ms"]
            ].mean().reset_index()

            resumo_tipo["tempo_busca_ms"] = resumo_tipo["tempo_busca_ms"].round(2)

            st.session_state.df_resultados = df_resultados
            st.session_state.df_logs = df_logs
            st.session_state.df_chunks = df_chunks
            st.session_state.resumo = resumo
            st.session_state.resumo_tipo = resumo_tipo

            st.success("Experimento finalizado!")

    if "df_resultados" in st.session_state:

        df_resultados = st.session_state.df_resultados
        df_logs = st.session_state.df_logs
        df_chunks = st.session_state.df_chunks
        resumo = st.session_state.resumo
        resumo_tipo = st.session_state.resumo_tipo

        recall_1 = resumo.loc[0, "recall_1"]
        recall_5 = resumo.loc[0, "recall_5"]
        recall_10 = resumo.loc[0, "recall_10"]
        mrr = resumo.loc[0, "mrr"]
        tempo_medio = resumo.loc[0, "tempo_medio_ms"]

        st.subheader("🎯 Percentual de Acerto das Respostas")
        st.caption(
            "Desempenho da estratégia de chunking na recuperação das respostas esperadas."
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Recall@1 ",
            f"{recall_1:.2%}",
            help="Percentual de perguntas em que a resposta correta apareceu exatamente na 1ª posição."
        )

        c2.metric(
            "Recall@5 ",
            f"{recall_5:.2%}",
            help="Percentual de perguntas em que a resposta correta apareceu entre a 2ª e a 5ª posição."
        )

        c3.metric(
            "Recall@10 ",
            f"{recall_10:.2%}",
            help="Percentual de perguntas em que a resposta correta apareceu entre a 6ª e a 10ª posição."
        )

        c4.metric(
            "MRR ",
            f"{mrr:.3f}",
            help="Mede a posição em que a resposta correta apareceu. Quanto mais próximo de 1, melhor."
        )

        c5.metric(
            "Tempo médio ",
            f"{tempo_medio:.2f} ms",
            help="Tempo médio necessário para processar uma pergunta e recuperar os chunks mais relevantes."
        )

        st.subheader("📊 Desempenho Geral")
        resumo_exibir = resumo.rename(columns={
            "estrategia": "Estratégia",
            "chunk_size": "Tamanho do Chunk",
            "overlap": "Sobreposição",
            "top_k": "Top-K",
            "total_chunks": "Quantidade de Chunks",
            "recall_1": "Recall@1",
            "recall_5": "Recall@5",
            "recall_10": "Recall@10",
            "mrr": "MRR",
            "tempo_medio_ms": "Tempo Médio de Busca (ms)",
            "data": "Data do Experimento"
        })

        st.dataframe(resumo_exibir)

        if st.button("📊 Gerar gráfico do desempenho geral"):
            grafico_resumo = pd.DataFrame({
                "Métrica": ["Recall@1", "Recall@5", "Recall@10", "MRR"],
                "Valor": [recall_1, recall_5, recall_10, mrr]
            })

            grafico_barra_altair(
                grafico_resumo,
                "Métrica",
                "Valor",
                "Resumo geral das métricas",
                "Valor da métrica",
                escala_0_1=True
            )

        col_resumo, col_resumo_help = st.columns([0.92, 0.08])

        with col_resumo:
            st.subheader("📋 Desempenho por Categoria de Pergunta")

        with col_resumo_help:
            with st.popover("❔"):
                st.write("""
Agrupa os resultados por categoria de pergunta.

A coluna tempo_busca_ms mostra o tempo médio, em milissegundos, necessário para recuperar os chunks de cada tipo de pergunta.
""")
                resumo_tipo_exibir = resumo_tipo.rename(columns={
                    "tipo_pergunta": "Tipo da pergunta",
                    "recall_1": "Recall@1",
                    "recall_5": "Recall@5",
                    "recall_10": "Recall@10",
                    "mrr": "MRR",
                    "tempo_busca_ms": "Tempo Médio de Busca (ms)"
                })

        st.dataframe(resumo_tipo_exibir)

        if st.button("📊 Gerar gráfico do desempenho por categoria de pergunta"):
            grafico_barras_multiplas_altair(
                resumo_tipo,
                "tipo_pergunta",
                ["recall_1", "recall_5", "recall_10", "mrr"],
                "Métricas por tipo de pergunta",
                "Valor da métrica"
            )

        col_resultados, col_resultados_help = st.columns([0.92, 0.08])

        with col_resultados:
            st.subheader("🔍 Resultado de Cada Pergunta")

        with col_resultados_help:
            with st.popover("❔"):
                st.write("""
Mostra o desempenho do sistema para cada pergunta individualmente.

Permite verificar o componente esperado, o campo esperado, o resultado retornado em primeiro lugar, o rank correto e o tempo de busca.

O campo 'Rank Correto' mostra em qual posição o chunk correto apareceu no ranking de resultados.

O campo 'Melhor disciplina encontrada' mostra qual disciplina foi recuperada na primeira posição do ranking para a pergunta realizada.

O campo campo_top1 mostra qual campo textual foi recuperado em primeiro lugar.
""")
                df_resultados_exibir = df_resultados.rename(columns={
                    "componente_top1": "Melhor disciplina encontrada",
                    "campo_top1": "Melhor campo encontrado",
                    "rank_correto": "Posição da resposta correta",
                    "tempo_busca_ms": "Tempo de busca (ms)",
                    "componente_esperado": "Disciplina esperada",
                    "campo_esperado": "Campo esperado",
                    "tipo_pergunta": "Categoria da pergunta",
                    "recall_1": "Recall@1",
                    "recall_5": "Recall@5",
                    "recall_10": "Recall@10",
                    "mrr": "MRR"
                })

        st.dataframe(df_resultados_exibir)

        if st.button("📊 Gerar gráfico dos resultado de cada pergunta"):
            graf = df_resultados.copy()

            graf["id"] = graf["id"].astype(str)

            graf["rank_correto"] = graf["rank_correto"].replace(
                "Não encontrado",
                0
            )

            graf["rank_correto"] = pd.to_numeric(
                graf["rank_correto"],
                errors="coerce"
            ).fillna(0)

            grafico_barra_altair(
                graf,
                "id",
                "rank_correto",
                "Rank correto por pergunta",
                "Posição da resposta correta",
                eixo_x="Perguntas",
                escala_0_1=False
            )

            grafico_barra_altair(
                graf,
                "id",
                "tempo_busca_ms",
                "Tempo de busca por pergunta",
                "Tempo de busca (ms)",
                eixo_x="Perguntas",
                escala_0_1=False
            )

        col_logs, col_logs_help = st.columns([0.92, 0.08])

        with col_logs:
            st.subheader("📚 Resultados Recuperados pelo Sistema")

        with col_logs_help:
            with st.popover("❔"):
                st.write("""
Logs Top-K registra os K resultados mais relevantes recuperados para cada pergunta.

Neste experimento, Top-K = 10.
""")
                df_logs_exibir = df_logs.rename(columns={
                    "similaridade": "Similaridade semântica",
                    "score_final_retrieval": "Pontuação final",
                    "rank": "Rank",
                    "componente": "Componente",
                    "campo": "Campo",
                    "texto_chunkado": "Trecho recuperado"

                })

        st.dataframe(df_logs_exibir)
        col_comp, col_comp_help = st.columns([0.92, 0.08])

        with col_comp:
            st.markdown("**Similaridade**")

        with col_comp_help:
            with st.popover("❔"):
                st.write("""
        Representa o grau de proximidade semântica entre a pergunta e o chunk recuperado.

Quanto maior o valor, mais parecido semanticamente o modelo considerou o texto.
        """)

        col_score, col_score_help = st.columns([0.92, 0.08])

        with col_score:
            st.markdown("**Score Final Retrieval**")

        with col_score_help:
            with st.popover("❔"):
                st.write("""
        Pontuação final utilizada para ordenar os resultados recuperados.

        No experimento atual, ela corresponde à similaridade semântica calculada pelos embeddings.
        """)
        if st.button("📊 Gerar gráfico dos Resultados Recuperados pelo Sistema"):
            sim_rank = (
                df_logs.groupby("rank")["similaridade"]
                .mean()
                .reset_index()
            )

            campo_logs = (
                df_logs["campo"]
                .value_counts()
                .reset_index()
            )

            campo_logs.columns = [
                "campo",
                "quantidade"
            ]

            # ======================================
            # QUALIDADE DOS RESULTADOS POR POSIÇÃO
            # ======================================

            sim_rank = (
                df_logs.groupby("rank")["similaridade"]
                .mean()
                .reset_index()
            )

            sim_rank["rank"] = sim_rank["rank"].astype(int).astype(str)

            chart_sim = alt.Chart(sim_rank).mark_line(point=True).encode(
                x=alt.X(
                    "rank:N",
                    sort=[
                        "1", "2", "3", "4", "5",
                        "6", "7", "8", "9", "10"
                    ],
                    axis=alt.Axis(
                        title="Posição do resultado",
                        labelAngle=0
                    )
                ),
                y=alt.Y(
                    "similaridade:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(
                        title="Similaridade média"
                    )
                ),
                tooltip=[
                    alt.Tooltip("rank:N", title="Posição"),
                    alt.Tooltip("similaridade:Q", title="Similaridade média")
                ]
            ).properties(
                title="Qualidade dos resultados por posição",
                height=380
            )

            st.altair_chart(chart_sim, use_container_width=True)
            st.info("""
            📖 Interpretação

            Este gráfico mostra a qualidade média dos resultados encontrados pelo sistema.

            🔹 Similaridade média: indica o quanto os resultados encontrados se relacionam com a pergunta realizada. Valores mais próximos de 1 representam resultados mais relevantes.

            O ideal é que os primeiros resultados apresentem valores maiores e que a linha diminua gradualmente ao longo das posições. Isso indica que os resultados mais relevantes estão aparecendo primeiro para o usuário.
            """)

            primeiro = sim_rank["similaridade"].iloc[0]
            ultimo = sim_rank["similaridade"].iloc[-1]

            if primeiro > ultimo:
                st.success(
                    "✅ Os resultados mais relevantes estão aparecendo nas primeiras posições do ranking."
                )
            else:
                st.warning(
                    "⚠️ O sistema não está diferenciando bem a relevância dos resultados."
                )

            # ======================================
            # QUANTIDADE DE TRECHOS POR CAMPO
            # ======================================

            grafico_barra_altair(
                campo_logs,
                "campo",
                "quantidade",
                "Quantidade de chunks recuperados por campo",
                "Quantidade de chunks",
                escala_0_1=False
            )

        st.subheader("Chunks gerados")
        st.dataframe(df_chunks)

        nome_arquivo = f"resultado_{resumo.loc[0, 'estrategia']}_{resumo.loc[0, 'chunk_size']}_{resumo.loc[0, 'overlap']}.xlsx"

        with pd.ExcelWriter(nome_arquivo) as writer:
            resumo.to_excel(
                writer,
                sheet_name="Desempenho_Geral",
                index=False
            )

            resumo_tipo.to_excel(
                writer,
                sheet_name="Desempenho_Categoria",
                index=False
            )

            df_resultados.to_excel(
                writer,
                sheet_name="Resultados_Perguntas",
                index=False
            )

            df_logs.to_excel(
                writer,
                sheet_name="Resultados_Recuperados",
                index=False
            )

            df_chunks.to_excel(
                writer,
                sheet_name="Trechos_Gerados",
                index=False
            )

        with open(nome_arquivo, "rb") as file:
            st.download_button(
                "Baixar Excel do experimento",
                file,
                file_name=nome_arquivo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# =========================
# MODO COMPARAÇÃO
# =========================

if modo_uso == "Comparar estratégias de chunking":

    st.subheader("⚖️ Comparar Estratégias de Chunking")

    st.info("""
Neste modo, você pode comparar duas ou três estratégias de chunking usando a mesma planilha de perguntas.

O sistema executa cada técnica separadamente e gera uma tabela comparativa com os principais resultados quantitativos.
""")

    with st.expander("ℹ️ Como deve ser a planilha de perguntas?"):

        st.info("""
A planilha deve conter obrigatoriamente:

- id
- pergunta
- tipo_pergunta
- componente_esperado
- campo_esperado

Formatos aceitos:
- CSV (.csv)
- XLSX (.xlsx)
- XLS (.xls)
""")

        exemplo = pd.DataFrame({
            "id": [1, 2],
            "pergunta": [
                "Qual é a ementa de Algoritmos?",
                "Quais são os objetivos de Inteligência Artificial?"
            ],
            "tipo_pergunta": [
                "ementa",
                "objetivos"
            ],
            "componente_esperado": [
                "Algoritmos",
                "Inteligência Artificial"
            ],
            "campo_esperado": [
                "Ementa",
                "Objetivos"
            ]
        })

        st.dataframe(exemplo)

    arquivo_perguntas_comp = st.file_uploader(
        "Envie a planilha de perguntas para comparação",
        type=["csv", "xlsx", "xls"],
        key="upload_comparacao"
    )

    df_perguntas_comp = None

    if arquivo_perguntas_comp is not None:

        extensao = arquivo_perguntas_comp.name.split(".")[-1].lower()

        if extensao == "csv":
            df_perguntas_comp = pd.read_csv(arquivo_perguntas_comp)

        elif extensao in ["xlsx", "xls"]:
            df_perguntas_comp = pd.read_excel(arquivo_perguntas_comp)

        else:
            st.error("Formato não suportado.")
            df_perguntas_comp = None

        if df_perguntas_comp is not None:
            df_perguntas_comp = validar_perguntas(df_perguntas_comp)

    estrategias_comparar = st.multiselect(
        "Selecione as estratégias que deseja comparar",
        ["recursivo", "fixo", "semântico"],
        default=["recursivo", "fixo", "semântico"]
    )

    st.markdown("### Parâmetros por estratégia")

    col1, col2, col3 = st.columns(3)

    parametros = {}

    with col1:
        st.markdown("**Recursivo**")
        parametros["recursivo"] = {
            "chunk_size": st.number_input(
                "Tamanho do chunk",
                min_value=100,
                max_value=2000,
                value=700,
                step=50,
                key="chunk_recursivo"
            ),
            "overlap": st.number_input(
                "Sobreposição (Overlap)",
                min_value=0,
                max_value=500,
                value=120,
                step=10,
                key="overlap_recursivo"
            )
        }

    with col2:
        st.markdown("**Fixo**")
        parametros["fixo"] = {
            "chunk_size": st.number_input(
                "Tamanho do chunk",
                min_value=100,
                max_value=2000,
                value=300,
                step=50,
                key="chunk_fixo"
            ),
            "overlap": st.number_input(
                "Sobreposição (Overlap)",
                min_value=0,
                max_value=500,
                value=0,
                step=10,
                key="overlap_fixo"
            )
        }

    with col3:
        st.markdown("**Semântico**")
        parametros["semântico"] = {
            "chunk_size": st.number_input(
                "Tamanho do chunk",
                min_value=100,
                max_value=2000,
                value=500,
                step=50,
                key="chunk_semantico"
            ),
            "overlap": st.number_input(
                "Sobreposição (Overlap)",
                min_value=0,
                max_value=500,
                value=50,
                step=10,
                key="overlap_semantico"
            )
        }

    if df_perguntas_comp is not None:
        st.metric("Perguntas carregadas", len(df_perguntas_comp))

    if (
            df_perguntas_comp is not None
            and len(estrategias_comparar) >= 2
            and st.button("⚖️ Rodar comparação")
    ):

        resumos = []
        resultados_por_estrategia = {}
        logs_por_estrategia = {}
        chunks_por_estrategia = {}

        for est in estrategias_comparar:
            with st.spinner(f"Rodando estratégia: {est}..."):
                resumo_est, resultados_est, logs_est, chunks_est = rodar_experimento_comparacao(
                    df_perguntas_comp,
                    df_disciplinas,
                    modelo,
                    est,
                    parametros[est]["chunk_size"],
                    parametros[est]["overlap"]
                )

                resumos.append(resumo_est)
                resultados_por_estrategia[est] = resultados_est
                logs_por_estrategia[est] = logs_est
                chunks_por_estrategia[est] = chunks_est

        df_comparacao = pd.DataFrame(resumos)

        st.session_state.df_comparacao = df_comparacao
        st.session_state.resultados_por_estrategia = resultados_por_estrategia
        st.session_state.logs_por_estrategia = logs_por_estrategia
        st.session_state.chunks_por_estrategia = chunks_por_estrategia

        st.success("Comparação finalizada!")

    if "df_comparacao" in st.session_state:

        df_comparacao = st.session_state.df_comparacao

        col_comp_titulo, col_comp_help = st.columns([0.92, 0.08])

        with col_comp_titulo:
            st.subheader("📊 Comparação Quantitativa das Estratégias")

        with col_comp_help:
            with st.popover("❔"):
                st.write("""
        Esta tabela compara os resultados quantitativos das estratégias de chunking selecionadas.

        Os valores de acerto são apresentados em porcentagens.

        Campos da tabela:

        - Total de trechos: quantidade de chunks gerados pela estratégia.
        - Acertos em 1º lugar: porcentagem de perguntas em que a resposta correta apareceu na primeira posição.
        - Acertos entre 2º e 5º: porcentagem de perguntas em que a resposta correta apareceu entre a segunda e a quinta posição.
        - Acertos entre 6º e 10º: porcentagem de perguntas em que a resposta correta apareceu entre a sexta e a décima posição.
        - Qualidade média da posição: corresponde ao MRR. Quanto mais próximo de 1, melhor, pois indica que as respostas corretas apareceram nas primeiras posições.
        - Tempo médio de busca: tempo médio necessário para recuperar os resultados, em milissegundos.
        """)

        df_comparacao_exibir = df_comparacao.rename(columns={
            "estrategia": "Estratégia",
            "chunk_size": "Tamanho do chunk",
            "overlap": "Sobreposição",
            "total_chunks": "Total de chunk",
            "acertos_1_lugar": "Acertos em 1º lugar (%)",
            "acertos_2_5_lugar": "Acertos entre 2º e 5º (%)",
            "acertos_6_10_lugar": "Acertos entre 6º e 10º (%)",
            "qualidade_media_posicao": "Qualidade média da posição",
            "tempo_medio_busca_ms": "Tempo médio de busca (ms)"
        })

        for col in [
            "Acertos em 1º lugar (%)",
            "Acertos entre 2º e 5º (%)",
            "Acertos entre 6º e 10º (%)"
        ]:
            df_comparacao_exibir[col] = (
                                                df_comparacao_exibir[col] * 100
                                        ).round(1).astype(str) + "%"

        df_comparacao_exibir["Qualidade média da posição"] = (
            df_comparacao_exibir["Qualidade média da posição"].round(4)
        )

        df_comparacao_exibir["Tempo médio de busca (ms)"] = (
            df_comparacao_exibir["Tempo médio de busca (ms)"].round(2)
        )

        st.dataframe(df_comparacao_exibir)

        st.subheader("Resultado da comparação")

        melhor_mrr = df_comparacao.loc[
            df_comparacao["qualidade_media_posicao"].idxmax()
        ]

        melhor_tempo = df_comparacao.loc[
            df_comparacao["tempo_medio_busca_ms"].idxmin()
        ]

        maior_recall = df_comparacao["acertos_1_lugar"].max()

        melhores_recall = df_comparacao[
            df_comparacao["acertos_1_lugar"] == maior_recall
            ]

        nomes_melhores_recall = ", ".join(
            melhores_recall["estrategia"].str.capitalize()
        )

        ranking_mrr = df_comparacao.sort_values(
            "qualidade_media_posicao",
            ascending=False
        )

        ranking_tempo = df_comparacao.sort_values(
            "tempo_medio_busca_ms",
            ascending=True
        )

        ranking_recall = df_comparacao.sort_values(
            "acertos_1_lugar",
            ascending=False
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.success(
                f"""
        🏆 Melhor qualidade

        **{melhor_mrr['estrategia'].capitalize()}**

        MRR: {melhor_mrr['qualidade_media_posicao']:.4f}
        """
            )

            st.caption(
                "Outras estratégias: " +
                " | ".join(
                    [
                        f"{row['estrategia'].capitalize()}: {row['qualidade_media_posicao']:.4f}"
                        for _, row in ranking_mrr.iterrows()
                        if row["estrategia"] != melhor_mrr["estrategia"]
                    ]
                )
            )

        with col2:
            st.info(
                f"""
        ⚡ Mais rápida

        **{melhor_tempo['estrategia'].capitalize()}**

        {melhor_tempo['tempo_medio_busca_ms']:.2f} ms
        """
            )

            st.caption(
                "Outras estratégias: " +
                " | ".join(
                    [
                        f"{row['estrategia'].capitalize()}: {row['tempo_medio_busca_ms']:.2f} ms"
                        for _, row in ranking_tempo.iterrows()
                        if row["estrategia"] != melhor_tempo["estrategia"]
                    ]
                )
            )

        with col3:

            maior_recall = df_comparacao["acertos_1_lugar"].max()

            melhores_recall = df_comparacao[
                df_comparacao["acertos_1_lugar"] == maior_recall
                ]

            nomes_melhores_recall = ", ".join(
                melhores_recall["estrategia"].str.capitalize()
            )

            outras_recall = df_comparacao[
                df_comparacao["acertos_1_lugar"] != maior_recall
                ].sort_values(
                "acertos_1_lugar",
                ascending=False
            )

            st.warning(
                f"""
        🎯 Mais acertos em 1º lugar

        **{nomes_melhores_recall}**

        {maior_recall * 100:.2f}%
        """
            )

            if not outras_recall.empty:
                st.caption(
                    "Outras estratégias: " +
                    " | ".join(
                        [
                            f"{row['estrategia'].capitalize()}: {row['acertos_1_lugar'] * 100:.2f}%"
                            for _, row in outras_recall.iterrows()
                        ]
                    )
                )
            else:
                st.caption(
                    "Todas as estratégias apresentaram o mesmo resultado nesta categoria."
                )
        if st.button("📊 Gerar gráfico comparativo"):
            df_grafico = df_comparacao.copy()

            grafico_barras_multiplas_altair(
                df_grafico,
                "estrategia",
                [
                    "acertos_1_lugar",
                    "acertos_2_5_lugar",
                    "acertos_6_10_lugar",
                    "qualidade_media_posicao"
                ],
                "Comparação de desempenho entre estratégias",
                "Valor da métrica"
            )

            grafico_barra_altair(
                df_grafico,
                "estrategia",
                "tempo_medio_busca_ms",
                "Tempo médio de busca por estratégia",
                "Tempo médio de busca (ms)",
                escala_0_1=False
            )

            grafico_barra_altair(
                df_grafico,
                "estrategia",
                "total_chunks",
                "Quantidade de chunks gerados por estratégia",
                "Total de chunks",
                escala_0_1=False
            )

        st.subheader("📁 Exportar comparação")

        nome_arquivo_comp = f"comparacao_estrategias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        with pd.ExcelWriter(nome_arquivo_comp) as writer:

            df_comparacao_exibir.to_excel(
                writer,
                sheet_name="Comparacao_Geral",
                index=False
            )

            for est, df_res in st.session_state.resultados_por_estrategia.items():
                df_res.to_excel(
                    writer,
                    sheet_name=f"Resultados_{est}"[:31],
                    index=False
                )

            for est, df_logs in st.session_state.logs_por_estrategia.items():
                df_logs.to_excel(
                    writer,
                    sheet_name=f"Recuperados_{est}"[:31],
                    index=False
                )

        with open(nome_arquivo_comp, "rb") as file:
            st.download_button(
                "Baixar Excel da comparação",
                file,
                file_name=nome_arquivo_comp,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    elif df_perguntas_comp is not None and len(estrategias_comparar) < 2:
        st.warning("Selecione pelo menos duas estratégias para comparar.")

# =========================
# MODO CHATBOT
# =========================

if modo_uso == "Chatbot interativo":

    from firestore import (
        listar_conversas,
        criar_conversa,
        listar_mensagens,
        atualizar_titulo_conversa,
        salvar_mensagem,
        excluir_conversa,
        verificar_rate_limit
    )

    from llm import gerar_resposta, reescrever_pergunta
    from firebase_config import db

    user = st.session_state["user"]

    st.markdown(
        """
        <div style="margin-left: 0; padding-top: 4px;">
            <div style="font-size: 28px; font-weight: 650; line-height: 1.2;">
                Chatbot Acadêmico <span style="font-size: 15px; opacity: .65;">⌄</span>
            </div>
            <div style="font-size: 12px; opacity: .55; margin-top: 8px;">
                Tire suas dúvidas sobre o Projeto Pedagógico do Curso Técnico em Informática.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # =========================
    # NOVA CONVERSA
    # =========================

    if st.sidebar.button(
            "➕ Nova conversa",
            type="primary",
            use_container_width=True,
            key="nova_conversa"
    ):
        conversation_id = criar_conversa(user["uid"])
        st.session_state["conversation_id"] = conversation_id
        st.session_state.pop("select_conversas", None)
        st.rerun()

    # =========================
    # LISTA DE CONVERSAS
    # =========================

    conversas = listar_conversas(user["uid"])

    if not conversas:
        conversation_id = criar_conversa(user["uid"])
        st.session_state["conversation_id"] = conversation_id
        st.rerun()

    conversas = listar_conversas(user["uid"])

    if "conversation_id" not in st.session_state or not any(
            c["id"] == st.session_state["conversation_id"] for c in conversas
    ):
        st.session_state["conversation_id"] = conversas[0]["id"]

    st.sidebar.markdown("### Conversas")

    # Cada conversa ocupa uma linha; o menu aparece somente no hover.
    for conversa_item in conversas:

        conversa_id = conversa_item["id"]
        titulo = conversa_item["title"]

        titulo_exibido = titulo
        if len(titulo_exibido) > 32:
            titulo_exibido = titulo_exibido[:29] + "..."

        with st.sidebar.container(key=f"convrow_{conversa_id}"):
            col_conversa, col_menu = st.columns([0.88, 0.12], gap="small")

            with col_conversa:
                if st.button(
                        titulo_exibido,
                        key=f"conversa_{conversa_id}",
                        use_container_width=True
                ):
                    st.session_state["conversation_id"] = conversa_id
                    st.rerun()

            with col_menu:
                with st.popover("⋯", use_container_width=True):
                    if st.download_button(
                            "⬇️ Exportar",
                            data=gerar_exportacao_conversa(
                                titulo,
                                listar_mensagens(user["uid"], conversa_id)
                            ),
                            file_name=nome_arquivo_conversa(titulo),
                            mime="text/markdown",
                            use_container_width=True,
                            key=f"exportar_{conversa_id}"
                    ):
                        pass

                    st.divider()

                    if st.button(
                            "🗑️ Excluir",
                            key=f"excluir_{conversa_id}",
                            use_container_width=True
                    ):
                        excluir_conversa(user["uid"], conversa_id)
                        if st.session_state.get("conversation_id") == conversa_id:
                            st.session_state.pop("conversation_id", None)
                            st.session_state.pop("select_conversas", None)
                        st.rerun()

    # =========================
    # CONVERSA ATUAL
    # =========================

    conversa_selecionada = next(
        c for c in conversas
        if c["id"] == st.session_state["conversation_id"]
    )

    titulo_conversa = conversa_selecionada["title"]

    mensagens = listar_mensagens(
        user["uid"],
        st.session_state["conversation_id"]
    )

    # =========================
    # HISTÓRICO
    # =========================

    for mensagem in mensagens:
        with st.chat_message(mensagem["role"]):
            st.write(mensagem["content"])
            if mensagem["role"] == "assistant":
                botao_copiar_resposta(mensagem["content"])

    # =========================
    # CAMPO DA PERGUNTA
    # =========================

    pergunta_usuario = st.chat_input("Digite sua pergunta...")

    if pergunta_usuario and pergunta_usuario.strip():

        permitido, segundos_restantes = verificar_rate_limit(
            user["uid"]
        )

        if not permitido:
            st.warning(
                f"⏳ Você enviou muitas perguntas em pouco tempo. "
                f"Aguarde aproximadamente {segundos_restantes} segundos "
                f"antes de tentar novamente."
            )
            st.stop()

        historico = ""
        for mensagem in mensagens:
            role = mensagem["role"]
            content = mensagem["content"]
            if role == "user":
                historico += f"Usuário: {content}\n"
            elif role == "assistant":
                historico += f"Assistente: {content}\n"

        with st.chat_message("user"):
            st.write(pergunta_usuario)

        conversa_doc = (
            db.collection("users")
            .document(user["uid"])
            .collection("conversations")
            .document(st.session_state["conversation_id"])
            .get()
        )

        dados = conversa_doc.to_dict()

        if dados["title"] == "Nova conversa":
            titulo = pergunta_usuario.strip()
            if len(titulo) > 60:
                titulo = titulo[:57] + "..."
            atualizar_titulo_conversa(
                user["uid"],
                st.session_state["conversation_id"],
                titulo
            )

        salvar_mensagem(
            user["uid"],
            st.session_state["conversation_id"],
            "user",
            pergunta_usuario
        )

        with st.spinner("Preparando base de conhecimento..."):
            df_chunks_chat, embeddings_chunks_chat = preparar_base_chat(
                estrategia,
                chunk_size,
                overlap
            )

        with st.spinner("Interpretando contexto da conversa..."):
            if precisa_reescrever(pergunta_usuario, historico):
                pergunta_busca = reescrever_pergunta(
                    pergunta_usuario,
                    historico
                )
            else:
                pergunta_busca = pergunta_usuario

        with st.spinner("Buscando resposta..."):
            resultados_chat = buscar_chat(
                pergunta_busca,
                df_chunks_chat,
                embeddings_chunks_chat,
                modelo,
                top_k=5
            )

        contexto = ""
        for _, r in resultados_chat.iterrows():
            contexto += f"""
Disciplina: {r["Componente"]}
Campo: {r["campo"]}

{r["texto_chunkado"]}

-----------------------------
"""

        with st.spinner("Gerando resposta com Gemini..."):
            resposta = gerar_resposta(
                contexto,
                pergunta_usuario,
                historico
            )

        salvar_mensagem(
            user["uid"],
            st.session_state["conversation_id"],
            "assistant",
            resposta
        )

        st.rerun()

if st.sidebar.button("Limpar cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.rerun()