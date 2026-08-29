import pdfplumber
import re
import pandas as pd

import re

def limpar(texto):
    if not texto:
        return ""

    texto = str(texto)
    texto = texto.replace("￾", "")
    texto = texto.replace("\xa0", " ")

    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()


def limpar_ruido_pdf(texto):
    if not texto:
        return ""

    texto = str(texto)

    texto = texto.replace("￾", "")
    texto = texto.replace("\xa0", " ")

    # remove cabeçalho completo mesmo se estiver com quebras de linha
    texto = re.sub(
        r"Projeto\s+Pedagógico\s+do\s+Curso\s+Técnico\s+em\s+Informática\s+Integrado\s+ao\s+Ensino\s+Médio",
        "",
        texto,
        flags=re.I
    )

    texto = re.sub(
        r"TÉCNICO\s+EM\s+INFORMÁTICA\s+INTEGRADO\s+AO\s+ENSINO\s+MÉDIO",
        "",
        texto,
        flags=re.I
    )

    texto = re.sub(
        r"\bCUBATÃO\b",
        "",
        texto,
        flags=re.I
    )

    # remove versões parcialmente quebradas
    texto = re.sub(
        r"Projeto\s+Pedagógico.*?Ensino\s+Médio",
        "",
        texto,
        flags=re.I | re.S
    )

    texto = re.sub(
        r"Curso\s+Técnico\s+em\s+Informática.*?Ensino\s+Médio",
        "",
        texto,
        flags=re.I | re.S
    )

    # remove números de página isolados
    texto = re.sub(
        r"(?m)^\s*\d{1,3}\s*$",
        "",
        texto
    )

    # remove linhas que sobraram
    linhas_limpas = []

    for linha in texto.splitlines():
        linha_limpa = linha.strip()

        if not linha_limpa:
            continue

        if re.fullmatch(r"\d{1,3}", linha_limpa):
            continue

        if "Projeto Pedagógico" in linha_limpa:
            continue

        if "Curso Técnico em Informática" in linha_limpa:
            continue

        if "Integrado ao Ensino Médio" in linha_limpa:
            continue

        if linha_limpa.upper() in [
            "CUBATÃO",
            "TÉCNICO EM INFORMÁTICA INTEGRADO AO ENSINO MÉDIO",
            "CAMPUS",
            "CÂMPUS"
        ]:
            continue

        linhas_limpas.append(linha_limpa)

    texto = "\n".join(linhas_limpas)

    # corrige hifenização
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)

    # normaliza espaços
    texto = re.sub(r"[ \t]+", " ", texto)

    # normaliza quebras
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()
def limpar_lixo_pdf(texto):
    """
    Limpeza aplicada em cada campo extraído.
    """
    return limpar_ruido_pdf(texto)

    return texto.strip()
def extrair_campo(bloco, campo):
    padrao = rf"{campo}:\s*(.*?)\n"
    match = re.search(padrao, bloco)
    return limpar_lixo_pdf(match.group(1)) if match else ""

def extrair_bloco(bloco, inicio, fim):
    padrao = rf"{inicio}(.*?){fim}"

    match = re.search(
        padrao,
        bloco,
        re.S | re.I
    )

    return limpar_lixo_pdf(match.group(1)) if match else ""
# =========================
# LER PDF
# =========================
with pdfplumber.open(r"C:\Users\sophi\Downloads\2022_PPC_Técnico_em_Informática_reformulado_ingressantes_a_partir_de_2023 (1).pdf") as pdf:
    texto = "\n".join([p.extract_text() or "" for p in pdf.pages])

# limpeza pesada no texto inteiro
texto = limpar_ruido_pdf(texto)

# normalizações específicas
texto = texto.replace("3--", "3-")
texto = texto.replace("5 –", "5-")
texto = texto.replace("4 –", "4-")
texto = texto.replace("8 –", "8-")

# =========================
# NORMALIZA TEXTO
# =========================
texto = texto.replace("3--", "3-")
texto = texto.replace("5 –", "5-")
texto = texto.replace("4 –", "4-")

# =========================
# SEPARAR DISCIPLINAS
# =========================
blocos = re.split(r"\n\s*1- IDENTIFICAÇÃO", texto)

dados = []

for bloco in blocos[1:]:
    info = {}

    # =========================
    # CAMPOS BÁSICOS
    # =========================
    info["Curso"] = extrair_campo(bloco, "Curso")
    info["Componente"] = extrair_campo(bloco, "Componente curricular")
    info["Tipo"] = extrair_campo(bloco, "Tipo")
    info["Núcleo"] = extrair_campo(bloco, "Núcleo")



    # =========================
    # LINHA ANO / SIGLA / AULAS
    # =========================

    # =========================
    # ANO / SIGLA / AULAS (ULTRA ROBUSTO)
    # =========================

    def extrair_ano(bloco):

        # disciplinas sem ano
        if "Libras Básico" in bloco or "Libras Intermediário" in bloco:
            return ""

        inicio = bloco[:1500]

        partes = re.split(
            r"3\s*-\s*EMENTA",
            inicio,
            flags=re.I
        )

        inicio = partes[0]

        inicio = re.sub(r"\s+", " ", inicio)

        match = re.search(
            r"\b([1-4])[º°o]\b",
            inicio
        )

        if match:
            return f"{match.group(1)}º"

        match = re.search(
            r"Ano[:\s]+([1-4])\b",
            inicio,
            re.I
        )

        if match:
            return f"{match.group(1)}º"

        sigla_match = re.search(
            r"Sigla:\s*([A-Z0-9]+)",
            inicio,
            re.I
        )

        if sigla_match:

            codigo = sigla_match.group(1)

            numero_final = re.search(
                r"([1-4])$",
                codigo
            )

            if numero_final:
                return f"{numero_final.group(1)}º"

        return ""

    def extrair_sigla(bloco):

        match = re.search(
            r"Sigla:\s*([A-Z0-9]+)",
            bloco,
            re.I
        )

        if match:
            return match.group(1)

        # fallback:
        siglas = re.findall(r"\bCBT[A-Z0-9]+\b", bloco)

        if siglas:
            return siglas[0]

        return ""


    def extrair_aulas_semanais(bloco):

        match = re.search(
            r"N[ºo°]?\s*de aulas semanais:\s*(\d+)",
            bloco,
            re.I
        )

        if match:
            return match.group(1)

        return ""


    def extrair_docentes(bloco):

        linhas = bloco.split("\n")

        for i, linha in enumerate(linhas):

            linha_limpa = linha.strip().lower()

            # =====================================
            # CASO 1:
            # docentes: 1
            # =====================================
            match = re.search(
                r"docentes:\s*(\d+)",
                linha_limpa
            )

            if match:
                numero = int(match.group(1))

                # evita pegar 30, 57 etc
                if 1 <= numero <= 10:
                    return str(numero)

            # =====================================
            # CASO 2:
            # docentes:
            # 1
            # =====================================
            if "docentes:" in linha_limpa:

                # olha SOMENTE próximas 2 linhas
                for j in range(i + 1, min(i + 3, len(linhas))):

                    valor = linhas[j].strip()

                    if valor == "":
                        continue

                    match_numero = re.search(r"(\d+)", valor)

                    if match_numero:

                        numero = int(match_numero.group(1))

                        # filtro de segurança
                        if 1 <= numero <= 10:
                            return str(numero)

        return "Não informado"
    ano_extraido = extrair_ano(bloco)

    if not ano_extraido:
        info["Ano"] = "Optativa"
    else:
        info["Ano"] = str(ano_extraido).strip()
    info["Sigla"] = extrair_sigla(bloco)
    info["Aulas_semanais"] = extrair_aulas_semanais(bloco)
    info["Docentes"] = extrair_docentes(bloco)
    # Total de aulas + CH
    def extrair_numero_campo(bloco, campo):
        """
        Extrai números mesmo com quebra de linha ou tabela quebrada.
        """

        padroes = [
            rf"{campo}:\s*([\d,]+)",
            rf"{campo}\s*\n\s*([\d,]+)",
            rf"{campo}.*?\n.*?([\d,]+)"
        ]

        for padrao in padroes:
            match = re.search(padrao, bloco, re.S)

            if match:
                valor = limpar(match.group(1))

                # evita pegar lixo
                if re.match(r"^[\d,]+$", valor):
                    return valor

        return ""


    info["Total_aulas"] = extrair_numero_campo(
        bloco,
        "Total de aulas"
    )

    info["CH_presencial"] = extrair_numero_campo(
        bloco,
        r"C\.H\. Presencial"
    )

    info["CH_distancia"] = extrair_numero_campo(
        bloco,
        r"C\.H\.\s*Dist[âa]ncia"
    ) or "0"

    info["Total_horas"] = extrair_numero_campo(
        bloco,
        "Total de horas"
    ) or "Não se aplica"


    def extrair_ch_laboratorio(bloco):

        linhas = bloco.split("\n")

        for i, linha in enumerate(linhas):

            linha_limpa = linha.strip().lower()

            # =====================================
            # CASO NORMAL
            # =====================================
            encontrou_campo = (
                    "carga horária prevista em laboratório" in linha_limpa
            )

            # =====================================
            # CASO QUEBRADO
            # "Carga horária prevista em"
            # "laboratório:"
            # =====================================
            if not encontrou_campo:

                if (
                        "carga horária prevista em" in linha_limpa
                        and i + 1 < len(linhas)
                        and "laboratório" in linhas[i + 1].lower()
                ):
                    encontrou_campo = True

            # =====================================
            # EXTRAI VALOR
            # =====================================
            if encontrou_campo:

                for j in range(i + 1, min(i + 6, len(linhas))):

                    valor = linhas[j].strip()

                    # ignora vazio
                    if valor == "":
                        continue

                    # ignora docentes
                    if "docente" in valor.lower():
                        continue

                    # NÃO SE APLICA -> 0
                    if "não se aplica" in valor.lower():
                        return "0"

                    # pega número do texto
                    match = re.search(r"(\d+)", valor)

                    if match:
                        return match.group(1)

        return "0"




    info["CH_laboratorio"] = extrair_ch_laboratorio(bloco)

    info["Aulas_semanais"] = extrair_aulas_semanais(bloco)
    # =========================
    # BLOCOS GRANDES CORRIGIDOS
    # =========================

    info["Conhecimentos"] = extrair_bloco(
        bloco,
        r"2\s*[-–]\s*CONHECIMENTOS ESSENCIAIS DO CURRÍCULO DE REFERÊNCIA",
        r"\n\s*3\s*[-–]"
    )

    info["Ementa"] = extrair_bloco(
        bloco,
        r"3\s*[-–]?\s*EMENTA:?",
        r"\n\s*4\s*[-–]?\s*OBJETIVOS"
    )

    info["Objetivos"] = extrair_bloco(
        bloco,
        r"4\s*[-–]?\s*OBJETIVOS:?",
        r"\n\s*5\s*[-–]?\s*(ÁREAS DE INTEGRAÇÃO|CONTE[ÚU]DO)"
    )

    info["Areas_integracao"] = extrair_bloco(
        bloco,
        r"5\s*[-–]?\s*ÁREAS DE INTEGRAÇÃO:?",
        r"\n\s*6\s*[-–]?\s*CONTE[ÚU]DO"
    ) or "Não se aplica"

    info["Conteudo"] = extrair_bloco(
        bloco,
        r"\n\s*(5|6)\s*[-–]?\s*CONTE[ÚU]DO\s*PROGRAM[ÁA]TICO:?",
        r"\n\s*(6|7)\s*[-–]?\s*BIBLIOGRAFIA B[ÁA]SICA"
    )

    info["Bibliografia_Basica"] = extrair_bloco(
        bloco,
        r"\n\s*(6|7)\s*[-–]?\s*BIBLIOGRAFIA B[ÁA]SICA:?",
        r"\n\s*(7|8)\s*[-–]?\s*BIBLIOGRAFIA COMPLEMENTAR"
    )

    info["Bibliografia_Complementar"] = extrair_bloco(
        bloco,
        r"\n\s*(7|8)\s*[-–]?\s*BIBLIOGRAFIA COMPLEMENTAR:?",
        r"(?=\n\s*(CAMPUS|CÂMPUS|1\s*[-–]\s*IDENTIFICAÇÃO)|$)"
    )

    info["CH_distancia"] = extrair_numero_campo(
        bloco,
        r"C\.H\.\s*Dist[âa]ncia"
    ) or "Não se aplica"

    if info["Componente"] in [
        "Algoritmos",
        "Sistemas Computacionais 1",
        "Linguagem de Programação 1"
    ]:
        print("\n====================")
        print(info["Componente"])
        print("====================")
        print(bloco[:5000])

    dados.append(info)


# =========================
# GERAR EXCEL
# =========================
df = pd.DataFrame(dados)

# =========================
# VALIDAÇÃO 100%
# =========================
print("\n========== VALIDAÇÃO GERAL ==========")

campos_essenciais = [
    "Curso",
    "Componente",
    "Ano",
    "Sigla",
    "Total_aulas",
    "CH_presencial",
    "CH_distancia",
    "Total_horas",
    "Docentes",
    "CH_laboratorio",
    "Conhecimentos",
    "Ementa",
    "Objetivos",
    "Areas_integracao",
    "Conteudo",
    "Bibliografia_Basica",
    "Bibliografia_Complementar"
]


for campo in campos_essenciais:
    if campo not in df.columns:
        print(f"❌ Campo ausente no DataFrame: {campo}")
        continue

    vazios = df[
        df[campo].astype(str).str.strip().isin(["", "nan", "None"])
    ]

    print(f"{campo}: {len(vazios)} vazios")

    if len(vazios) > 0:
        print(vazios[["Componente", "Ano", "Sigla"]].head(10))

# =========================
# VALIDAÇÃO
# =========================

print("\n========== VALIDAÇÃO ==========")

campos_obrigatorios = [
    "Curso",
    "Componente",
    "Ano",
    "Sigla",
    "Total_aulas",
    "CH_presencial",
    "Docentes",
    "Ementa",
    "Objetivos",
    "Conteudo"
]


