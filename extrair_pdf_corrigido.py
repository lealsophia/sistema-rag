from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pdfplumber


EXPECTED_COMPONENTS = 56

COLUNAS = [
    "Curso",
    "Componente",
    "Tipo",
    "Núcleo",
    "Ano",
    "Sigla",
    "Aulas_semanais",
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
    "Bibliografia_Complementar",
]


# -----------------------------------------------------------------------------
# Limpeza
# -----------------------------------------------------------------------------

def _normalizar_unicode(texto: str) -> str:
    if not texto:
        return ""
    return (
        str(texto)
        .replace("\xa0", " ")
        .replace("￾", "")
        .replace("\u00ad", "")
    )


def _eh_ruido_exato(linha: str) -> bool:
    """Remove somente linhas que são cabeçalho/rodapé, sem apagar conteúdo válido."""
    s = re.sub(r"\s+", " ", linha).strip()
    if not s:
        return True

    upper = s.upper()
    if upper in {
        "CUBATÃO",
        "CAMPUS",
        "CÂMPUS",
        "CBT",
        "TÉCNICO EM INFORMÁTICA INTEGRADO AO ENSINO MÉDIO",
        "PROJETO PEDAGÓGICO DO CURSO TÉCNICO EM INFORMÁTICA INTEGRADO AO ENSINO MÉDIO",
    }:
        return True

    # Alguns PDFs trazem setas/logotipo antes do título.
    if "TÉCNICO EM INFORMÁTICA INTEGRADO AO ENSINO MÉDIO" in upper and re.match(r"^[>»›]+", s):
        return True

    return False


def limpar_texto_pdf(texto: str) -> str:
    """Limpa ruído de página preservando valores reais dos campos."""
    texto = _normalizar_unicode(texto)
    linhas: List[str] = []

    for linha in texto.splitlines():
        linha = linha.rstrip()
        if _eh_ruido_exato(linha):
            continue
        linhas.append(linha)

    texto = "\n".join(linhas)

    # Une palavras realmente hifenizadas no fim da linha. Não mexe em hífens internos.
    texto = re.sub(r"([A-Za-zÀ-ÿ])-\n([A-Za-zÀ-ÿ])", r"\1\2", texto)

    # Normalizações de espaçamento, sem destruir quebras de linha úteis.
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def limpar_campo(texto: str) -> str:
    texto = _normalizar_unicode(texto)
    if not texto:
        return ""

    linhas = []
    for linha in texto.splitlines():
        linha = re.sub(r"[ \t]+", " ", linha).strip()
        if linha:
            linhas.append(linha)

    texto = "\n".join(linhas)
    texto = re.sub(r"([A-Za-zÀ-ÿ])-\n([A-Za-zÀ-ÿ])", r"\1\2", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# -----------------------------------------------------------------------------
# Extração estrutural
# -----------------------------------------------------------------------------

def extrair_entre(texto: str, inicio: str, fim: str) -> str:
    """Extrai conteúdo entre dois rótulos. Os padrões devem ser não-capturantes."""
    m = re.search(rf"(?:{inicio})\s*(.*?)(?={fim})", texto, flags=re.I | re.S)
    return limpar_campo(m.group(1)) if m else ""


def _primeiro(padroes: List[str], texto: str, default: str = "") -> str:
    for p in padroes:
        m = re.search(p, texto, flags=re.I | re.S)
        if m:
            return limpar_campo(m.group(1))
    return default


def _numero_decimal(texto: str) -> str:
    m = re.search(r"\d+(?:[.,]\d+)?", texto or "")
    return m.group(0) if m else ""


def extrair_identificacao(bloco: str) -> Dict[str, str]:
    # Somente o cabeçalho do plano de ensino, antes da seção 2.
    cab = re.split(
        r"\n\s*2\s*[-–—]+\s*CONHECIMENTOS\s+ESSENCIAIS\s+DO\s+CURR[ÍI]CULO\s+DE\s+REFER[ÊE]NCIA",
        bloco,
        maxsplit=1,
        flags=re.I,
    )[0]

    info: Dict[str, str] = {}

    info["Curso"] = extrair_entre(
        cab,
        r"Curso\s*:",
        r"\s*Componente\s+curricular\s*:",
    )

    info["Componente"] = extrair_entre(
        cab,
        r"Componente\s+curricular\s*:",
        r"\s*Tipo\s*:",
    )
    info["Componente"] = re.sub(r"\s*\n\s*", " ", info["Componente"]).strip()

    info["Tipo"] = extrair_entre(cab, r"Tipo\s*:", r"\s*N[úu]cleo\s*:")
    info["Núcleo"] = extrair_entre(cab, r"N[úu]cleo\s*:", r"\s*Ano\s*:")

    ano = _primeiro([
        r"\bAno\s*:\s*([1-4])\s*[º°oª]?\b",
    ], cab)
    info["Ano"] = f"{ano}º" if ano else "Optativa"

    info["Sigla"] = _primeiro([
        r"\bSigla\s*:\s*([A-Z0-9]+)\b",
        r"\b(CBT[A-Z0-9]+)\b",
    ], cab)

    info["Aulas_semanais"] = _primeiro([
        r"N\s*[º°o]?\s*de\s+aulas\s+semanais\s*:\s*(\d+(?:[.,]\d+)?)",
    ], cab)

    # Carga presencial é normalmente bem comportada.
    info["CH_presencial"] = _primeiro([
        r"C\.\s*H\.\s*Presencial\s*:\s*(\d+(?:[.,]\d+)?)",
    ], cab)

    # Distância e total de horas não aparecem em todos os componentes.
    info["CH_distancia"] = _primeiro([
        r"C\.\s*H\.\s*Dist[âa]ncia\s*:\s*(\d+(?:[.,]\d+)?)",
    ], cab, default="0")

    info["Total_horas"] = _primeiro([
        r"Total\s+de\s+horas\s*:\s*(\d+(?:[.,]\d+)?)",
    ], cab, default="Não se aplica")

    # Total de aulas: primeiro tenta o caso normal.
    total = _primeiro([
        r"Total\s+de\s+aulas\s*:\s*(\d+(?:[.,]\d+)?)",
    ], cab)

    # Em algumas tabelas pdfplumber extrai assim:
    # Total de aulas: C.H. Presencial: 57\n76 C.H. Distância: 0
    if not total:
        total = _primeiro([
            r"Total\s+de\s+aulas\s*:\s*C\.\s*H\.\s*Presencial\s*:\s*\d+(?:[.,]\d+)?\s*\n\s*(\d+(?:[.,]\d+)?)\s+C\.\s*H\.\s*Dist[âa]ncia",
        ], cab)
    info["Total_aulas"] = total

    # Docentes + laboratório: a tabela é extraída em ordens diferentes.
    trecho_dl = ""
    m_dl = re.search(
        r"Quantidade\s+de.*?Carga\s+hor[áa]ria\s+prevista\s+em\s+laborat[óo]rio\s*:(.*)$",
        cab,
        flags=re.I | re.S,
    )
    if m_dl:
        trecho_dl = m_dl.group(1)
    else:
        m_dl = re.search(
            r"Quantidade\s+de\s+docentes\s*:(.*)$",
            cab,
            flags=re.I | re.S,
        )
        if m_dl:
            trecho_dl = m_dl.group(1)

    # Docentes inline é o caso mais seguro.
    docentes = _primeiro([
        r"docentes[ \\t]*:[ \\t]*(\\d{1,2})(?:[ \\t]*\\([^\\n]*\\))?",
        r"Quantidade\\s+de\\s+docentes[ \\t]*:[ \\t]*(\\d{1,2})",
    ], cab)

    # Captura os valores numéricos no trecho da pequena tabela.
    nums = re.findall(r"\d+(?:[.,]\d+)?", trecho_dl)

    # Se não achou docentes inline, usa a heurística segura: docentes é o menor
    # inteiro de 1 a 10 quando há outro valor plausível de laboratório.
    if not docentes and nums:
        candidatos_doc = [n for n in nums if re.fullmatch(r"\d+", n) and 1 <= int(n) <= 10]
        if candidatos_doc:
            # Em planos com lab 0/1/2, preferimos o último valor pequeno, pois
            # pdfplumber costuma extrair laboratório antes de docentes.
            docentes = candidatos_doc[-1]

    info["Docentes"] = docentes or "Não informado"

    # Laboratório: "Não se aplica" => 0. Caso contrário, escolhe valor diferente
    # do docente quando há dois números; aceita decimais brasileiros.
    if re.search(r"n[ãa]o\s+se\s+aplica", trecho_dl, flags=re.I):
        lab = "0"
    else:
        lab = ""
        if nums:
            # Remove uma ocorrência do valor usado como docentes (quando possível).
            restantes = nums.copy()
            if docentes in restantes and len(restantes) > 1:
                # Se a extração veio "57 / 1", remove o 1. Se veio "1 / 04,5",
                # também remove o 1.
                restantes.remove(docentes)
            if len(nums) == 1 and docentes == nums[0]:
                # Há planos em que a célula de laboratório está realmente vazia.
                # Nesse caso o único número após o rótulo é a quantidade de docentes.
                lab = "0"
            elif restantes:
                # O laboratório é normalmente o primeiro valor restante.
                lab = restantes[0]
            elif nums and docentes not in nums:
                lab = nums[0]

        # Caso normal: valor imediatamente depois do rótulo de laboratório.
        direto = _primeiro([
            r"Carga\s+hor[áa]ria\s+prevista\s+em\s+laborat[óo]rio\s*:\s*(\d+(?:[.,]\d+)?)",
        ], cab)
        if direto:
            lab = direto

        if not lab:
            lab = "0"

    info["CH_laboratorio"] = lab

    return info


# -----------------------------------------------------------------------------
# Seções textuais
# -----------------------------------------------------------------------------

SECTION_RE = re.compile(
    r"(?im)^\s*(?P<num>[2-8])\s*[-–—]+\s*(?P<title>"
    r"CONHECIMENTOS\s+ESSENCIAIS\s+DO\s+CURR[ÍI]CULO\s+DE\s+REFER[ÊE]NCIA"
    r"|EMENTA"
    r"|OBJETIVOS"
    r"|[ÁA]REAS\s+DE\s+INTEGRA[ÇC][ÃA]O"
    r"|CONTE[ÚU]DO\s+PROGRAM[ÁA]TICO"
    r"|BIBLIOGRAFIA\s+B[ÁA]SICA"
    r"|BIBLIOGRAFIA\s+COMPLEMENTAR"
    r")\s*:?\s*$"
)


def _canonizar_titulo(titulo: str) -> str:
    t = re.sub(r"\s+", " ", titulo).upper().strip()
    t = t.replace("Á", "A").replace("Ã", "A").replace("Ç", "C").replace("Ú", "U").replace("Í", "I").replace("Ê", "E")
    if t.startswith("CONHECIMENTOS"):
        return "Conhecimentos"
    if t == "EMENTA":
        return "Ementa"
    if t == "OBJETIVOS":
        return "Objetivos"
    if t.startswith("AREAS DE INTEGRACAO"):
        return "Areas_integracao"
    if t.startswith("CONTEUDO PROGRAMATICO"):
        return "Conteudo"
    if t.startswith("BIBLIOGRAFIA BASICA"):
        return "Bibliografia_Basica"
    if t.startswith("BIBLIOGRAFIA COMPLEMENTAR"):
        return "Bibliografia_Complementar"
    raise ValueError(f"Título de seção desconhecido: {titulo!r}")


def extrair_secoes(bloco: str) -> Tuple[Dict[str, str], bool]:
    matches = list(SECTION_RE.finditer(bloco))
    secoes = {
        "Conhecimentos": "",
        "Ementa": "",
        "Objetivos": "",
        "Areas_integracao": "Não se aplica",
        "Conteudo": "",
        "Bibliografia_Basica": "",
        "Bibliografia_Complementar": "",
    }

    tinha_areas = False
    for i, m in enumerate(matches):
        chave = _canonizar_titulo(m.group("title"))
        inicio = m.end()
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(bloco)
        valor = limpar_campo(bloco[inicio:fim])

        if chave == "Areas_integracao":
            tinha_areas = True
            secoes[chave] = valor or "Não se aplica"
        else:
            secoes[chave] = valor

    return secoes, tinha_areas


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def ler_pdf(caminho_pdf: Path) -> str:
    paginas = []
    with pdfplumber.open(caminho_pdf) as pdf:
        for pagina in pdf.pages:
            bruto = _normalizar_unicode(pagina.extract_text() or "")
            linhas = bruto.splitlines()
            limpas = []
            for i, linha in enumerate(linhas):
                s = re.sub(r"\s+", " ", linha).strip()

                # Número de página: só removemos quando aparece perto das bordas
                # da página. Isso evita apagar valores legítimos como 57, 76 etc.
                perto_da_borda = i < 5 or i >= max(0, len(linhas) - 5)
                if perto_da_borda and re.fullmatch(r"\d{1,3}", s):
                    continue

                if _eh_ruido_exato(linha):
                    continue

                limpas.append(linha)

            paginas.append("\n".join(limpas))
    return "\n".join(paginas)


def separar_planos(texto: str) -> List[str]:
    texto = limpar_texto_pdf(texto)

    # Aceita 1- IDENTIFICAÇÃO, 1 – IDENTIFICAÇÃO, etc.
    partes = re.split(
        r"(?im)^\s*1\s*[-–—]+\s*IDENTIFICA[ÇC][ÃA]O\s*$",
        texto,
    )

    blocos = []
    for parte in partes[1:]:
        if re.search(r"Componente\s+curricular\s*:", parte, flags=re.I):
            blocos.append(parte.strip())
    return blocos


def extrair_pdf(caminho_pdf: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    bruto = ler_pdf(caminho_pdf)
    blocos = separar_planos(bruto)

    dados = []
    qtd_secoes_areas = 0

    for bloco in blocos:
        info = extrair_identificacao(bloco)
        secoes, tinha_areas = extrair_secoes(bloco)
        qtd_secoes_areas += int(tinha_areas)
        info.update(secoes)
        dados.append(info)

    df = pd.DataFrame(dados)
    for c in COLUNAS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLUNAS]

    estatisticas = {
        "blocos": len(blocos),
        "linhas": len(df),
        "secoes_areas_integracao": qtd_secoes_areas,
    }
    return df, estatisticas


# -----------------------------------------------------------------------------
# Validação forte deste PPC
# -----------------------------------------------------------------------------

def validar(df: pd.DataFrame, stats: Dict[str, int]) -> List[str]:
    erros: List[str] = []

    if len(df) != EXPECTED_COMPONENTS:
        erros.append(f"Esperados {EXPECTED_COMPONENTS} componentes; foram extraídos {len(df)}.")

    if df["Componente"].duplicated().any():
        dups = df.loc[df["Componente"].duplicated(keep=False), "Componente"].tolist()
        erros.append(f"Componentes duplicados: {dups}")

    # Campos que devem existir para todos os componentes deste documento.
    obrigatorios = [
        "Curso", "Componente", "Tipo", "Núcleo", "Sigla", "Aulas_semanais",
        "Total_aulas", "CH_presencial", "Docentes", "Conhecimentos", "Ementa",
        "Objetivos", "Conteudo", "Bibliografia_Basica", "Bibliografia_Complementar",
    ]
    for campo in obrigatorios:
        mask = df[campo].astype(str).str.strip().isin(["", "nan", "None"])
        if mask.any():
            comps = df.loc[mask, "Componente"].tolist()
            erros.append(f"Campo {campo} vazio em {len(comps)} componente(s): {comps}")

    if stats.get("secoes_areas_integracao") != 19:
        erros.append(
            f"Esperadas 19 seções explícitas de ÁREAS DE INTEGRAÇÃO; "
            f"foram encontradas {stats.get('secoes_areas_integracao')}."
        )

    # Verificações específicas que já sabemos serem sensíveis.
    esperado_nome = "Tópicos Avançados em Tecnologia da Informação e Comunicação"
    if esperado_nome not in set(df["Componente"]):
        erros.append("Nome multilinha de 'Tópicos Avançados...' não foi preservado corretamente.")

    # Não pode sobrar cabeçalho/rodapé nos campos textuais.
    campos_textuais = [
        "Conhecimentos", "Ementa", "Objetivos", "Areas_integracao", "Conteudo",
        "Bibliografia_Basica", "Bibliografia_Complementar",
    ]
    padroes_ruido = [
        "Projeto Pedagógico do Curso Técnico em Informática Integrado ao Ensino Médio",
        "TÉCNICO EM INFORMÁTICA INTEGRADO AO ENSINO MÉDIO",
    ]
    for campo in campos_textuais:
        for ruido in padroes_ruido:
            mask = df[campo].astype(str).str.contains(ruido, case=False, regex=False, na=False)
            if mask.any():
                erros.append(f"Ruído de cabeçalho ainda presente em {campo}: {df.loc[mask, 'Componente'].tolist()}")

    # As áreas de integração não podem continuar grudadas nos objetivos.
    mask = df["Objetivos"].astype(str).str.contains("ÁREAS DE INTEGRAÇÃO", case=False, regex=False, na=False)
    if mask.any():
        erros.append(f"Seção ÁREAS DE INTEGRAÇÃO vazou para Objetivos: {df.loc[mask, 'Componente'].tolist()}")

    # Validações pontuais contra valores do PDF.
    checks = {
        "Linguagem de Programação 1": {"CH_laboratorio": "85,5"},
        "Linguagem de Programação 2": {"CH_laboratorio": "85,5"},
        "Prática de Laboratório 2": {"CH_laboratorio": "85,5"},
        "Física 1": {"CH_laboratorio": "04,5"},
        "Algoritmos": {"Total_aulas": "76", "CH_presencial": "57", "CH_laboratorio": "57"},
        "Relações Humanas no Trabalho": {"CH_laboratorio": "0"},
        "Gestão de Negócios": {"CH_laboratorio": "0"},
    }
    idx = df.set_index("Componente", drop=False)
    for componente, campos in checks.items():
        if componente not in idx.index:
            erros.append(f"Componente esperado ausente: {componente}")
            continue
        row = idx.loc[componente]
        for campo, esperado in campos.items():
            obtido = str(row[campo]).strip()
            if obtido != esperado:
                erros.append(f"{componente} / {campo}: esperado {esperado!r}, obtido {obtido!r}.")

    # Português 2 deve conter pontos posteriores ao 4, para detectar truncamento.
    if "Língua Portuguesa 2" in idx.index:
        conteudo = str(idx.loc["Língua Portuguesa 2", "Conteudo"])
        for trecho in ["4.2.", "Intertextualidade", "Variação linguística regional"]:
            if trecho.lower() not in conteudo.lower():
                erros.append(f"Conteúdo de Língua Portuguesa 2 parece truncado: não contém {trecho!r}.")

    return erros


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai os planos de ensino do PPC para Excel.")
    parser.add_argument("pdf", type=Path, help="Caminho do PDF do PPC")
    parser.add_argument("-o", "--output", type=Path, default=Path("disciplinas.xlsx"), help="Arquivo XLSX de saída")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF não encontrado: {args.pdf}")

    df, stats = extrair_pdf(args.pdf)
    erros = validar(df, stats)

    print("\n========== RESUMO DA EXTRAÇÃO ==========")
    print(f"Componentes extraídos: {len(df)}")
    print(f"Seções explícitas de Áreas de Integração: {stats['secoes_areas_integracao']}")
    print(f"Áreas de integração com conteúdo real: {(~df['Areas_integracao'].str.strip().str.lower().isin(['não se aplica', 'não se aplica.'])).sum()}")

    if erros:
        print("\n========== FALHAS DE VALIDAÇÃO ==========")
        for erro in erros:
            print(f"❌ {erro}")
        print("\nO Excel NÃO foi salvo porque a validação encontrou problemas.")
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.output, index=False)

    print("\n========== VALIDAÇÃO ==========")
    print("✅ Todas as validações passaram.")
    print(f"✅ Arquivo salvo em: {args.output.resolve()}")


if __name__ == "__main__":
    main()
