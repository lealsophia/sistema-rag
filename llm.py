import time
from google import genai
from google.genai.errors import ServerError, ClientError

from config import GEMINI_API_KEY


client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# REESCREVER PERGUNTA
# =========================

def reescrever_pergunta(pergunta, historico=""):

    # Se não existe histórico, não precisa reescrever
    if not historico.strip():
        return pergunta

    prompt = f"""
Você deve reescrever a pergunta atual para que ela possa ser entendida
sem precisar consultar o histórico da conversa.

Use o histórico apenas para resolver referências como:
"ela", "ele", "essa disciplina", "esse componente",
"esse conteúdo" e expressões semelhantes.

Não responda à pergunta.

Não acrescente informações que não estejam presentes no histórico
ou na pergunta atual.

Retorne somente a pergunta reescrita.

Histórico da conversa:

{historico}

Pergunta atual:

{pergunta}
"""

    for tentativa in range(3):
        try:

            resposta = client.models.generate_content(
                model="models/gemini-3.5-flash",
                contents=prompt
            )

            pergunta_reescrita = resposta.text.strip()

            if pergunta_reescrita:
                return pergunta_reescrita

            return pergunta


        except ServerError as erro:

            if erro.code == 503:

                if tentativa < 2:
                    time.sleep(2 ** tentativa)
                    continue

                # Se o Gemini estiver indisponível,
                # usa a pergunta original
                return pergunta

            raise

        except ClientError as erro:
            if erro.code == 429:
                return pergunta

            raise



# =========================
# GERAR RESPOSTA
# =========================

def gerar_resposta(contexto, pergunta, historico=""):

    prompt = f"""
Você é um assistente acadêmico especializado no Projeto Pedagógico
do Curso Técnico em Informática do IFSP.

Responda à pergunta utilizando exclusivamente:

1. o contexto recuperado dos documentos;
2. o histórico da conversa, apenas para entender referências
   feitas pelo usuário.

Nunca invente informações.

Se uma informação factual não estiver presente no contexto recuperado,
responda:

"A informação não foi encontrada nos documentos utilizados pelo sistema."

Histórico da conversa:

{historico}

Contexto recuperado pelo RAG:

{contexto}

Pergunta atual:

{pergunta}
"""

    for tentativa in range(3):
        try:

            resposta = client.models.generate_content(
                model="models/gemini-3.5-flash",
                contents=prompt
            )

            return resposta.text

        except ClientError as erro:
            if erro.code == 429:
                return (
                    "O limite temporário de uso do modelo foi atingido. "
                    "Tente novamente em alguns instantes."
                )

            raise

        except ServerError as erro:

            if erro.code == 503:

                if tentativa < 2:
                    time.sleep(2 ** tentativa)
                    continue

                return (
                    "O modelo de linguagem está temporariamente "
                    "indisponível devido à alta demanda. "
                    "Tente novamente em alguns instantes."
                )

            raise