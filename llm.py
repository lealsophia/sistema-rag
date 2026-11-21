
from groq import Groq
from config import GROQ_API_KEY


# =========================
# CONFIGURAÇÃO GROQ
# =========================

client = Groq(api_key=GROQ_API_KEY)

MODELO = "openai/gpt-oss-120b"


# =========================
# FUNÇÃO AUXILIAR
# =========================

def consultar_groq(prompt):

    resposta = client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=2048
    )

    texto = resposta.choices[0].message.content

    return texto.strip() if texto else ""


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

    try:
        pergunta_reescrita = consultar_groq(prompt)

        if pergunta_reescrita:
            return pergunta_reescrita

        return pergunta

    except Exception as erro:
        print("Erro ao reescrever pergunta com Groq:", erro)
        return pergunta


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

    try:
        resposta = consultar_groq(prompt)

        print("\n===== DEBUG GROQ =====")
        print("Resposta:", repr(resposta))
        print("======================\n")

        if resposta:
            return resposta

        return "Não foi possível gerar uma resposta. Tente novamente."

    except Exception as erro:
        print("Erro ao gerar resposta com Groq:", erro)

        return (
            "Não foi possível gerar uma resposta no momento. "
            "Verifique a conexão com o serviço de IA e tente novamente."
        )
