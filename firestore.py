from firebase_config import db
from firebase_admin import firestore
import time


# =========================
# RATE LIMIT
# =========================

LIMITE_PERGUNTAS_MINUTO = 5
JANELA_RATE_LIMIT = 60


def verificar_rate_limit(uid):

    agora = time.time()

    limite_ref = (
        db.collection("users")
        .document(uid)
        .collection("system")
        .document("rate_limit")
    )

    transaction = db.transaction()

    @firestore.transactional
    def atualizar_limite(transaction):

        snapshot = limite_ref.get(
            transaction=transaction
        )

        # Primeira pergunta do usuário
        if not snapshot.exists:

            transaction.set(
                limite_ref,
                {
                    "inicio_janela": agora,
                    "quantidade": 1
                }
            )

            return True, 0

        dados = snapshot.to_dict()

        inicio_janela = dados.get(
            "inicio_janela",
            agora
        )

        quantidade = dados.get(
            "quantidade",
            0
        )

        tempo_decorrido = (
            agora - inicio_janela
        )

        # =========================
        # NOVA JANELA
        # =========================

        if tempo_decorrido >= JANELA_RATE_LIMIT:

            transaction.set(
                limite_ref,
                {
                    "inicio_janela": agora,
                    "quantidade": 1
                }
            )

            return True, 0

        # =========================
        # LIMITE ATINGIDO
        # =========================

        if quantidade >= LIMITE_PERGUNTAS_MINUTO:

            segundos_restantes = int(
                JANELA_RATE_LIMIT - tempo_decorrido
            ) + 1

            return False, segundos_restantes

        # =========================
        # REGISTRA NOVA PERGUNTA
        # =========================

        transaction.update(
            limite_ref,
            {
                "quantidade": quantidade + 1
            }
        )

        return True, 0

    return atualizar_limite(transaction)

# ==========================================
# CRIAR OU ATUALIZAR USUÁRIO
# ==========================================

def criar_usuario(uid, email):

    doc_ref = db.collection("users").document(uid)

    doc = doc_ref.get()

    if not doc.exists:

        doc_ref.set({

            "email": email,

            "created_at": firestore.SERVER_TIMESTAMP,

            "last_login": firestore.SERVER_TIMESTAMP

        })

    else:

        doc_ref.update({

            "last_login": firestore.SERVER_TIMESTAMP

        })


# ==========================================
# OBTER DADOS DO USUÁRIO
# ==========================================

def obter_usuario(uid):

    doc = db.collection("users").document(uid).get()

    if doc.exists:

        return doc.to_dict()

    return None

from firebase_config import db
from firebase_admin import firestore


def criar_conversa(uid, titulo="Nova conversa"):

    conversa_ref = (
        db.collection("users")
          .document(uid)
          .collection("conversations")
          .document()
    )

    conversa_ref.set({

        "title": titulo,

        "created_at": firestore.SERVER_TIMESTAMP,

        "updated_at": firestore.SERVER_TIMESTAMP

    })

    return conversa_ref.id

def listar_conversas(uid):

    docs = (
        db.collection("users")
          .document(uid)
          .collection("conversations")
          .order_by("updated_at", direction=firestore.Query.DESCENDING)
          .stream()
    )

    conversas = []

    for doc in docs:

        dados = doc.to_dict()

        dados["id"] = doc.id

        conversas.append(dados)

    return conversas

def atualizar_titulo_conversa(uid, conversation_id, titulo):

    db.collection("users") \
      .document(uid) \
      .collection("conversations") \
      .document(conversation_id) \
      .update({

          "title": titulo,
          "updated_at": firestore.SERVER_TIMESTAMP

      })

from google.cloud import firestore
def salvar_mensagem(uid, conversation_id, role, content, fontes=None):

    dados_mensagem = {
        "role": role,
        "content": content,
        "created_at": firestore.SERVER_TIMESTAMP
    }

    # Adiciona as fontes apenas quando existirem
    if fontes is not None:
        dados_mensagem["fontes"] = fontes

    db.collection("users") \
      .document(uid) \
      .collection("conversations") \
      .document(conversation_id) \
      .collection("messages") \
      .add(dados_mensagem)

    # Atualiza a data da conversa
    db.collection("users") \
      .document(uid) \
      .collection("conversations") \
      .document(conversation_id) \
      .update({
          "updated_at": firestore.SERVER_TIMESTAMP
      })

def listar_mensagens(uid, conversation_id):

    mensagens_ref = (
        db.collection("users")
        .document(uid)
        .collection("conversations")
        .document(conversation_id)
        .collection("messages")
        .order_by("created_at")
        .stream()
    )

    mensagens = []

    for doc in mensagens_ref:
        dados = doc.to_dict()

        mensagens.append({
            "id": doc.id,
            "role": dados.get("role"),
            "content": dados.get("content"),
            "fontes": dados.get("fontes", [])
        })

    return mensagens

def excluir_conversa(uid, conversation_id):

    conversa_ref = (
        db.collection("users")
        .document(uid)
        .collection("conversations")
        .document(conversation_id)
    )

    mensagens_ref = conversa_ref.collection("messages")

    # Apaga todas as mensagens
    for mensagem in mensagens_ref.stream():
        mensagem.reference.delete()

    # Apaga a conversa
    conversa_ref.delete()