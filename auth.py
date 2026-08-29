import requests
import streamlit as st

from firestore import criar_usuario

# =====================================================
# FIREBASE
# =====================================================

FIREBASE_API_KEY = "AIzaSyCs6bKgKQ10D4R5GMiNTmjwWj_sv4lVCo0"


# =====================================================
# LOGIN
# =====================================================

def login(email: str, senha: str):

    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:"
        f"signInWithPassword?key={FIREBASE_API_KEY}"
    )

    payload = {
        "email": email,
        "password": senha,
        "returnSecureToken": True
    }

    r = requests.post(url, json=payload)

    if r.status_code == 200:

        dados = r.json()

        st.session_state["user"] = {
            "uid": dados["localId"],
            "email": dados["email"],
            "idToken": dados["idToken"],
            "refreshToken": dados["refreshToken"]
        }

        criar_usuario(
            dados["localId"],
            dados["email"]
        )

        return True, "Login realizado com sucesso."

    else:

        erro = r.json()["error"]["message"]

        return False, erro


# =====================================================
# CADASTRO
# =====================================================

def register(email: str, senha: str):

    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:"
        f"signUp?key={FIREBASE_API_KEY}"
    )

    payload = {
        "email": email,
        "password": senha,
        "returnSecureToken": True
    }

    r = requests.post(url, json=payload)

    if r.status_code == 200:

        dados = r.json()

        st.session_state["user"] = {
            "uid": dados["localId"],
            "email": dados["email"],
            "idToken": dados["idToken"],
            "refreshToken": dados["refreshToken"]
        }

        criar_usuario(
            dados["localId"],
            dados["email"]
        )

        return True, "Conta criada com sucesso."

    else:

        erro = r.json()["error"]["message"]

        return False, erro


# =====================================================
# RECUPERAR SENHA
# =====================================================

def recuperar_senha(email):

    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:"
        f"sendOobCode?key={FIREBASE_API_KEY}"
    )

    payload = {
        "requestType": "PASSWORD_RESET",
        "email": email
    }

    r = requests.post(url, json=payload)

    if r.status_code == 200:

        return True, "Email enviado."

    else:

        erro = r.json()["error"]["message"]

        return False, erro


# =====================================================
# LOGOUT
# =====================================================

def logout():

    st.session_state.clear()


# =====================================================
# USUÁRIO LOGADO?
# =====================================================

def usuario_logado():

    return "user" in st.session_state


# =====================================================
# DADOS DO USUÁRIO
# =====================================================

def usuario():

    return st.session_state.get("user")