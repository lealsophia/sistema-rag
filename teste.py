import streamlit as st
from auth import *

st.title("Teste Firebase")

aba = st.radio(
    "",
    ["Login", "Cadastro", "Recuperar senha"]
)

if aba == "Cadastro":

    email = st.text_input("Email")
    senha = st.text_input("Senha", type="password")

    if st.button("Cadastrar"):
        ok, msg = register(email, senha)

        if ok:
            st.success(msg)
        else:
            st.error(msg)

elif aba == "Login":

    email = st.text_input("Email")
    senha = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        ok, msg = login(email, senha)

        if ok:
            st.success(msg)
        else:
            st.error(msg)

else:

    email = st.text_input("Email")

    if st.button("Enviar email"):
        ok, msg = recuperar_senha(email)

        if ok:
            st.success(msg)
        else:
            st.error(msg)

if usuario_logado():

    st.divider()

    st.write(usuario())

    if st.button("Logout"):
        logout()
        st.rerun()