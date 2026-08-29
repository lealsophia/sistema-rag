import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

if not firebase_admin._apps:

    cred = credentials.Certificate(
        "rag-ifsp-81d95-firebase-adminsdk-fbsvc-e79e6d42bf.json"
    )

    firebase_admin.initialize_app(cred)

db = firestore.client()