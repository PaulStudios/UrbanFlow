import firebase_admin
from firebase_admin import credentials, firestore, storage

# Path to your service account key file
cred = credentials.Certificate('firebase-accesskey.json')
firebase_admin.initialize_app(cred)

firestore_db = firestore.client()
firebase_bucket = storage.bucket("img_store")
