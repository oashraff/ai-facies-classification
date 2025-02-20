from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from app import app as flask_app  # assuming your Flask app is defined in app.py

app = FastAPI()
app.mount("/", WSGIMiddleware(flask_app))
