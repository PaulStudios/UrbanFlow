import os

from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from .routers import traffic_signals, auth
from .database import SessionLocal, Base, engine
from .models import User
from .utils.auth import get_current_user

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth.router)
app.include_router(traffic_signals.router)

@app.get("/")
async def root(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("base.html", {"request": request, "current_user": current_user})
