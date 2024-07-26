from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from .. import models, schemas, database
from ..utils.auth import create_access_token, verify_password, oauth2_scheme, get_password_hash
from datetime import timedelta

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=schemas.User)
async def register_user(user: schemas.UserCreate, request: Request, db: Session = Depends(get_db)):
    if request.client.host != "127.0.0.1" and request.client.host != "::1":
        raise HTTPException(status_code=403, detail="This endpoint can only be accessed from localhost")

    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.get("/register", response_class=HTMLResponse)
async def register_form():
    return """
    <html>
        <body>
            <h2>Register New User</h2>
            <form method="post">
                <input type="text" name="username" placeholder="Username" required><br>
                <input type="password" name="password" placeholder="Password" required><br>
                <input type="submit" value="Register">
            </form>
        </body>
    </html>
    """
