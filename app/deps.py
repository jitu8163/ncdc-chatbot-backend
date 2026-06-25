"""Shared FastAPI dependencies (current user / admin guard)."""
from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, UserRole
from app.security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
# Same scheme, but doesn't 401 when the Authorization header is absent — used by
# endpoints that allow anonymous access (e.g. the public chat with a free-message
# allowance) yet still want to identify the user when a valid token is present.
optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_optional_user(
    token: str | None = Depends(optional_oauth2_scheme), db: Session = Depends(get_db)
) -> User | None:
    """Resolve the signed-in user, or None for an anonymous caller. Never raises:
    a missing, malformed, expired or unknown token simply degrades to anonymous."""
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
    except jwt.PyJWTError:
        return None
    if not user_id:
        return None
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        return None
    return user


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise credentials_exc
    except jwt.PyJWTError as exc:
        raise credentials_exc from exc

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise credentials_exc
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )
    return user
