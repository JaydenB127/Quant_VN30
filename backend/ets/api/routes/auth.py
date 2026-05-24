# -*- coding: utf-8 -*-
"""
Authentication router.
Handles User registration, login, and profile operations.
"""
from __future__ import annotations

import logging
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.models import User as DBUser, UserSession as DBUserSession
from ets.api.routes.datasets import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Hash password using PBKDF2-HMAC-SHA256."""
    if not salt:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()
    return pw_hash, salt


@router.post("/register")
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user in the system."""
    # Check if username exists
    stmt = select(DBUser).where(DBUser.username == req.username)
    res = await db.execute(stmt)
    if res.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    password_hash, salt = hash_password(req.password)
    user = DBUser(
        username=req.username,
        password_hash=password_hash,
        salt=salt,
        email=req.email
    )
    db.add(user)
    await db.commit()
    
    return {"status": "success", "message": "User registered successfully"}


@router.post("/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate credentials and return a session token."""
    stmt = select(DBUser).where(DBUser.username == req.username)
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    input_hash, _ = hash_password(req.password, user.salt)
    if input_hash != user.password_hash:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    # Generate session token
    token = secrets.token_hex(32)
    session = DBUserSession(
        user_id=user.id,
        token=token,
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(session)
    await db.commit()
    
    return {
        "status": "success",
        "token": token,
        "username": user.username,
        "email": user.email
    }


@router.get("/me")
async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve logged-in user profile from token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")
    
    token = authorization.split(" ")[1]
    stmt = select(DBUserSession).where(DBUserSession.token == token)
    res = await db.execute(stmt)
    session = res.scalar_one_or_none()
    if not session or session.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    
    stmt_user = select(DBUser).where(DBUser.id == session.user_id)
    res_user = await db.execute(stmt_user)
    user = res_user.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
        
    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "created_at": user.created_at.isoformat()
    }
