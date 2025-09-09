#!/usr/bin/env python3
"""
Authentication utilities for the forecasting application
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database import get_db, User
import os

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return username"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    import logging
    logger = logging.getLogger("auth")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    username = verify_token(credentials.credentials)
    if username is None:
        logger.warning("Token verification failed: invalid token")
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        logger.warning(f"User not found for username: {username}")
        raise credentials_exception
    
    if not user.is_active:
        logger.warning(f"Inactive user attempted access: {username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    if not user.is_approved and not user.is_admin:
        logger.warning(f"Unapproved user attempted access: {username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not approved by admin"
        )
    
    return user

def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None"""
    if not credentials:
        return None
    
    try:
        username = verify_token(credentials.credentials)
        if username is None:
            return None
        
        user = db.query(User).filter(User.username == username).first()
        if user is None or not user.is_active:
            return None
        
        # Allow only approved users or admins
        if not user.is_approved and not user.is_admin:
            return None
        
        return user
    except:
        return None