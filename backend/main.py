#!/usr/bin/env python3
"""
Advanced Multi-variant Forecasting API with PostgreSQL

This API provides comprehensive forecasting capabilities with multiple algorithms,
database integration, user authentication, and advanced features.
"""

import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, and_, or_, text, desc
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

# Import database components
from database import (
    get_db, init_database, ForecastData, ForecastConfiguration, 
    ProductCustomerLocationCombination, ExternalFactorData,
    User, SavedForecastResult, ScheduledForecast, ForecastExecution
)
from database_utils import get_or_create_combination, get_unique_values, get_aggregated_data_optimized
from auth import get_current_user, get_current_user_optional, create_access_token
from validation import DateRangeValidator
from model_persistence import ModelPersistenceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting up forecasting API...")
    if not init_database():
        logger.error("Failed to initialize database")
        raise RuntimeError("Database initialization failed")
    
    # Import and start scheduler
    try:
        from scheduler import start_scheduler
        start_scheduler()
        logger.info("Forecast scheduler started")
    except Exception as e:
        logger.warning(f"Failed to start scheduler: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down forecasting API...")
    try:
        from scheduler import stop_scheduler
        stop_scheduler()
        logger.info("Forecast scheduler stopped")
    except Exception as e:
        logger.warning(f"Failed to stop scheduler: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Multi-variant Forecasting API",
    description="Comprehensive forecasting API with multiple algorithms and database integration",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = Field(None, max_length=100)

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_approved: bool
    is_admin: bool
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class ForecastConfig(BaseModel):
    forecastBy: str
    selectedItem: Optional[str] = None
    selectedProduct: Optional[str] = None
    selectedCustomer: Optional[str] = None
    selectedLocation: Optional[str] = None
    selectedProducts: Optional[List[str]] = None
    selectedCustomers: Optional[List[str]] = None
    selectedLocations: Optional[List[str]] = None
    selectedItems: Optional[List[str]] = None
    algorithm: str = 'best_fit'
    interval: str = 'month'
    historicPeriod: int = 12
    forecastPeriod: int = 6
    multiSelect: Optional[bool] = False
    advancedMode: Optional[bool] = False
    externalFactors: Optional[List[str]] = None

class DataViewRequest(BaseModel):
    product: Optional[str] = None
    customer: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    page: int = 1
    page_size: int = 50

class SaveConfigRequest(BaseModel):
    name: str
    description: Optional[str] = None
    config: ForecastConfig

class SaveForecastRequest(BaseModel):
    name: str
    description: Optional[str] = None
    forecast_config: ForecastConfig
    forecast_data: Dict[str, Any]

class ScheduledForecastCreate(BaseModel):
    name: str
    description: Optional[str] = None
    forecast_config: ForecastConfig
    frequency: str  # 'daily', 'weekly', 'monthly'
    start_date: str
    end_date: Optional[str] = None

class ScheduledForecastUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: Optional[str] = None

class FredDataRequest(BaseModel):
    series_ids: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class UserActiveUpdate(BaseModel):
    is_active: bool

# Health check endpoint
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"message": "Advanced Multi-variant Forecasting API is running"}

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Check if username or email already exists
        existing_user = db.query(User).filter(
            or_(User.username == user_data.username, User.email == user_data.email)
        ).first()
        
        if existing_user:
            if existing_user.username == user_data.username:
                raise HTTPException(status_code=400, detail="Username already registered")
            else:
                raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=User.hash_password(user_data.password),
            full_name=user_data.full_name,
            is_active=True,
            is_approved=False,  # Requires admin approval
            is_admin=False
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            full_name=new_user.full_name,
            is_active=new_user.is_active,
            is_approved=new_user.is_approved,
            is_admin=new_user.is_admin,
            created_at=new_user.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login user and return access token"""
    try:
        user = db.query(User).filter(User.username == credentials.username).first()
        
        if not user or not user.verify_password(credentials.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account is deactivated"
            )
        
        # Create access token
        access_token = create_access_token(data={"sub": user.username})
        
        user_response = UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_approved=user.is_approved,
            is_admin=user.is_admin,
            created_at=user.created_at
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_approved=current_user.is_approved,
        is_admin=current_user.is_admin,
        created_at=current_user.created_at
    )

# Admin endpoints
@app.get("/admin/users", response_model=List[UserResponse])
async def list_users(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all users (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_approved=user.is_approved,
            is_admin=user.is_admin,
            created_at=user.created_at
        )
        for user in users
    ]

@app.post("/admin/users/{user_id}/approve", response_model=UserResponse)
async def approve_user(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Approve a user (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_approved = True
    db.commit()
    db.refresh(user)
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_approved=user.is_approved,
        is_admin=user.is_admin,
        created_at=user.created_at
    )

@app.post("/admin/users/{user_id}/active", response_model=UserResponse)
async def set_user_active(user_id: int, update_data: UserActiveUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Activate/deactivate a user (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.is_admin:
        raise HTTPException(status_code=400, detail="Cannot deactivate admin users")
    
    user.is_active = update_data.is_active
    db.commit()
    db.refresh(user)
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_approved=user.is_approved,
        is_admin=user.is_admin,
        created_at=user.created_at
    )

# Database endpoints
@app.get("/database/stats")
async def get_database_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get database statistics"""
    try:
        # Get total records from forecast_data
        total_records = db.query(func.count(ForecastData.id)).scalar() or 0
        
        if total_records == 0:
            return {
                "totalRecords": 0,
                "dateRange": {"start": "", "end": ""},
                "uniqueProducts": 0,
                "uniqueCustomers": 0,
                "uniqueLocations": 0
            }
        
        # Get date range from forecast_data
        date_range = db.query(
            func.min(ForecastData.date).label('min_date'),
            func.max(ForecastData.date).label('max_date')
        ).first()
        
        # Get unique counts from combinations table
        unique_products = db.query(func.count(func.distinct(ProductCustomerLocationCombination.product))).filter(
            ProductCustomerLocationCombination.product.isnot(None)
        ).scalar() or 0
        
        unique_customers = db.query(func.count(func.distinct(ProductCustomerLocationCombination.customer))).filter(
            ProductCustomerLocationCombination.customer.isnot(None)
        ).scalar() or 0
        
        unique_locations = db.query(func.count(func.distinct(ProductCustomerLocationCombination.location))).filter(
            ProductCustomerLocationCombination.location.isnot(None)
        ).scalar() or 0
        
        return {
            "totalRecords": total_records,
            "dateRange": {
                "start": date_range.min_date.isoformat() if date_range.min_date else "",
                "end": date_range.max_date.isoformat() if date_range.max_date else ""
            },
            "uniqueProducts": unique_products,
            "uniqueCustomers": unique_customers,
            "uniqueLocations": unique_locations
        }
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get database statistics: {str(e)}")

@app.get("/database/options")
async def get_database_options(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get unique products, customers, and locations"""
    try:
        # Get unique values from combinations table
        products = db.query(ProductCustomerLocationCombination.product).filter(
            ProductCustomerLocationCombination.product.isnot(None)
        ).distinct().all()
        products = sorted([p[0] for p in products if p[0]])
        
        customers = db.query(ProductCustomerLocationCombination.customer).filter(
            ProductCustomerLocationCombination.customer.isnot(None)
        ).distinct().all()
        customers = sorted([c[0] for c in customers if c[0]])
        
        locations = db.query(ProductCustomerLocationCombination.location).filter(
            ProductCustomerLocationCombination.location.isnot(None)
        ).distinct().all()
        locations = sorted([l[0] for l in locations if l[0]])
        
        return {
            "products": products,
            "customers": customers,
            "locations": locations
        }
        
    except Exception as e:
        logger.error(f"Error getting database options: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get database options: {str(e)}")

@app.post("/database/filtered_options")
async def get_filtered_options(
    filters: Dict[str, List[str]],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get filtered unique values based on selected filters"""
    try:
        query = db.query(ProductCustomerLocationCombination)
        
        # Apply filters
        if filters.get('selectedProducts'):
            query = query.filter(ProductCustomerLocationCombination.product.in_(filters['selectedProducts']))
        if filters.get('selectedCustomers'):
            query = query.filter(ProductCustomerLocationCombination.customer.in_(filters['selectedCustomers']))
        if filters.get('selectedLocations'):
            query = query.filter(ProductCustomerLocationCombination.location.in_(filters['selectedLocations']))
        
        combinations = query.all()
        
        # Extract unique values from filtered combinations
        products = sorted(list(set([c.product for c in combinations if c.product])))
        customers = sorted(list(set([c.customer for c in combinations if c.customer])))
        locations = sorted(list(set([c.location for c in combinations if c.location])))
        
        return {
            "products": products,
            "customers": customers,
            "locations": locations
        }
        
    except Exception as e:
        logger.error(f"Error getting filtered options: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get filtered options: {str(e)}")

@app.post("/database/view")
async def view_database_data(
    request: DataViewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """View database data with filtering and pagination"""
    try:
        # Build query with joins
        query = db.query(ForecastData, ProductCustomerLocationCombination).join(
            ProductCustomerLocationCombination,
            ForecastData.combination_id == ProductCustomerLocationCombination.id
        )
        
        # Apply filters
        if request.product:
            query = query.filter(ProductCustomerLocationCombination.product == request.product)
        if request.customer:
            query = query.filter(ProductCustomerLocationCombination.customer == request.customer)
        if request.location:
            query = query.filter(ProductCustomerLocationCombination.location == request.location)
        if request.start_date:
            query = query.filter(ForecastData.date >= request.start_date)
        if request.end_date:
            query = query.filter(ForecastData.date <= request.end_date)
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination
        offset = (request.page - 1) * request.page_size
        results = query.offset(offset).limit(request.page_size).all()
        
        # Format results
        data = []
        for forecast_data, combination in results:
            data.append({
                'id': forecast_data.id,
                'product': combination.product,
                'customer': combination.customer,
                'location': combination.location,
                'product_group': combination.product_group,
                'product_hierarchy': combination.product_hierarchy,
                'location_region': combination.location_region,
                'customer_group': combination.customer_group,
                'customer_region': combination.customer_region,
                'ship_to_party': combination.ship_to_party,
                'sold_to_party': combination.sold_to_party,
                'quantity': float(forecast_data.quantity),
                'uom': forecast_data.uom,
                'date': forecast_data.date.isoformat(),
                'unit_price': float(forecast_data.unit_price) if forecast_data.unit_price else None,
                'created_at': forecast_data.created_at.isoformat(),
                'updated_at': forecast_data.updated_at.isoformat()
            })
        
        total_pages = (total_count + request.page_size - 1) // request.page_size
        
        return {
            "data": data,
            "total_records": total_count,
            "page": request.page,
            "page_size": request.page_size,
            "total_pages": total_pages
        }
        
    except Exception as e:
        logger.error(f"Error viewing database data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to view database data: {str(e)}")

# File upload endpoint
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and process Excel/CSV file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel (.xlsx, .xls) and CSV files are supported")
        
        # Read file content
        content = await file.read()
        
        # Process based on file type
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(pd.io.common.BytesIO(content))
        else:
            df = pd.read_excel(pd.io.common.BytesIO(content))
        
        # Validate required columns
        required_columns = ['date', 'quantity']
        missing_columns = [col for col in required_columns if col.lower() not in [c.lower() for c in df.columns]]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Normalize column names (case-insensitive)
        column_mapping = {}
        for col in df.columns:
            lower_col = col.lower().strip()
            if lower_col in ['date']:
                column_mapping[col] = 'date'
            elif lower_col in ['quantity', 'qty']:
                column_mapping[col] = 'quantity'
            elif lower_col in ['product']:
                column_mapping[col] = 'product'
            elif lower_col in ['customer']:
                column_mapping[col] = 'customer'
            elif lower_col in ['location']:
                column_mapping[col] = 'location'
            elif lower_col in ['product_group', 'productgroup']:
                column_mapping[col] = 'product_group'
            elif lower_col in ['product_hierarchy', 'producthierarchy']:
                column_mapping[col] = 'product_hierarchy'
            elif lower_col in ['location_region', 'locationregion']:
                column_mapping[col] = 'location_region'
            elif lower_col in ['customer_group', 'customergroup']:
                column_mapping[col] = 'customer_group'
            elif lower_col in ['customer_region', 'customerregion']:
                column_mapping[col] = 'customer_region'
            elif lower_col in ['ship_to_party', 'shiptoparty']:
                column_mapping[col] = 'ship_to_party'
            elif lower_col in ['sold_to_party', 'soldtoparty']:
                column_mapping[col] = 'sold_to_party'
            elif lower_col in ['uom', 'unit_of_measure']:
                column_mapping[col] = 'uom'
            elif lower_col in ['unit_price', 'unitprice', 'price']:
                column_mapping[col] = 'unit_price'
        
        df = df.rename(columns=column_mapping)
        
        # Ensure at least one dimension column exists
        dimension_columns = ['product', 'customer', 'location']
        if not any(col in df.columns for col in dimension_columns):
            raise HTTPException(
                status_code=400,
                detail="At least one dimension column (product, customer, or location) is required"
            )
        
        # Process data
        inserted_count = 0
        duplicate_count = 0
        
        for _, row in df.iterrows():
            try:
                # Parse date
                date_value = pd.to_datetime(row['date']).date()
                
                # Parse quantity
                quantity_value = float(row['quantity'])
                
                # Get or create combination
                combination_id = get_or_create_combination(
                    db=db,
                    product=row.get('product'),
                    customer=row.get('customer'),
                    location=row.get('location'),
                    product_group=row.get('product_group'),
                    product_hierarchy=row.get('product_hierarchy'),
                    location_region=row.get('location_region'),
                    customer_group=row.get('customer_group'),
                    customer_region=row.get('customer_region'),
                    ship_to_party=row.get('ship_to_party'),
                    sold_to_party=row.get('sold_to_party')
                )
                
                # Create forecast data record
                forecast_data = ForecastData(
                    combination_id=combination_id,
                    quantity=quantity_value,
                    uom=row.get('uom'),
                    date=date_value,
                    unit_price=float(row['unit_price']) if pd.notna(row.get('unit_price')) else None
                )
                
                db.add(forecast_data)
                db.commit()
                inserted_count += 1
                
            except IntegrityError:
                db.rollback()
                duplicate_count += 1
                continue
            except Exception as e:
                db.rollback()
                logger.error(f"Error processing row: {e}")
                continue
        
        # Get total records after upload
        total_records = db.query(func.count(ForecastData.id)).scalar()
        
        return {
            "message": f"File processed successfully",
            "inserted": inserted_count,
            "duplicates": duplicate_count,
            "totalRecords": total_records,
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

# External factors endpoints
@app.get("/external_factors")
async def get_external_factors(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get list of available external factors"""
    try:
        factors = db.query(ExternalFactorData.factor_name).distinct().all()
        factor_names = [f[0] for f in factors]
        return {"external_factors": sorted(factor_names)}
    except Exception as e:
        logger.error(f"Error getting external factors: {e}")
        raise HTTPException(status_code=500, detail="Failed to get external factors")

@app.post("/upload_external_factors")
async def upload_external_factors(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload external factor data"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel (.xlsx, .xls) and CSV files are supported")
        
        # Read file content
        content = await file.read()
        
        # Process based on file type
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(pd.io.common.BytesIO(content))
        else:
            df = pd.read_excel(pd.io.common.BytesIO(content))
        
        # Validate required columns
        required_columns = ['date', 'factor_name', 'factor_value']
        missing_columns = [col for col in required_columns if col.lower() not in [c.lower() for c in df.columns]]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}. Required: date, factor_name, factor_value"
            )
        
        # Normalize column names
        column_mapping = {}
        for col in df.columns:
            lower_col = col.lower().strip()
            if lower_col == 'date':
                column_mapping[col] = 'date'
            elif lower_col in ['factor_name', 'factorname']:
                column_mapping[col] = 'factor_name'
            elif lower_col in ['factor_value', 'factorvalue', 'value']:
                column_mapping[col] = 'factor_value'
        
        df = df.rename(columns=column_mapping)
        
        # Validate data with main forecast data
        validation_result = DateRangeValidator.validate_upload_data(df, db)
        
        # Process data
        inserted_count = 0
        duplicate_count = 0
        
        for _, row in df.iterrows():
            try:
                # Parse date
                date_value = pd.to_datetime(row['date']).date()
                
                # Parse factor value
                factor_value = float(row['factor_value'])
                
                # Create external factor record
                external_factor = ExternalFactorData(
                    date=date_value,
                    factor_name=str(row['factor_name']).strip(),
                    factor_value=factor_value
                )
                
                db.add(external_factor)
                db.commit()
                inserted_count += 1
                
            except IntegrityError:
                db.rollback()
                duplicate_count += 1
                continue
            except Exception as e:
                db.rollback()
                logger.error(f"Error processing external factor row: {e}")
                continue
        
        return {
            "message": f"External factors processed successfully",
            "inserted": inserted_count,
            "duplicates": duplicate_count,
            "filename": file.filename,
            "validation": validation_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"External factor upload error: {e}")
        raise HTTPException(status_code=500, detail=f"External factor processing failed: {str(e)}")

# FRED data endpoints
@app.post("/fetch_fred_data")
async def fetch_fred_data(
    request: FredDataRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Fetch data from FRED API and store as external factors"""
    try:
        import requests
        
        # Check if FRED API key is configured
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            raise HTTPException(
                status_code=500, 
                detail="FRED API key not configured. Please set FRED_API_KEY environment variable."
            )
        
        total_inserted = 0
        total_duplicates = 0
        series_details = []
        
        for series_id in request.series_ids:
            try:
                # Fetch data from FRED API
                params = {
                    'series_id': series_id,
                    'api_key': fred_api_key,
                    'file_type': 'json'
                }
                
                if request.start_date:
                    params['observation_start'] = request.start_date
                if request.end_date:
                    params['observation_end'] = request.end_date
                
                response = requests.get(
                    'https://api.stlouisfed.org/fred/series/observations',
                    params=params,
                    timeout=30
                )
                
                if response.status_code != 200:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': f'FRED API error: {response.status_code}',
                        'inserted': 0
                    })
                    continue
                
                data = response.json()
                
                if 'observations' not in data:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': 'No observations found in FRED response',
                        'inserted': 0
                    })
                    continue
                
                # Process observations
                inserted_count = 0
                duplicate_count = 0
                
                for obs in data['observations']:
                    try:
                        if obs['value'] == '.':  # FRED uses '.' for missing values
                            continue
                        
                        date_value = pd.to_datetime(obs['date']).date()
                        factor_value = float(obs['value'])
                        
                        # Create external factor record
                        external_factor = ExternalFactorData(
                            date=date_value,
                            factor_name=series_id,
                            factor_value=factor_value
                        )
                        
                        db.add(external_factor)
                        db.commit()
                        inserted_count += 1
                        
                    except IntegrityError:
                        db.rollback()
                        duplicate_count += 1
                        continue
                    except Exception as e:
                        db.rollback()
                        logger.error(f"Error processing FRED observation: {e}")
                        continue
                
                total_inserted += inserted_count
                total_duplicates += duplicate_count
                
                series_details.append({
                    'series_id': series_id,
                    'status': 'success',
                    'message': f'Successfully processed {inserted_count} observations',
                    'inserted': inserted_count,
                    'duplicates': duplicate_count
                })
                
            except Exception as e:
                logger.error(f"Error processing FRED series {series_id}: {e}")
                series_details.append({
                    'series_id': series_id,
                    'status': 'error',
                    'message': str(e),
                    'inserted': 0
                })
        
        return {
            "message": f"FRED data fetch completed",
            "inserted": total_inserted,
            "duplicates": total_duplicates,
            "series_processed": len(request.series_ids),
            "series_details": series_details
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FRED data fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"FRED data fetch failed: {str(e)}")

@app.get("/fred_series_info")
async def get_fred_series_info(current_user: User = Depends(get_current_user)):
    """Get information about popular FRED series"""
    series_info = {
        "message": "Popular FRED Economic Data Series",
        "series": {
            "Economic Growth": {
                "GDP": "Gross Domestic Product",
                "GDPC1": "Real Gross Domestic Product",
                "GDPPOT": "Real Potential Gross Domestic Product"
            },
            "Inflation": {
                "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
                "CPILFESL": "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy",
                "PCEPI": "Personal Consumption Expenditures: Chain-type Price Index"
            },
            "Employment": {
                "UNRATE": "Unemployment Rate",
                "CIVPART": "Labor Force Participation Rate",
                "PAYEMS": "All Employees, Total Nonfarm"
            },
            "Interest Rates": {
                "FEDFUNDS": "Federal Funds Effective Rate",
                "DGS10": "10-Year Treasury Constant Maturity Rate",
                "DGS2": "2-Year Treasury Constant Maturity Rate"
            },
            "Money Supply": {
                "M1SL": "M1 Money Stock",
                "M2SL": "M2 Money Stock",
                "BASE": "St. Louis Adjusted Monetary Base"
            }
        },
        "note": "Visit https://fred.stlouisfed.org to explore more series. You need a free FRED API key."
    }
    
    return series_info

# Configuration management endpoints
@app.get("/configurations")
async def get_configurations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all saved configurations"""
    try:
        configs = db.query(ForecastConfiguration).order_by(desc(ForecastConfiguration.updated_at)).all()
        
        configurations = []
        for config in configs:
            # Parse the stored configuration
            forecast_config = {
                'forecastBy': config.forecast_by,
                'selectedItem': config.selected_item,
                'selectedProduct': config.selected_product,
                'selectedCustomer': config.selected_customer,
                'selectedLocation': config.selected_location,
                'algorithm': config.algorithm,
                'interval': config.interval,
                'historicPeriod': config.historic_period,
                'forecastPeriod': config.forecast_period
            }
            
            configurations.append({
                'id': config.id,
                'name': config.name,
                'description': config.description,
                'config': forecast_config,
                'createdAt': config.created_at.isoformat(),
                'updatedAt': config.updated_at.isoformat()
            })
        
        return {"configurations": configurations}
        
    except Exception as e:
        logger.error(f"Error getting configurations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get configurations")

@app.post("/configurations")
async def save_configuration(
    request: SaveConfigRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save a forecast configuration"""
    try:
        # Check if name already exists
        existing = db.query(ForecastConfiguration).filter(ForecastConfiguration.name == request.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Configuration name already exists")
        
        # Create new configuration
        config = ForecastConfiguration(
            name=request.name,
            description=request.description,
            forecast_by=request.config.forecastBy,
            selected_item=request.config.selectedItem,
            selected_product=request.config.selectedProduct,
            selected_customer=request.config.selectedCustomer,
            selected_location=request.config.selectedLocation,
            algorithm=request.config.algorithm,
            interval=request.config.interval,
            historic_period=request.config.historicPeriod,
            forecast_period=request.config.forecastPeriod
        )
        
        db.add(config)
        db.commit()
        db.refresh(config)
        
        return {
            "message": "Configuration saved successfully",
            "id": config.id,
            "name": config.name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to save configuration")

@app.get("/configurations/{config_id}")
async def get_configuration(
    config_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific configuration"""
    try:
        config = db.query(ForecastConfiguration).filter(ForecastConfiguration.id == config_id).first()
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        forecast_config = {
            'forecastBy': config.forecast_by,
            'selectedItem': config.selected_item,
            'selectedProduct': config.selected_product,
            'selectedCustomer': config.selected_customer,
            'selectedLocation': config.selected_location,
            'algorithm': config.algorithm,
            'interval': config.interval,
            'historicPeriod': config.historic_period,
            'forecastPeriod': config.forecast_period
        }
        
        return {
            'id': config.id,
            'name': config.name,
            'description': config.description,
            'config': forecast_config,
            'createdAt': config.created_at.isoformat(),
            'updatedAt': config.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to get configuration")

@app.put("/configurations/{config_id}")
async def update_configuration(
    config_id: int,
    request: SaveConfigRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a forecast configuration"""
    try:
        config = db.query(ForecastConfiguration).filter(ForecastConfiguration.id == config_id).first()
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        # Check if new name conflicts with existing (excluding current)
        existing = db.query(ForecastConfiguration).filter(
            and_(ForecastConfiguration.name == request.name, ForecastConfiguration.id != config_id)
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Configuration name already exists")
        
        # Update configuration
        config.name = request.name
        config.description = request.description
        config.forecast_by = request.config.forecastBy
        config.selected_item = request.config.selectedItem
        config.selected_product = request.config.selectedProduct
        config.selected_customer = request.config.selectedCustomer
        config.selected_location = request.config.selectedLocation
        config.algorithm = request.config.algorithm
        config.interval = request.config.interval
        config.historic_period = request.config.historicPeriod
        config.forecast_period = request.config.forecastPeriod
        
        db.commit()
        
        return {
            "message": "Configuration updated successfully",
            "id": config.id,
            "name": config.name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")

@app.delete("/configurations/{config_id}")
async def delete_configuration(
    config_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a forecast configuration"""
    try:
        config = db.query(ForecastConfiguration).filter(ForecastConfiguration.id == config_id).first()
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        db.delete(config)
        db.commit()
        
        return {"message": "Configuration deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete configuration")

# Saved forecasts endpoints
@app.post("/saved_forecasts")
async def save_forecast(
    request: SaveForecastRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save a forecast result"""
    try:
        # Check if name already exists for this user
        existing = db.query(SavedForecastResult).filter(
            and_(
                SavedForecastResult.user_id == current_user.id,
                SavedForecastResult.name == request.name
            )
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail="Forecast name already exists")
        
        # Create saved forecast
        saved_forecast = SavedForecastResult(
            user_id=current_user.id,
            name=request.name,
            description=request.description,
            forecast_config=json.dumps(request.forecast_config.dict()),
            forecast_data=json.dumps(request.forecast_data)
        )
        
        db.add(saved_forecast)
        db.commit()
        db.refresh(saved_forecast)
        
        return {
            "id": saved_forecast.id,
            "user_id": saved_forecast.user_id,
            "name": saved_forecast.name,
            "description": saved_forecast.description,
            "forecast_config": json.loads(saved_forecast.forecast_config),
            "forecast_data": json.loads(saved_forecast.forecast_data),
            "created_at": saved_forecast.created_at.isoformat(),
            "updated_at": saved_forecast.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to save forecast")

@app.get("/saved_forecasts")
async def get_saved_forecasts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's saved forecasts"""
    try:
        saved_forecasts = db.query(SavedForecastResult).filter(
            SavedForecastResult.user_id == current_user.id
        ).order_by(desc(SavedForecastResult.created_at)).all()
        
        results = []
        for forecast in saved_forecasts:
            results.append({
                "id": forecast.id,
                "user_id": forecast.user_id,
                "name": forecast.name,
                "description": forecast.description,
                "forecast_config": json.loads(forecast.forecast_config),
                "forecast_data": json.loads(forecast.forecast_data),
                "created_at": forecast.created_at.isoformat(),
                "updated_at": forecast.updated_at.isoformat()
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting saved forecasts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get saved forecasts")

@app.delete("/saved_forecasts/{forecast_id}")
async def delete_saved_forecast(
    forecast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a saved forecast"""
    try:
        forecast = db.query(SavedForecastResult).filter(
            and_(
                SavedForecastResult.id == forecast_id,
                SavedForecastResult.user_id == current_user.id
            )
        ).first()
        
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found")
        
        db.delete(forecast)
        db.commit()
        
        return {"message": "Forecast deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting saved forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete forecast")

# Scheduled forecasts endpoints
@app.post("/scheduled_forecasts")
async def create_scheduled_forecast(
    request: ScheduledForecastCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a scheduled forecast"""
    try:
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00')) if request.end_date else None
        
        # Create scheduled forecast
        scheduled_forecast = ScheduledForecast(
            user_id=current_user.id,
            name=request.name,
            description=request.description,
            forecast_config=json.dumps(request.forecast_config.dict()),
            frequency=request.frequency,
            start_date=start_date,
            end_date=end_date,
            next_run=start_date,
            status='active'
        )
        
        db.add(scheduled_forecast)
        db.commit()
        db.refresh(scheduled_forecast)
        
        return {
            "id": scheduled_forecast.id,
            "user_id": scheduled_forecast.user_id,
            "name": scheduled_forecast.name,
            "description": scheduled_forecast.description,
            "forecast_config": json.loads(scheduled_forecast.forecast_config),
            "frequency": scheduled_forecast.frequency,
            "start_date": scheduled_forecast.start_date.isoformat(),
            "end_date": scheduled_forecast.end_date.isoformat() if scheduled_forecast.end_date else None,
            "next_run": scheduled_forecast.next_run.isoformat(),
            "last_run": scheduled_forecast.last_run.isoformat() if scheduled_forecast.last_run else None,
            "status": scheduled_forecast.status,
            "run_count": scheduled_forecast.run_count,
            "success_count": scheduled_forecast.success_count,
            "failure_count": scheduled_forecast.failure_count,
            "last_error": scheduled_forecast.last_error,
            "created_at": scheduled_forecast.created_at.isoformat(),
            "updated_at": scheduled_forecast.updated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating scheduled forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to create scheduled forecast")

@app.get("/scheduled_forecasts")
async def get_scheduled_forecasts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's scheduled forecasts"""
    try:
        scheduled_forecasts = db.query(ScheduledForecast).filter(
            ScheduledForecast.user_id == current_user.id
        ).order_by(desc(ScheduledForecast.created_at)).all()
        
        results = []
        for forecast in scheduled_forecasts:
            results.append({
                "id": forecast.id,
                "user_id": forecast.user_id,
                "name": forecast.name,
                "description": forecast.description,
                "forecast_config": json.loads(forecast.forecast_config),
                "frequency": forecast.frequency,
                "start_date": forecast.start_date.isoformat(),
                "end_date": forecast.end_date.isoformat() if forecast.end_date else None,
                "next_run": forecast.next_run.isoformat(),
                "last_run": forecast.last_run.isoformat() if forecast.last_run else None,
                "status": forecast.status,
                "run_count": forecast.run_count,
                "success_count": forecast.success_count,
                "failure_count": forecast.failure_count,
                "last_error": forecast.last_error,
                "created_at": forecast.created_at.isoformat(),
                "updated_at": forecast.updated_at.isoformat()
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting scheduled forecasts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scheduled forecasts")

@app.get("/scheduled_forecasts/{forecast_id}")
async def get_scheduled_forecast(
    forecast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific scheduled forecast"""
    try:
        forecast = db.query(ScheduledForecast).filter(
            and_(
                ScheduledForecast.id == forecast_id,
                ScheduledForecast.user_id == current_user.id
            )
        ).first()
        
        if not forecast:
            raise HTTPException(status_code=404, detail="Scheduled forecast not found")
        
        return {
            "id": forecast.id,
            "user_id": forecast.user_id,
            "name": forecast.name,
            "description": forecast.description,
            "forecast_config": json.loads(forecast.forecast_config),
            "frequency": forecast.frequency,
            "start_date": forecast.start_date.isoformat(),
            "end_date": forecast.end_date.isoformat() if forecast.end_date else None,
            "next_run": forecast.next_run.isoformat(),
            "last_run": forecast.last_run.isoformat() if forecast.last_run else None,
            "status": forecast.status,
            "run_count": forecast.run_count,
            "success_count": forecast.success_count,
            "failure_count": forecast.failure_count,
            "last_error": forecast.last_error,
            "created_at": forecast.created_at.isoformat(),
            "updated_at": forecast.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scheduled forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scheduled forecast")

@app.put("/scheduled_forecasts/{forecast_id}")
async def update_scheduled_forecast(
    forecast_id: int,
    request: ScheduledForecastUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a scheduled forecast"""
    try:
        forecast = db.query(ScheduledForecast).filter(
            and_(
                ScheduledForecast.id == forecast_id,
                ScheduledForecast.user_id == current_user.id
            )
        ).first()
        
        if not forecast:
            raise HTTPException(status_code=404, detail="Scheduled forecast not found")
        
        # Update fields
        if request.name is not None:
            forecast.name = request.name
        if request.description is not None:
            forecast.description = request.description
        if request.frequency is not None:
            forecast.frequency = request.frequency
        if request.start_date is not None:
            forecast.start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        if request.end_date is not None:
            forecast.end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        if request.status is not None:
            forecast.status = request.status
        
        db.commit()
        db.refresh(forecast)
        
        return {
            "id": forecast.id,
            "user_id": forecast.user_id,
            "name": forecast.name,
            "description": forecast.description,
            "forecast_config": json.loads(forecast.forecast_config),
            "frequency": forecast.frequency,
            "start_date": forecast.start_date.isoformat(),
            "end_date": forecast.end_date.isoformat() if forecast.end_date else None,
            "next_run": forecast.next_run.isoformat(),
            "last_run": forecast.last_run.isoformat() if forecast.last_run else None,
            "status": forecast.status,
            "run_count": forecast.run_count,
            "success_count": forecast.success_count,
            "failure_count": forecast.failure_count,
            "last_error": forecast.last_error,
            "created_at": forecast.created_at.isoformat(),
            "updated_at": forecast.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating scheduled forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to update scheduled forecast")

@app.delete("/scheduled_forecasts/{forecast_id}")
async def delete_scheduled_forecast(
    forecast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a scheduled forecast"""
    try:
        forecast = db.query(ScheduledForecast).filter(
            and_(
                ScheduledForecast.id == forecast_id,
                ScheduledForecast.user_id == current_user.id
            )
        ).first()
        
        if not forecast:
            raise HTTPException(status_code=404, detail="Scheduled forecast not found")
        
        db.delete(forecast)
        db.commit()
        
        return {"message": "Scheduled forecast deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting scheduled forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete scheduled forecast")

@app.get("/scheduled_forecasts/{forecast_id}/executions")
async def get_forecast_executions(
    forecast_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get execution history for a scheduled forecast"""
    try:
        # Verify forecast belongs to user
        forecast = db.query(ScheduledForecast).filter(
            and_(
                ScheduledForecast.id == forecast_id,
                ScheduledForecast.user_id == current_user.id
            )
        ).first()
        
        if not forecast:
            raise HTTPException(status_code=404, detail="Scheduled forecast not found")
        
        executions = db.query(ForecastExecution).filter(
            ForecastExecution.scheduled_forecast_id == forecast_id
        ).order_by(desc(ForecastExecution.execution_time)).all()
        
        results = []
        for execution in executions:
            result_summary = json.loads(execution.result_summary) if execution.result_summary else None
            
            results.append({
                "id": execution.id,
                "scheduled_forecast_id": execution.scheduled_forecast_id,
                "execution_time": execution.execution_time.isoformat(),
                "status": execution.status,
                "duration_seconds": execution.duration_seconds,
                "result_summary": result_summary,
                "error_message": execution.error_message,
                "created_at": execution.created_at.isoformat()
            })
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting forecast executions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get forecast executions")

@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status"""
    try:
        from scheduler import get_scheduler_status
        return get_scheduler_status()
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        return {"running": False, "check_interval": 60}

# Model cache endpoints
@app.get("/model_cache_info")
async def get_model_cache_info(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get model cache information"""
    try:
        from model_persistence import SavedModel
        
        models = db.query(SavedModel).order_by(desc(SavedModel.last_used)).all()
        
        cache_info = []
        for model in models:
            cache_info.append({
                "model_hash": model.model_hash,
                "algorithm": model.algorithm,
                "accuracy": model.accuracy,
                "created_at": model.created_at.isoformat(),
                "last_used": model.last_used.isoformat(),
                "use_count": model.use_count
            })
        
        return cache_info
        
    except Exception as e:
        logger.error(f"Error getting model cache info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model cache info")

@app.post("/clear_model_cache")
async def clear_model_cache(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Clear old cached models"""
    try:
        ModelPersistenceManager.cleanup_old_models(db, days_old=7, max_models=50)
        
        # Count remaining models
        from model_persistence import SavedModel
        remaining_count = db.query(SavedModel).count()
        
        return {
            "message": "Model cache cleaned successfully",
            "cleared_count": "Old models cleared",
            "remaining_count": remaining_count
        }
        
    except Exception as e:
        logger.error(f"Error clearing model cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear model cache")

@app.post("/clear_all_model_cache")
async def clear_all_model_cache(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Clear all cached models"""
    try:
        from model_persistence import SavedModel
        
        # Count models before deletion
        count_before = db.query(SavedModel).count()
        
        # Delete all models
        db.query(SavedModel).delete()
        db.commit()
        
        return {
            "message": "All cached models cleared successfully",
            "cleared_count": count_before
        }
        
    except Exception as e:
        logger.error(f"Error clearing all model cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear all model cache")

# Forecasting algorithms
ALGORITHMS = {
    'linear_regression': 'Linear Regression',
    'polynomial_regression': 'Polynomial Regression', 
    'exponential_smoothing': 'Exponential Smoothing',
    'holt_winters': 'Holt-Winters',
    'arima': 'ARIMA',
    'random_forest': 'Random Forest',
    'seasonal_decomposition': 'Seasonal Decomposition',
    'moving_average': 'Moving Average',
    'best_fit': 'Best Fit (Auto-select)'
}

@app.get("/algorithms")
async def get_algorithms():
    """Get available forecasting algorithms"""
    return {"algorithms": ALGORITHMS}

def generate_forecast_internal(config: ForecastConfig, db: Session) -> Dict[str, Any]:
    """Internal forecast generation function"""
    try:
        # Import forecasting modules
        from forecasting_algorithms import ForecastingEngine
        
        # Initialize forecasting engine
        engine = ForecastingEngine(db)
        
        # Generate forecast based on configuration
        if config.multiSelect or (config.selectedItems and len(config.selectedItems) > 1):
            # Multi-selection forecasting
            return engine.generate_multi_forecast(config)
        else:
            # Single forecasting
            return engine.generate_single_forecast(config)
            
    except Exception as e:
        logger.error(f"Forecast generation error: {e}")
        logger.error(traceback.format_exc())
        raise e

@app.post("/forecast")
async def generate_forecast(
    config: ForecastConfig,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate forecast based on configuration"""
    try:
        logger.info(f"Generating forecast for user {current_user.username}")
        logger.info(f"Config: {config.dict()}")
        
        # Generate forecast
        result = generate_forecast_internal(config, db)
        
        # Auto-save the forecast result
        try:
            # Generate auto-save name
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            if config.multiSelect:
                auto_name = f"Auto-saved Multi-Forecast - {timestamp}"
            elif config.selectedItems and len(config.selectedItems) > 1:
                auto_name = f"Auto-saved {len(config.selectedItems)} {config.forecastBy}s - {timestamp}"
            else:
                item_name = config.selectedItem or f"{config.selectedProduct}-{config.selectedCustomer}-{config.selectedLocation}"
                auto_name = f"Auto-saved {item_name} - {timestamp}"
            
            # Save forecast
            saved_forecast = SavedForecastResult(
                user_id=current_user.id,
                name=auto_name,
                description="Automatically saved forecast result",
                forecast_config=json.dumps(config.dict()),
                forecast_data=json.dumps(result)
            )
            
            db.add(saved_forecast)
            db.commit()
            logger.info(f"Auto-saved forecast with name: {auto_name}")
            
        except Exception as save_error:
            logger.warning(f"Failed to auto-save forecast: {save_error}")
            # Don't fail the main request if auto-save fails
        
        return result
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

# Excel download endpoints
@app.post("/download_forecast_excel")
async def download_forecast_excel(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Download forecast results as Excel file"""
    try:
        import io
        import xlsxwriter
        from datetime import datetime
        
        forecast_result = request.get('forecastResult')
        forecast_by = request.get('forecastBy', 'Unknown')
        selected_item = request.get('selectedItem', 'Unknown')
        
        if not forecast_result:
            raise HTTPException(status_code=400, detail="No forecast result provided")
        
        # Create Excel file in memory
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        data_format = workbook.add_format({'border': 1})
        number_format = workbook.add_format({'border': 1, 'num_format': '#,##0.00'})
        
        # Summary worksheet
        summary_ws = workbook.add_worksheet('Summary')
        summary_ws.write('A1', 'Forecast Summary', header_format)
        summary_ws.write('A3', 'Item:', header_format)
        summary_ws.write('B3', selected_item, data_format)
        summary_ws.write('A4', 'Algorithm:', header_format)
        summary_ws.write('B4', forecast_result.get('selectedAlgorithm', 'Unknown'), data_format)
        summary_ws.write('A5', 'Accuracy:', header_format)
        summary_ws.write('B5', f"{forecast_result.get('accuracy', 0):.2f}%", data_format)
        summary_ws.write('A6', 'MAE:', header_format)
        summary_ws.write('B6', forecast_result.get('mae', 0), number_format)
        summary_ws.write('A7', 'RMSE:', header_format)
        summary_ws.write('B7', forecast_result.get('rmse', 0), number_format)
        summary_ws.write('A8', 'Trend:', header_format)
        summary_ws.write('B8', forecast_result.get('trend', 'Unknown'), data_format)
        
        # Historical data worksheet
        if 'historicData' in forecast_result:
            hist_ws = workbook.add_worksheet('Historical Data')
            hist_ws.write('A1', 'Period', header_format)
            hist_ws.write('B1', 'Quantity', header_format)
            
            for i, data_point in enumerate(forecast_result['historicData'], 2):
                hist_ws.write(f'A{i}', data_point.get('period', ''), data_format)
                hist_ws.write(f'B{i}', data_point.get('quantity', 0), number_format)
        
        # Forecast data worksheet
        if 'forecastData' in forecast_result:
            forecast_ws = workbook.add_worksheet('Forecast Data')
            forecast_ws.write('A1', 'Period', header_format)
            forecast_ws.write('B1', 'Quantity', header_format)
            
            for i, data_point in enumerate(forecast_result['forecastData'], 2):
                forecast_ws.write(f'A{i}', data_point.get('period', ''), data_format)
                forecast_ws.write(f'B{i}', data_point.get('quantity', 0), number_format)
        
        # Algorithm comparison worksheet
        if 'allAlgorithms' in forecast_result and forecast_result['allAlgorithms']:
            algo_ws = workbook.add_worksheet('Algorithm Comparison')
            algo_ws.write('A1', 'Algorithm', header_format)
            algo_ws.write('B1', 'Accuracy (%)', header_format)
            algo_ws.write('C1', 'MAE', header_format)
            algo_ws.write('D1', 'RMSE', header_format)
            algo_ws.write('E1', 'Trend', header_format)
            
            for i, algo in enumerate(forecast_result['allAlgorithms'], 2):
                algo_ws.write(f'A{i}', algo.get('algorithm', ''), data_format)
                algo_ws.write(f'B{i}', algo.get('accuracy', 0), number_format)
                algo_ws.write(f'C{i}', algo.get('mae', 0), number_format)
                algo_ws.write(f'D{i}', algo.get('rmse', 0), number_format)
                algo_ws.write(f'E{i}', algo.get('trend', ''), data_format)
        
        workbook.close()
        output.seek(0)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_{selected_item}_{timestamp}.xlsx"
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Excel download error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate Excel file: {str(e)}")

@app.post("/download_multi_forecast_excel")
async def download_multi_forecast_excel(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Download multi-forecast results as Excel file"""
    try:
        import io
        import xlsxwriter
        from datetime import datetime
        
        multi_forecast_result = request.get('multiForecastResult')
        
        if not multi_forecast_result:
            raise HTTPException(status_code=400, detail="No multi-forecast result provided")
        
        # Create Excel file in memory
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        data_format = workbook.add_format({'border': 1})
        number_format = workbook.add_format({'border': 1, 'num_format': '#,##0.00'})
        
        # Summary worksheet
        summary_ws = workbook.add_worksheet('Summary')
        summary_ws.write('A1', 'Multi-Forecast Summary', header_format)
        summary_ws.write('A3', 'Total Combinations:', header_format)
        summary_ws.write('B3', multi_forecast_result.get('totalCombinations', 0), data_format)
        summary_ws.write('A4', 'Successful:', header_format)
        summary_ws.write('B4', multi_forecast_result.get('summary', {}).get('successfulCombinations', 0), data_format)
        summary_ws.write('A5', 'Failed:', header_format)
        summary_ws.write('B5', multi_forecast_result.get('summary', {}).get('failedCombinations', 0), data_format)
        summary_ws.write('A6', 'Average Accuracy:', header_format)
        summary_ws.write('B6', f"{multi_forecast_result.get('summary', {}).get('averageAccuracy', 0):.2f}%", data_format)
        
        # Results overview worksheet
        results_ws = workbook.add_worksheet('Results Overview')
        results_ws.write('A1', 'Product', header_format)
        results_ws.write('B1', 'Customer', header_format)
        results_ws.write('C1', 'Location', header_format)
        results_ws.write('D1', 'Algorithm', header_format)
        results_ws.write('E1', 'Accuracy (%)', header_format)
        results_ws.write('F1', 'MAE', header_format)
        results_ws.write('G1', 'RMSE', header_format)
        results_ws.write('H1', 'Trend', header_format)
        
        for i, result in enumerate(multi_forecast_result.get('results', []), 2):
            combination = result.get('combination', {})
            results_ws.write(f'A{i}', combination.get('product', ''), data_format)
            results_ws.write(f'B{i}', combination.get('customer', ''), data_format)
            results_ws.write(f'C{i}', combination.get('location', ''), data_format)
            results_ws.write(f'D{i}', result.get('selectedAlgorithm', ''), data_format)
            results_ws.write(f'E{i}', result.get('accuracy', 0), number_format)
            results_ws.write(f'F{i}', result.get('mae', 0), number_format)
            results_ws.write(f'G{i}', result.get('rmse', 0), number_format)
            results_ws.write(f'H{i}', result.get('trend', ''), data_format)
        
        workbook.close()
        output.seek(0)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_forecast_{timestamp}.xlsx"
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Multi-forecast Excel download error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate Excel file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)