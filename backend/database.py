#!/usr/bin/env python3
"""
Database configuration and models for PostgreSQL integration

"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, UniqueConstraint, Boolean, DECIMAL, Text # type: ignore
from sqlalchemy.ext.declarative import declarative_base # type: ignore
from sqlalchemy.orm import sessionmaker # type: ignore
from sqlalchemy import ForeignKey # type: ignore
from sqlalchemy.orm import relationship # type: ignore
from passlib.context import CryptContext # type: ignore
import os
from datetime import datetime

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Make sure which credantals are used to connect locally or in cloud environment
DB_HOST = os.getenv("DB_HOST", "localhost") #172.17.0.1
DB_PORT = os.getenv("DB_PORT", "5432")  #5432
DB_USER = os.getenv("DB_USER", "postgres")  # Changed from root to postgres
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_NAME = os.getenv("DB_NAME", "forecasting_db")

# Database configuration
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# Create engine with better error handling
try:
    engine = create_engine(DATABASE_URL, echo=False)
except Exception as e:
    print(f"❌ Error connecting to database: {e}")
    print("Please ensure:")
    print("1. PostgreSQL server is running")
    print("2. Database 'forecasting_db' exists")
    print("3. Connection credentials are correct")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ExternalFactorData(Base):
    __tablename__ = 'external_factor_data'
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    factor_name = Column(String(255), index=True)
    factor_value = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('date', 'factor_name', name='unique_external_factor_record'),
    )

class User(Base):
    """Model for user authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    is_approved = Column(Boolean, default=False)  # New: requires admin approval
    is_admin = Column(Boolean, default=False)      # New: admin flag
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)
    
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)
    
class ForecastSelectionKey(Base):
    """Model for storing unique forecast selection combinations"""
    __tablename__ = "forecast_selection_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    product = Column(String(255), nullable=True)
    customer = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint to ensure each combination is stored only once
    __table_args__ = (
        UniqueConstraint('product', 'customer', 'location', name='unique_selection_combination'),
    )
    
    def get_display_name(self) -> str:
        """Get a human-readable display name for this selection"""
        parts = []
        if self.product:
            parts.append(f"Product: {self.product}")
        if self.customer:
            parts.append(f"Customer: {self.customer}")
        if self.location:
            parts.append(f"Location: {self.location}")
        return " | ".join(parts) if parts else "No selection"
    
    def get_combination_dict(self) -> dict:
        """Get combination as dictionary for API responses"""
        return {
            'product': self.product,
            'customer': self.customer,
            'location': self.location
        }
class ForecastData(Base):
    """Model for storing forecast data from Excel uploads"""
    __tablename__ = "forecast_data"
    
    id = Column(Integer, primary_key=True, index=True)
    product = Column(String(255), nullable=True)
    quantity = Column(DECIMAL(15, 2), nullable=False)
    product_group = Column(String(255), nullable=True)
    product_hierarchy = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    location_region = Column(String(255), nullable=True)
    customer = Column(String(255), nullable=True)
    customer_group = Column(String(255), nullable=True)
    customer_region = Column(String(255), nullable=True)
    ship_to_party = Column(String(255), nullable=True)
    sold_to_party = Column(String(255), nullable=True)
    uom = Column(String(50), nullable=True)
    date = Column(Date, nullable=False)
    unit_price = Column(DECIMAL(15, 2), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint to prevent duplicates
    __table_args__ = (
        UniqueConstraint(
            'product', 'customer', 'location', 'date',
            name='unique_forecast_record'
        ),
    )

class SavedForecastResult(Base):
    """Model for storing user's saved forecast results"""
    __tablename__ = "saved_forecast_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)  # Foreign key to users table
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    forecast_config = Column(Text, nullable=False)  # JSON string of ForecastConfig
    forecast_data = Column(Text, nullable=False)  # JSON string of ForecastResult
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint for user + name combination
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='unique_user_forecast_name'),
    )

class ForecastConfiguration(Base):
    """Model for storing forecast configurations"""
    __tablename__ = "forecast_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    forecast_by = Column(String(50), nullable=False)
    selection_key_id = Column(Integer, ForeignKey('forecast_selection_keys.id'), nullable=True)
    algorithm = Column(String(100), nullable=False, default='best_fit')
    interval = Column(String(20), nullable=False, default='month')
    historic_period = Column(Integer, nullable=False, default=12)
    forecast_period = Column(Integer, nullable=False, default=6)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to ForecastSelectionKey
    selection_key = relationship("ForecastSelectionKey")
    
    # Unique constraint for configuration name
    __table_args__ = (
        UniqueConstraint('name', name='unique_config_name'),
    )

def create_default_user():
    """Create default admin user if no users exist"""
    try:
        db = SessionLocal()
        user_count = db.query(User).count()
        if user_count == 0:
            default_user = User(
                username="admin",
                email="admin@forecasting.com",
                hashed_password=User.hash_password("admin123"),
                full_name="System Administrator",
                is_active=True,
                is_approved=True,
                is_admin=True
            )
            db.add(default_user)
            db.commit()
            print("✅ Default admin user created (username: admin, password: admin123)")
        db.close()
    except Exception as e:
        print(f"⚠️  Error creating default user: {e}")

# Import model persistence tables
from model_persistence import SavedModel, ModelAccuracyHistory

def create_tables():
    """Create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        # Also create model persistence tables
        SavedModel.metadata.create_all(bind=engine)
        ModelAccuracyHistory.metadata.create_all(bind=engine)
        print("✅ Database tables verified/created successfully!")
        print("✅ Saved forecast results table created!")
        create_default_user()
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables (assumes database already exists)"""
    try:
        # Test connection first
        with engine.connect() as connection:
            print("✅ Database connection successful!")
        
        # Create tables if they don't exist
        return create_tables()
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        print("Please ensure:")
        print("1. PostgreSQL server is running")
        print("2. Database 'forecasting_db' exists")
        print("3. User has proper permissions")
        return False