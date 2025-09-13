#!/usr/bin/env python3
"""
Advanced Multi-variant Forecasting API with PostgreSQL

This FastAPI application provides comprehensive forecasting capabilities with:
- 8 different forecasting algorithms
- Best fit mode for automatic algorithm selection
- PostgreSQL database integration
- Multi-selection forecasting
- External factor integration
- User authentication and authorization
- Model caching for performance
- Scheduled forecasting
"""

import os
import sys
import logging
import traceback
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.holtwinters import ExponentialSmoothing # type: ignore
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import warnings
from sqlalchemy import func, distinct, and_, or_ # type: ignore
import requests

# Import database and authentication modules
from database import (
    get_db, init_database, ForecastData, ExternalFactorData, 
    ForecastConfiguration, SavedForecastResult, User, ForecastSelectionKey
)
from auth import get_current_user, get_current_user_optional, create_access_token
from model_persistence import ModelPersistenceManager, SavedModel, ModelAccuracyHistory
from validation import DateRangeValidator

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Multi-variant Forecasting API",
    description="Comprehensive forecasting system with multiple algorithms and PostgreSQL integration",
    version="2.0.0"
)

# FRED API Configuration
FRED_API_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.getenv("FRED_API_KEY", "82a8e6191d71f41b22cf33bf73f7a0c2")  # Set this environment variable

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
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

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class ForecastSelectionKeyRequest(BaseModel):
    product: Optional[str] = None
    customer: Optional[str] = None
    location: Optional[str] = None

class ForecastSelectionKeyResponse(BaseModel):
    id: int
    product: Optional[str]
    customer: Optional[str]
    location: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ResolveSelectionKeyResponse(BaseModel):
    selection_key_id: int
    selection_key: ForecastSelectionKeyResponse
    created: bool

class ForecastConfig(BaseModel):
    forecastBy: str
    selectedItem: Optional[str] = None
    selectedProduct: Optional[str] = None  # Keep for backward compatibility
    selectedCustomer: Optional[str] = None  # Keep for backward compatibility
    selectedLocation: Optional[str] = None  # Keep for backward compatibility
    selectedProducts: Optional[List[str]] = None  # New multi-select fields
    selectedCustomers: Optional[List[str]] = None
    selectedLocations: Optional[List[str]] = None
    selectedItems: Optional[List[str]] = None  # For simple mode multi-select
    algorithm: str = "linear_regression"
    interval: str = "month"
    historicPeriod: int = 12
    forecastPeriod: int = 6
    multiSelect: bool = False  # Flag to indicate multi-selection mode
    advancedMode: bool = False  # Flag to indicate advanced mode (precise combinations)
    externalFactors: Optional[List[str]] = None

class ForecastConfigRequest(BaseModel):
    forecastBy: str
    selectionKeyId: Optional[int] = None
    selectedProducts: Optional[List[str]] = []
    selectedCustomers: Optional[List[str]] = []
    selectedLocations: Optional[List[str]] = []
    selectedItems: Optional[List[str]] = []
    algorithm: str = 'best_fit'
    interval: str = 'month'
    historicPeriod: int = 12
    forecastPeriod: int = 6
    multiSelect: Optional[bool] = False
    advancedMode: Optional[bool] = False
    externalFactors: Optional[List[str]] = []

class DataPoint(BaseModel):
    date: str
    quantity: float
    period: str

class AlgorithmResult(BaseModel):
    algorithm: str
    accuracy: float
    mae: float
    rmse: float
    historicData: List[DataPoint]
    forecastData: List[DataPoint]
    trend: str

class ForecastResult(BaseModel):
    combination: Optional[Dict[str, str]] = None  # Track which combination this result is for
    selectedAlgorithm: str
    accuracy: float
    mae: float
    rmse: float
    historicData: List[DataPoint]
    forecastData: List[DataPoint]
    trend: str
    allAlgorithms: Optional[List[AlgorithmResult]] = None
    processLog: Optional[List[str]] = None
    configHash: Optional[str] = None # For caching

class MultiForecastResult(BaseModel):
    results: List[ForecastResult]
    totalCombinations: int
    summary: Dict[str, Any]
    processLog: Optional[List[str]] = None

class SaveConfigRequest(BaseModel):
    name: str
    description: Optional[str] = None
    config: ForecastConfig

class ConfigurationResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    config: ForecastConfig
    createdAt: str
    updatedAt: str

class AdminSetActiveRequest(BaseModel):
    is_active: bool

class SaveForecastRequest(BaseModel):
    name: str
    description: Optional[str] = None
    forecast_config: ForecastConfig
    forecast_data: Dict[str, Any]

class SavedForecastRequest(BaseModel):
    name: str
    description: Optional[str] = None
    forecast_config: ForecastConfig
    forecast_data: Union[ForecastResult, MultiForecastResult]

class SavedForecastResponse(BaseModel):
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    forecast_config: ForecastConfig
    forecast_data: Union[ForecastResult, MultiForecastResult]
    created_at: str
    updated_at: str

class ScheduledForecastCreate(BaseModel):
    name: str
    description: Optional[str] = None
    forecast_config: ForecastConfigRequest
    frequency: str  # 'daily', 'weekly', 'monthly'
    start_date: datetime
    end_date: Optional[datetime] = None

class ScheduledForecastUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: Optional[str] = None

class DataViewRequest(BaseModel):
    product: Optional[str] = None
    customer: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    page: Optional[int] = 1
    page_size: Optional[int] = 50

class DataViewResponse(BaseModel):
    data: List[Dict[str, Any]]
    total_records: int
    page: int
    page_size: int
    total_pages: int

class FredDataRequest(BaseModel):
    series_ids: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class FredDataResponse(BaseModel):
    message: str
    inserted: int
    duplicates: int
    series_processed: int
    series_details: List[Dict[str, Any]]

class DatabaseStats(BaseModel):
    totalRecords: int
    dateRange: Dict[str, str]
    uniqueProducts: int
    uniqueCustomers: int
    uniqueLocations: int

class ForecastingEngine:
    """Advanced forecasting engine with multiple algorithms"""

    ALGORITHMS = {
        "linear_regression": "Linear Regression",
        "polynomial_regression": "Polynomial Regression",
        "exponential_smoothing": "Exponential Smoothing",
        "holt_winters": "Holt-Winters",
        "arima": "ARIMA (Simple)",
        "random_forest": "Random Forest",
        "seasonal_decomposition": "Seasonal Decomposition",
        "moving_average": "Moving Average",
        "sarima": "SARIMA (Seasonal ARIMA)",
        "prophet_like": "Prophet-like Forecasting",
        "lstm_like": "Simple LSTM-like",
        "xgboost": "XGBoost Regression",
        "svr": "Support Vector Regression",
        "knn": "K-Nearest Neighbors",
        "gaussian_process": "Gaussian Process",
        "neural_network": "Neural Network (MLP)",
        "theta_method": "Theta Method",
        "croston": "Croston's Method",
        "ses": "Simple Exponential Smoothing",
        "damped_trend": "Damped Trend Method",
        "naive_seasonal": "Naive Seasonal",
        "drift_method": "Drift Method",
        "best_fit": "Best Fit (Auto-Select)",
        "best_statistical": "Best Statistical Method",
        "best_ml": "Best Machine Learning Method",
        "best_specialized": "Best Specialized Method"
    }

    @staticmethod
    def load_data_from_db(db: Session, config: ForecastConfig) -> pd.DataFrame:
        """Load data from PostgreSQL database based on forecast configuration"""
        query = db.query(ForecastData)

        # Filter by forecastBy and selected items
        if config.forecastBy == 'product':
            if config.multiSelect and config.selectedProducts:
                query = query.filter(ForecastData.product.in_(config.selectedProducts))
            elif config.selectedProduct:
                query = query.filter(ForecastData.product == config.selectedProduct)
            elif config.selectedItem:
                query = query.filter(ForecastData.product == config.selectedItem)
        elif config.forecastBy == 'customer':
            if config.multiSelect and config.selectedCustomers:
                query = query.filter(ForecastData.customer.in_(config.selectedCustomers))
            elif config.selectedCustomer:
                query = query.filter(ForecastData.customer == config.selectedCustomer)
            elif config.selectedItem:
                query = query.filter(ForecastData.customer == config.selectedItem)
        elif config.forecastBy == 'location':
            if config.multiSelect and config.selectedLocations:
                query = query.filter(ForecastData.location.in_(config.selectedLocations))
            elif config.selectedLocation:
                query = query.filter(ForecastData.location == config.selectedLocation)
            elif config.selectedItem:
                query = query.filter(ForecastData.location == config.selectedItem)

        results = query.all()

        if not results:
            raise ValueError("No data found for the selected configuration")

        # Convert to DataFrame
        data = []
        for record in results:
            data.append({
                'date': record.date,
                'quantity': float(record.quantity),
                'product': record.product,
                'customer': record.customer,
                'location': record.location
            })

        df = pd.DataFrame(data)

        # Load and merge external factors if specified
        if config.externalFactors and len(config.externalFactors) > 0:
            df = ForecastingEngine.merge_external_factors(db, df, config.externalFactors)

        return df

    @staticmethod
    def merge_external_factors(db: Session, main_df: pd.DataFrame, external_factors: List[str]) -> pd.DataFrame:
        """Merge external factor data with main forecast data"""
        try:
            # Get date range from main data
            min_date = main_df['date'].min()
            max_date = main_df['date'].max()

            # Query external factor data
            external_query = db.query(ExternalFactorData).filter(
                ExternalFactorData.factor_name.in_(external_factors),
                ExternalFactorData.date >= min_date,
                ExternalFactorData.date <= max_date
            )

            external_results = external_query.all()
            print(f"External factor query results: {external_results}")

            if not external_results:
                print(f"No external factor data found for factors: {external_factors}")
                return main_df

            # Convert external factor data to DataFrame
            external_data = []
            for record in external_results:
                external_data.append({
                    'date': record.date,
                    'factor_name': record.factor_name,
                    'factor_value': float(record.factor_value)
                })

            external_df = pd.DataFrame(external_data)

            # Pivot external factors so each factor becomes a column
            external_pivot = external_df.pivot(index='date', columns='factor_name', values='factor_value')
            external_pivot.reset_index(inplace=True)

            # Ensure date columns are the same type
            main_df['date'] = pd.to_datetime(main_df['date']).dt.date
            external_pivot['date'] = pd.to_datetime(external_pivot['date']).dt.date

            # Merge with main data
            merged_df = pd.merge(main_df, external_pivot, on='date', how='left')
            print(f"Merged DataFrame columns: {merged_df.columns.tolist()}")

            # Forward fill missing external factor values
            for factor in external_factors:
                if factor in merged_df.columns:
                    merged_df[factor] = merged_df[factor].fillna(method='ffill').fillna(method='bfill')
            print(merged_df.head())
            print(f"Successfully merged external factors: {external_factors}")
            print(f"Merged data shape: {merged_df.shape}")

            return merged_df

        except Exception as e:
            print(f"Error merging external factors: {e}")
            return main_df

    @staticmethod
    def time_based_split(data: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
        """Proper time-based train/test split for time series"""
        n = len(data)
        if n < 6:
            return data.copy(), None

        split_idx = int(n * (1 - test_ratio))
        train = data.iloc[:split_idx].copy()
        test = data.iloc[split_idx:].copy()

        print(f"Time-based split: Train={len(train)} records, Test={len(test)} records")
        print(f"Train period: {train['date'].min()} to {train['date'].max()}")
        print(f"Test period: {test['date'].min()} to {test['date'].max()}")

        return train, test

    @staticmethod
    def forecast_external_factors(data: pd.DataFrame, external_factor_cols: List[str], periods: int) -> Dict[str, np.ndarray]:
        """Forecast external factors using simple trend analysis

        WARNING: This is a simplified approach for external factor forecasting.
        In production, external factors should be provided by the user or forecasted
        using specialized models. Current approach may lead to reduced accuracy.
        """
        future_factors = {}

        if external_factor_cols:
            print("⚠️  WARNING: External factors are being forecasted using simple trend analysis.")
            print("   For better accuracy, consider providing future external factor values.")
            print("   Current forecasting method may reduce model performance.")

        for col in external_factor_cols:
            if col not in data.columns:
                continue

            values = data[col].dropna().values
            if len(values) < 2:
                # Use last known value if insufficient data
                future_factors[col] = np.full(periods, values[-1] if len(values) > 0 else 0)
                print(f"External factor '{col}': Using last known value (insufficient data)")
                continue

            try:
                # Simple linear trend forecasting
                x = np.arange(len(values))

                # Fit linear trend
                slope, intercept = np.polyfit(x, values, 1)

                # Generate future values
                future_x = np.arange(len(values), len(values) + periods)
                future_values = slope * future_x + intercept

                # Add some uncertainty by using trend confidence
                trend_strength = abs(slope) / (np.std(values) + 1e-8)
                if trend_strength < 0.1:  # Weak trend, use last value
                    future_values = np.full(periods, values[-1])
                    print(f"External factor '{col}': Using last value (weak trend)")
                else:
                    print(f"External factor '{col}': Using linear trend (slope={slope:.4f})")

                future_factors[col] = future_values

            except Exception as e:
                # Fallback to last known value
                future_factors[col] = np.full(periods, values[-1])
                print(f"External factor '{col}': Fallback to last value due to error: {e}")

        return future_factors

    @staticmethod
    def aggregate_by_period(df: pd.DataFrame, interval: str, config: ForecastConfig = None) -> pd.DataFrame:
        """Aggregate data by time period"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Identify external factor columns
        external_factor_cols = [col for col in df.columns if col not in ['quantity', 'product', 'customer', 'location']]
        print(f"External factor columns found in aggregate_by_period: {external_factor_cols}")

        # Determine grouping columns based on configuration
        group_cols = ['date']
        if config and config.multiSelect:
            # For multi-select mode, determine which dimensions to group by
            selected_dimensions = []
            if config.selectedProducts and len(config.selectedProducts) > 0:
                selected_dimensions.append('product')
            if config.selectedCustomers and len(config.selectedCustomers) > 0:
                selected_dimensions.append('customer')
            if config.selectedLocations and len(config.selectedLocations) > 0:
                selected_dimensions.append('location')

            # If 2 or more dimensions are selected, group by all selected dimensions plus date
            if len(selected_dimensions) >= 2:
                group_cols = selected_dimensions + ['date']
        elif config and config.selectedItems and len(config.selectedItems) > 1:
            # Simple mode multi-select - group by the selected dimension plus date
            group_cols = [config.forecastBy, 'date']

        # Aggregate by period
        df_reset = df.reset_index()
        if interval == 'week':
            df_reset['period_group'] = df_reset['date'].dt.to_period('W-MON')
        elif interval == 'month':
            df_reset['period_group'] = df_reset['date'].dt.to_period('M')
        elif interval == 'year':
            df_reset['period_group'] = df_reset['date'].dt.to_period('Y')
        else:
            df_reset['period_group'] = df_reset['date'].dt.to_period('M')

        # Group by period and selected dimensions
        if len(group_cols) > 1:
            # Multi-dimensional grouping
            group_cols_with_period = [col for col in group_cols if col != 'date'] + ['period_group']

            # Aggregate quantity (sum) and external factors (mean)
            agg_dict = {'quantity': 'sum'}
            for col in external_factor_cols:
                if col in df_reset.columns:
                    agg_dict[col] = 'mean'  # Use mean for external factors

            aggregated = df_reset.groupby(group_cols_with_period).agg(agg_dict).reset_index()

            # Convert period back to timestamp for the first date of each period
            aggregated['date'] = aggregated['period_group'].dt.start_time
        else:
            # Single dimension grouping (original behavior)
            agg_dict = {'quantity': 'sum'}
            for col in external_factor_cols:
                if col in df_reset.columns:
                    agg_dict[col] = 'mean'  # Use mean for external factors

            aggregated = df_reset.groupby('period_group').agg(agg_dict).reset_index()
            aggregated['date'] = aggregated['period_group'].dt.start_time

        # Add period labels
        aggregated['period'] = aggregated['date'].apply(
            lambda x: ForecastingEngine.format_period(pd.Timestamp(x), interval)
        )

        # Clean up
        aggregated = aggregated.drop('period_group', axis=1)

        print(f"Aggregated DataFrame columns: {aggregated.columns.tolist()}")
        print(f"Sample aggregated data with external factors:")
        print(aggregated.head())

        return aggregated.reset_index(drop=True)

    @staticmethod
    def format_period(date: pd.Timestamp, interval: str) -> str:
        """Format period for display"""
        if interval == 'week':
            return f"Week of {date.strftime('%b %d, %Y')}"
        elif interval == 'month':
            return date.strftime('%b %Y')
        elif interval == 'year':
            return date.strftime('%Y')
        else:
            return date.strftime('%b %Y')

    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        # Calculate accuracy as percentage
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        accuracy = max(0, 100 - mape)

        return {
            'accuracy': min(accuracy, 99.9),
            'mae': mae,
            'rmse': rmse
        }

    @staticmethod
    def calculate_trend(data: np.ndarray) -> str:
        """Calculate trend direction"""
        if len(data) < 2:
            return 'stable'

        # Linear regression to find trend
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)

        threshold = np.mean(data) * 0.02  # 2% threshold

        if slope > threshold:
            return 'increasing'
        elif slope < -threshold:
            return 'decreasing'
        else:
            return 'stable'

    @staticmethod
    def linear_regression_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Linear regression forecasting with feature engineering"""
        y = data['quantity'].values
        n = len(y)

        # Feature engineering: create lag features and time index
        window = min(5, n - 1)

        # Get external factor columns
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        print(f"External factor columns: {external_factor_cols}")
        if window < 1:
            # Not enough data for feature engineering, fallback
            x = np.arange(n).reshape(-1, 1)
            if external_factor_cols:
                x = np.hstack([x, data[external_factor_cols].values])
            model = LinearRegression()
            model.fit(x, y)
            future_x = np.arange(n, n + periods).reshape(-1, 1)

            if external_factor_cols:
                # For simplicity, assume external factors remain at their last known value
                last_factors = data[external_factor_cols].iloc[-1].values
                future_factors = np.tile(last_factors, (periods, 1))
                future_x = np.hstack([future_x, future_factors])
            forecast = model.predict(future_x)
            forecast = np.maximum(forecast, 0)
            predicted = model.predict(x)
            metrics = ForecastingEngine.calculate_metrics(y, predicted)
            return forecast, metrics

        X = []
        y_target = []
        for i in range(window, n):
            lags = y[i-window:i]
            time_idx = i
            features = list(lags) + [time_idx]
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[i].values)
            X.append(features)
            y_target.append(y[i])

        X = np.array(X)
        y_target = np.array(y_target)

        # Debug print feature engineered data
        print(f"\nFeature engineered data for linear_regression:")
        print("Features (X) first 5 rows:")
        print(X[:5])
        print("Targets (y) first 5 values:")
        print(y_target[:5])

        model = LinearRegression()
        model.fit(X, y_target)

        # Forecast
        forecast = []
        recent_lags = list(y[-window:])
        for i in range(periods):
            features = recent_lags + [n + i]
            if external_factor_cols:
                # For simplicity, assume external factors remain at their last known value
                last_factors = data[external_factor_cols].iloc[-1].values
                features.extend(last_factors)

            pred = model.predict([features])[0]
            pred = max(0, pred)
            forecast.append(pred)
            recent_lags = recent_lags[1:] + [pred]

        forecast = np.array(forecast)

        # Calculate metrics on training data
        predicted = model.predict(X)
        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)

        return forecast, metrics

    @staticmethod
    def exponential_smoothing_forecast(data: pd.DataFrame, periods: int, alphas: list = [0.1,0.3,0.5]) -> tuple:
        """Enhanced exponential smoothing with external factors integration"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if n < 3:
            return np.full(periods, y[-1] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': np.std(y), 'rmse': np.std(y)}

        best_metrics = None
        best_forecast = None

        for alpha in alphas:
            print(f"Running Exponential Smoothing with alpha={alpha}")

            if external_factor_cols:
                # Use regression-based exponential smoothing with external factors
                window = min(5, n - 1)
                X, y_target = [], []

                for i in range(window, n):
                    # Exponentially weighted historical values
                    weights = np.array([alpha * (1 - alpha) ** j for j in range(window)])
                    weights = weights / weights.sum()
                    weighted_history = np.sum(weights * y[i-window:i])

                    features = [weighted_history, i]  # Smoothed value + trend
                    if external_factor_cols:
                        features.extend(data[external_factor_cols].iloc[i].values)

                    X.append(features)
                    y_target.append(y[i])

                if len(X) > 1:
                    X = np.array(X)
                    y_target = np.array(y_target)

                    # Fit linear model with smoothed features
                    model = LinearRegression()
                    model.fit(X, y_target)

                    # Forecast with external factors
                    forecast = []
                    last_values = y[-window:]

                    for i in range(periods):
                        weights = np.array([alpha * (1 - alpha) ** j for j in range(len(last_values))])
                        weights = weights / weights.sum()
                        weighted_history = np.sum(weights * last_values)

                        features = [weighted_history, n + i]
                        if external_factor_cols:
                            # Use last known external factor values
                            features.extend(data[external_factor_cols].iloc[-1].values)

                        pred = model.predict([features])[0]
                        pred = max(0, pred)
                        forecast.append(pred)

                        # Update last_values for next prediction
                        last_values = np.append(last_values[1:], pred)

                    predicted = model.predict(X)
                    metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
                else:
                    # Fallback to simple smoothing
                    smoothed = pd.Series(y).ewm(alpha=alpha).mean().values
                    forecast = np.full(periods, smoothed[-1])
                    metrics = ForecastingEngine.calculate_metrics(y[1:], smoothed[1:])
            else:
                # Traditional exponential smoothing without external factors
                smoothed = pd.Series(y).ewm(alpha=alpha).mean().values
                forecast = np.full(periods, smoothed[-1])
                metrics = ForecastingEngine.calculate_metrics(y[1:], smoothed[1:])

            print(f"Alpha={alpha}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, Accuracy={metrics['accuracy']:.2f}")

            if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                best_metrics = metrics
                best_forecast = forecast

        return np.array(best_forecast), best_metrics

    @staticmethod
    def generate_forecast_dates(last_date: pd.Timestamp, periods: int, interval: str) -> List[pd.Timestamp]:
        """Generate future dates for forecast"""
        dates = []
        current_date = last_date

        for i in range(periods):
            if interval == 'week':
                current_date = current_date + timedelta(weeks=1)
            elif interval == 'month':
                current_date = current_date + pd.DateOffset(months=1)
            elif interval == 'year':
                current_date = current_date + pd.DateOffset(years=1)
            else:
                current_date = current_date + pd.DateOffset(months=1)

            dates.append(current_date)

        return dates

    @staticmethod
    def run_algorithm(algorithm: str, data: pd.DataFrame, config: ForecastConfig, save_model: bool = True) -> AlgorithmResult:
        """Run a specific forecasting algorithm"""
        from database import SessionLocal
        db = SessionLocal()
        try:
            # Print first 5 rows of data fed to algorithm
            print(f"\nData fed to algorithm '{algorithm}':")
            print(data.head(5))

            # Time-based train/test split for realistic metrics
            train, test = ForecastingEngine.time_based_split(data, test_ratio=0.2)

            # Train model on train set
            if algorithm == "linear_regression":
                forecast, metrics = ForecastingEngine.linear_regression_forecast(train, len(test) if test is not None else config.forecastPeriod)
            elif algorithm == "exponential_smoothing":
                forecast, metrics = ForecastingEngine.exponential_smoothing_forecast(train, len(test) if test is not None else config.forecastPeriod)
            else:
                # Fallback to linear regression for unsupported algorithms
                forecast, metrics = ForecastingEngine.linear_regression_forecast(train, len(test) if test is not None else config.forecastPeriod)

            # Compute test metrics
            if test is not None and len(test) > 0:
                actual = test['quantity'].values
                predicted = forecast[:len(test)]
                metrics = ForecastingEngine.calculate_metrics(actual, predicted)
            else:
                # Fallback to training metrics
                y = train['quantity'].values
                x = np.arange(len(y)).reshape(-1, 1)
                model = LinearRegression().fit(x, y)
                predicted = model.predict(x)
                metrics = ForecastingEngine.calculate_metrics(y, predicted)

            # Prepare output
            last_date = data['date'].iloc[-1]
            forecast_dates = ForecastingEngine.generate_forecast_dates(last_date, config.forecastPeriod, config.interval)

            historic_data = []
            historic_subset = data.tail(config.historicPeriod)
            for _, row in historic_subset.iterrows():
                historic_data.append(DataPoint(
                    date=row['date'].strftime('%Y-%m-%d'),
                    quantity=float(row['quantity']),
                    period=row['period']
                ))

            forecast_data = []
            for i, (date, quantity) in enumerate(zip(forecast_dates, forecast)):
                forecast_data.append(DataPoint(
                    date=date.strftime('%Y-%m-%d'),
                    quantity=float(quantity),
                    period=ForecastingEngine.format_period(date, config.interval)
                ))

            trend = ForecastingEngine.calculate_trend(data['quantity'].values)

            return AlgorithmResult(
                algorithm=ForecastingEngine.ALGORITHMS[algorithm],
                accuracy=round(metrics['accuracy'], 1),
                mae=round(metrics['mae'], 2),
                rmse=round(metrics['rmse'], 2),
                historicData=historic_data,
                forecastData=forecast_data,
                trend=trend
            )
        except Exception as e:
            print(f"Error in {algorithm}: {str(e)}")
            return AlgorithmResult(
                algorithm=ForecastingEngine.ALGORITHMS[algorithm],
                accuracy=0.0,
                mae=999.0,
                rmse=999.0,
                historicData=[],
                forecastData=[],
                trend='stable'
            )
        finally:
            db.close()

    @staticmethod
    def generate_forecast(db: Session, config: ForecastConfig, process_log: List[str] = None) -> ForecastResult:
        """Generate forecast using data from database"""
        if process_log is not None:
            process_log.append("Loading data from database...")

        df = ForecastingEngine.load_data_from_db(db, config)

        if process_log is not None:
            process_log.append(f"Data loaded: {len(df)} records")
            process_log.append("Aggregating data by period...")

        aggregated_df = ForecastingEngine.aggregate_by_period(df, config.interval, config)

        if process_log is not None:
            process_log.append(f"Data aggregated: {len(aggregated_df)} records")

        if len(aggregated_df) < 2:
            raise ValueError("Insufficient data for forecasting")

        if config.algorithm == "best_fit":
            if process_log is not None:
                process_log.append("Running best fit algorithm selection...")

            # For now, just use linear regression as best fit
            result = ForecastingEngine.run_algorithm("linear_regression", aggregated_df, config, save_model=True)
            
            return ForecastResult(
                selectedAlgorithm=f"{result.algorithm} (Best Fit)",
                accuracy=result.accuracy,
                mae=result.mae,
                rmse=result.rmse,
                historicData=result.historicData,
                forecastData=result.forecastData,
                trend=result.trend,
                allAlgorithms=[result],
                processLog=process_log
            )
        else:
            if process_log is not None:
                process_log.append(f"Running algorithm: {config.algorithm}")

            result = ForecastingEngine.run_algorithm(config.algorithm, aggregated_df, config)
            return ForecastResult(
                selectedAlgorithm=result.algorithm,
                combination=None,
                accuracy=result.accuracy,
                mae=result.mae,
                rmse=result.rmse,
                historicData=result.historicData,
                forecastData=result.forecastData,
                trend=result.trend,
                processLog=process_log
            )

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    try:
        success = init_database()
        if success:
            logger.info("✅ Database initialized successfully")
            # Create model persistence tables
            try:
                from database import engine
                SavedModel.metadata.create_all(bind=engine)
                ModelAccuracyHistory.metadata.create_all(bind=engine)
                print("✅ Model persistence tables initialized!")
            except Exception as e:
                print(f"⚠️  Model persistence table creation failed: {e}")
        else:
            logger.error("❌ Database initialization failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Database startup error: {e}")
        sys.exit(1)

# Health check endpoint
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"message": "Advanced Multi-variant Forecasting API is running", "algorithms": list(ForecastingEngine.ALGORITHMS.values())}

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = User.hash_password(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=True,
        is_approved=False,  # Requires admin approval
        is_admin=False
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse.from_orm(db_user)

@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return access token"""
    user = db.query(User).filter(User.username == credentials.username).first()
    
    if not user or not user.verify_password(credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    if not user.is_approved and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not approved by admin"
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.from_orm(user)
    )

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse.from_orm(current_user)

# Admin endpoints
@app.get("/admin/users", response_model=List[UserResponse])
async def list_users(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all users (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [UserResponse.from_orm(user) for user in users]

@app.post("/admin/users/{user_id}/approve", response_model=UserResponse)
async def approve_user(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Approve a user (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_approved = True
    db.commit()
    db.refresh(user)
    
    return UserResponse.from_orm(user)

@app.post("/admin/users/{user_id}/active", response_model=UserResponse)
async def set_user_active(
    user_id: int, 
    request: AdminSetActiveRequest,
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Set user active status (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_active = request.is_active
    db.commit()
    db.refresh(user)
    
    return UserResponse.from_orm(user)

# Forecast Selection Key endpoints
@app.post("/forecast_selection_keys/resolve", response_model=ResolveSelectionKeyResponse)
async def resolve_selection_key(
    request: ForecastSelectionKeyRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Resolve or create a forecast selection key"""
    try:
        # Query for existing selection key
        existing_key = db.query(ForecastSelectionKey).filter(
            and_(
                ForecastSelectionKey.product == request.product,
                ForecastSelectionKey.customer == request.customer,
                ForecastSelectionKey.location == request.location
            )
        ).first()
        
        if existing_key:
            return ResolveSelectionKeyResponse(
                selection_key_id=existing_key.id,
                selection_key=ForecastSelectionKeyResponse.from_orm(existing_key),
                created=False
            )
        
        # Create new selection key
        new_key = ForecastSelectionKey(
            product=request.product,
            customer=request.customer,
            location=request.location
        )
        
        db.add(new_key)
        db.commit()
        db.refresh(new_key)
        
        return ResolveSelectionKeyResponse(
            selection_key_id=new_key.id,
            selection_key=ForecastSelectionKeyResponse.from_orm(new_key),
            created=True
        )
        
    except Exception as e:
        logger.error(f"Error resolving selection key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resolve selection key: {str(e)}"
        )

@app.get("/forecast_selection_keys/{key_id}", response_model=ForecastSelectionKeyResponse)
async def get_selection_key(
    key_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a forecast selection key by ID"""
    selection_key = db.query(ForecastSelectionKey).filter(ForecastSelectionKey.id == key_id).first()
    
    if not selection_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Selection key not found"
        )
    
    return ForecastSelectionKeyResponse.from_orm(selection_key)

@app.get("/algorithms")
async def get_algorithms():
    """Get available algorithms"""
    return {"algorithms": ForecastingEngine.ALGORITHMS}

@app.get("/database/stats")
async def get_database_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get database statistics"""
    try:
        total_records = db.query(func.count(ForecastData.id)).scalar()

        # Get date range
        min_date = db.query(func.min(ForecastData.date)).scalar()
        max_date = db.query(func.max(ForecastData.date)).scalar()

        # Get unique counts
        unique_products = db.query(func.count(distinct(ForecastData.product))).scalar()
        unique_customers = db.query(func.count(distinct(ForecastData.customer))).scalar()
        unique_locations = db.query(func.count(distinct(ForecastData.location))).scalar()

        return DatabaseStats(
            totalRecords=total_records or 0,
            dateRange={
                "start": min_date.strftime('%Y-%m-%d') if min_date else "No data",
                "end": max_date.strftime('%Y-%m-%d') if max_date else "No data"
            },
            uniqueProducts=unique_products or 0,
            uniqueCustomers=unique_customers or 0,
            uniqueLocations=unique_locations or 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database stats: {str(e)}")

@app.get("/database/options")
async def get_database_options(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get unique values for dropdowns from database"""
    try:
        products = db.query(distinct(ForecastData.product)).filter(ForecastData.product.isnot(None)).all()
        customers = db.query(distinct(ForecastData.customer)).filter(ForecastData.customer.isnot(None)).all()
        locations = db.query(distinct(ForecastData.location)).filter(ForecastData.location.isnot(None)).all()

        return {
            "products": sorted([p[0] for p in products if p[0]]),
            "customers": sorted([c[0] for c in customers if c[0]]),
            "locations": sorted([l[0] for l in locations if l[0]])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database options: {str(e)}")

@app.post("/database/filtered_options")
async def get_filtered_options(
    filters: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get filtered unique values for dropdowns based on selected filters"""
    try:
        # Start with base query
        query = db.query(ForecastData)

        # Apply filters
        selected_products = filters.get('selectedProducts', [])
        selected_customers = filters.get('selectedCustomers', [])
        selected_locations = filters.get('selectedLocations', [])

        if selected_products:
            query = query.filter(ForecastData.product.in_(selected_products))
        if selected_customers:
            query = query.filter(ForecastData.customer.in_(selected_customers))
        if selected_locations:
            query = query.filter(ForecastData.location.in_(selected_locations))

        # Get filtered unique values
        products = query.with_entities(distinct(ForecastData.product)).filter(ForecastData.product.isnot(None)).all()
        customers = query.with_entities(distinct(ForecastData.customer)).filter(ForecastData.customer.isnot(None)).all()
        locations = query.with_entities(distinct(ForecastData.location)).filter(ForecastData.location.isnot(None)).all()

        return {
            "products": sorted([p[0] for p in products if p[0]]),
            "customers": sorted([c[0] for c in customers if c[0]]),
            "locations": sorted([l[0] for l in locations if l[0]])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting filtered options: {str(e)}")

@app.get("/external_factors")
async def get_external_factors(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get unique values for external factors from database"""
    try:
        factors = db.query(distinct(ExternalFactorData.factor_name)).filter(ExternalFactorData.factor_name.isnot(None)).all()

        return {
            "external_factors": sorted([f[0] for f in factors if f[0]])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting external factors: {str(e)}")

@app.post("/fetch_fred_data", response_model=FredDataResponse)
async def fetch_fred_data(
    request: FredDataRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Fetch live economic data from FRED API and store in database"""
    if not FRED_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="FRED API key not configured. Please set FRED_API_KEY environment variable."
        )

    # Clean the API key - remove any leading/trailing whitespace and special characters
    cleaned_api_key = FRED_API_KEY.strip().lstrip('+')

    try:
        total_inserted = 0
        total_duplicates = 0
        series_details = []

        for series_id in request.series_ids:
            try:
                # Validate series_id (basic check)
                if not series_id or not isinstance(series_id, str):
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': 'Invalid series ID provided',
                        'inserted': 0
                    })
                    continue

                # Construct FRED API URL
                params = {
                    'series_id': series_id.upper().strip(),  # Ensure uppercase and clean
                    'api_key': cleaned_api_key,
                    'file_type': 'json',
                    'limit': 1000
                }

                # Add date parameters if provided
                if request.start_date:
                    # Ensure proper date format
                    if isinstance(request.start_date, str):
                        params['observation_start'] = request.start_date
                    else:
                        params['observation_start'] = request.start_date.strftime('%Y-%m-%d')

                if request.end_date:
                    # Ensure proper date format
                    if isinstance(request.end_date, str):
                        params['observation_end'] = request.end_date
                    else:
                        params['observation_end'] = request.end_date.strftime('%Y-%m-%d')

                # Make API request with better error handling
                print(f"Making FRED API request for series: {series_id}")
                print(f"Request URL: {FRED_API_BASE_URL}")
                print(f"Request params: {params}")

                response = requests.get(
                    FRED_API_BASE_URL, 
                    params=params, 
                    timeout=30,
                    headers={'User-Agent': 'YourApp/1.0'}  # Add user agent
                )

                # Log the actual URL being called (without API key for security)
                safe_params = {k: v if k != 'api_key' else '***HIDDEN***' for k, v in params.items()}
                print(f"Actual request URL: {response.url}")

                response.raise_for_status()

                data = response.json()

                # Check for API-level errors
                if 'error_message' in data:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': f'FRED API error: {data["error_message"]}',
                        'inserted': 0
                    })
                    continue

                if 'observations' not in data:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': 'No observations found in API response',
                        'inserted': 0
                    })
                    continue

                observations = data['observations']

                if not observations:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'warning',
                        'message': 'No data available for the specified date range',
                        'inserted': 0
                    })
                    continue

                # Prepare records for insertion
                records_to_insert = []
                existing_records = set()

                # Get existing records for this series to avoid duplicates
                existing_query = db.query(ExternalFactorData.date).filter(
                    ExternalFactorData.factor_name == series_id
                ).all()

                for rec in existing_query:
                    existing_records.add(rec.date)

                inserted_count = 0
                duplicate_count = 0
                skipped_count = 0

                for obs in observations:
                    try:
                        # Parse date and value with better error handling
                        obs_date = pd.to_datetime(obs['date']).date()

                        # Handle missing values (FRED uses '.' for missing data)
                        if obs['value'] == '.' or obs['value'] is None or obs['value'] == '':
                            skipped_count += 1
                            continue

                        obs_value = float(obs['value'])

                        if obs_date not in existing_records:
                            record_data = ExternalFactorData(
                                date=obs_date,
                                factor_name=series_id,
                                factor_value=obs_value
                            )
                            records_to_insert.append(record_data)
                            inserted_count += 1
                        else:
                            duplicate_count += 1

                    except (ValueError, TypeError) as e:
                        print(f"Error processing observation for {series_id}: {e}")
                        print(f"Problematic observation: {obs}")
                        skipped_count += 1
                        continue

                # Bulk insert new records
                if records_to_insert:
                    try:
                        db.bulk_save_objects(records_to_insert)
                        db.commit()
                        print(f"Successfully inserted {len(records_to_insert)} records for {series_id}")
                    except Exception as db_error:
                        db.rollback()
                        series_details.append({
                            'series_id': series_id,
                            'status': 'error',
                            'message': f'Database insertion failed: {str(db_error)}',
                            'inserted': 0
                        })
                        continue

                total_inserted += inserted_count
                total_duplicates += duplicate_count

                message_parts = [f'Successfully processed {len(observations)} observations']
                if skipped_count > 0:
                    message_parts.append(f'{skipped_count} skipped (missing values)')

                series_details.append({
                    'series_id': series_id,
                    'status': 'success',
                    'message': ', '.join(message_parts),
                    'inserted': inserted_count,
                    'duplicates': duplicate_count
                })

            except requests.RequestException as e:
                error_msg = f'API request failed: {str(e)}'
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        if 'error_message' in error_data:
                            error_msg += f' - {error_data["error_message"]}'
                    except:
                        error_msg += f' - HTTP {e.response.status_code}'

                series_details.append({
                    'series_id': series_id,
                    'status': 'error',
                    'message': error_msg,
                    'inserted': 0
                })
            except Exception as e:
                series_details.append({
                    'series_id': series_id,
                    'status': 'error',
                    'message': f'Processing failed: {str(e)}',
                    'inserted': 0
                })

        return FredDataResponse(
            message=f"FRED data fetch completed. Processed {len(request.series_ids)} series.",
            inserted=total_inserted,
            duplicates=total_duplicates,
            series_processed=len(request.series_ids),
            series_details=series_details
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching FRED data: {str(e)}")

@app.get("/fred_series_info")
async def get_fred_series_info(
    current_user: User = Depends(get_current_user)
):
    """Get information about popular FRED series for users"""
    popular_series = {
        "Economic Indicators": {
            "GDP": "Gross Domestic Product",
            "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
            "UNRATE": "Unemployment Rate",
            "FEDFUNDS": "Federal Funds Rate",
            "PAYEMS": "All Employees, Total Nonfarm",
            "INDPRO": "Industrial Production Index"
        },
        "Financial Markets": {
            "DGS10": "10-Year Treasury Constant Maturity Rate",
            "DGS3MO": "3-Month Treasury Constant Maturity Rate",
            "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
            "DEXJPUS": "Japan / U.S. Foreign Exchange Rate"
        },
        "Business & Trade": {
            "HOUST": "Housing Starts",
            "RSAFS": "Advance Retail Sales",
            "IMPGS": "Imports of Goods and Services",
            "EXPGS": "Exports of Goods and Services"
        }
    }

    return {
        "message": "Popular FRED series for economic forecasting",
        "series": popular_series,
        "note": "Visit https://fred.stlouisfed.org to explore more series"
    }

@app.post("/upload_external_factors")
async def upload_external_factors(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload and store external factor data file in PostgreSQL database"""
    try:
        # Read file content
        content = await file.read()

        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel files.")

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

        # Validate required columns
        if 'date' not in df.columns or 'factor_name' not in df.columns or 'factor_value' not in df.columns:
            raise HTTPException(status_code=400, detail="Data must contain 'date', 'factor_name', and 'factor_value' columns")

        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # Convert quantity to numeric
        df['factor_value'] = pd.to_numeric(df['factor_value'], errors='coerce')
        df = df.dropna(subset=['factor_value'])

        # Validate date ranges
        validation_result = DateRangeValidator.validate_upload_data(df, db)

        # Prepare records for batch insert
        records_to_insert = []
        existing_records = set()

        # Fetch existing records keys to avoid duplicates
        existing_query = db.query(ExternalFactorData.date, ExternalFactorData.factor_name).all()
        for rec in existing_query:
            existing_records.add((rec.date, rec.factor_name))

        for _, row in df.iterrows():
            # Fix: avoid calling .date() if already datetime.date
            date_value = row['date']
            if hasattr(date_value, 'date'):
                date_value = date_value.date()
            key = (date_value, row['factor_name'])
            if key not in existing_records:
                record_data = {
                    'date': date_value,
                    'factor_name': row['factor_name'],
                    'factor_value': row['factor_value']
                }
                records_to_insert.append(ExternalFactorData(**record_data))
            else:
                # Count duplicates
                pass

        # Bulk save all new records
        db.bulk_save_objects(records_to_insert)
        db.commit()

        inserted_count = len(records_to_insert)
        duplicate_count = len(df) - inserted_count

        # Get updated statistics
        total_records = db.query(func.count(ExternalFactorData.id)).scalar()

        response = {
            "message": "File processed and stored in database successfully",
            "inserted": inserted_count,
            "duplicates": duplicate_count,
            "totalRecords": total_records,
            "filename": file.filename,
            "validation": validation_result
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/database/view", response_model=DataViewResponse)
async def view_database_data(
    request: DataViewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """View database data with filters and pagination"""
    try:
        # Build query with filters
        query = db.query(ForecastData)
        
        if request.product:
            query = query.filter(ForecastData.product == request.product)
        if request.customer:
            query = query.filter(ForecastData.customer == request.customer)
        if request.location:
            query = query.filter(ForecastData.location == request.location)
        if request.start_date:
            start_date = datetime.strptime(request.start_date, '%Y-%m-%d').date()
            query = query.filter(ForecastData.date >= start_date)
        if request.end_date:
            end_date = datetime.strptime(request.end_date, '%Y-%m-%d').date()
            query = query.filter(ForecastData.date <= end_date)
        
        # Get total count
        total_records = query.count()
        
        # Apply pagination
        offset = (request.page - 1) * request.page_size
        results = query.order_by(ForecastData.date.desc()).offset(offset).limit(request.page_size).all()
        
        # Convert to dict format
        data = []
        for record in results:
            data.append({
                'id': record.id,
                'product': record.product,
                'quantity': float(record.quantity) if record.quantity else 0,
                'product_group': record.product_group,
                'product_hierarchy': record.product_hierarchy,
                'location': record.location,
                'location_region': record.location_region,
                'customer': record.customer,
                'customer_group': record.customer_group,
                'customer_region': record.customer_region,
                'ship_to_party': record.ship_to_party,
                'sold_to_party': record.sold_to_party,
                'uom': record.uom,
                'date': record.date.strftime('%Y-%m-%d') if record.date else None,
                'unit_price': float(record.unit_price) if record.unit_price else None,
                'created_at': record.created_at.isoformat() if record.created_at else None,
                'updated_at': record.updated_at.isoformat() if record.updated_at else None
            })
        
        total_pages = (total_records + request.page_size - 1) // request.page_size
        
        return DataViewResponse(
            data=data,
            total_records=total_records,
            page=request.page,
            page_size=request.page_size,
            total_pages=total_pages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error viewing database data: {str(e)}")

@app.get("/configurations")
async def get_configurations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all saved configurations"""
    try:
        configs = db.query(ForecastConfiguration).order_by(ForecastConfiguration.updated_at.desc()).all()
        
        result = []
        for config in configs:
            result.append(ConfigurationResponse(
                id=config.id,
                name=config.name,
                description=config.description,
                config=ForecastConfig(
                    forecastBy=config.forecast_by,
                    selectedItem=config.selected_item,
                    selectedProduct=config.selected_product,
                    selectedCustomer=config.selected_customer,
                    selectedLocation=config.selected_location,
                    algorithm=config.algorithm,
                    interval=config.interval,
                    historicPeriod=config.historic_period,
                    forecastPeriod=config.forecast_period
                ),
                createdAt=config.created_at.isoformat(),
                updatedAt=config.updated_at.isoformat()
            ))
        
        return {"configurations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting configurations: {str(e)}")

@app.post("/configurations")
async def save_configuration(
    request: SaveConfigRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save a new configuration"""
    try:
        # Check if configuration name already exists
        existing = db.query(ForecastConfiguration).filter(ForecastConfiguration.name == request.name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Configuration with name '{request.name}' already exists")
        
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
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving configuration: {str(e)}")

@app.get("/configurations/{config_id}")
async def get_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific configuration by ID"""
    try:
        config = db.query(ForecastConfiguration).filter(ForecastConfiguration.id == config_id).first()
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        return ConfigurationResponse(
            id=config.id,
            name=config.name,
            description=config.description,
            config=ForecastConfig(
                forecastBy=config.forecast_by,
                selectedItem=config.selected_item,
                selectedProduct=config.selected_product,
                selectedCustomer=config.selected_customer,
                selectedLocation=config.selected_location,
                algorithm=config.algorithm,
                interval=config.interval,
                historicPeriod=config.historic_period,
                forecastPeriod=config.forecast_period
            ),
            createdAt=config.created_at.isoformat(),
            updatedAt=config.updated_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")

@app.put("/configurations/{config_id}")
async def update_configuration(
    config_id: int,
    request: SaveConfigRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update an existing configuration"""
    try:
        config = db.query(ForecastConfiguration).filter(ForecastConfiguration.id == config_id).first()
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        # Check if new name conflicts with existing (excluding current config)
        if request.name != config.name:
            existing = db.query(ForecastConfiguration).filter(
                and_(ForecastConfiguration.name == request.name, ForecastConfiguration.id != config_id)
            ).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Configuration with name '{request.name}' already exists")
        
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
        config.updated_at = datetime.utcnow()
        
        db.commit()
        
        return {
            "message": "Configuration updated successfully",
            "id": config.id,
            "name": config.name
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

@app.delete("/configurations/{config_id}")
async def delete_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a configuration"""
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
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting configuration: {str(e)}")

@app.get("/saved_forecasts")
async def get_saved_forecasts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all saved forecasts for the current user"""
    try:
        saved_forecasts = db.query(SavedForecastResult).filter(
            SavedForecastResult.user_id == current_user.id
        ).order_by(SavedForecastResult.created_at.desc()).all()

        result = []
        for forecast in saved_forecasts:
            try:
                forecast_config = ForecastConfig(**json.loads(forecast.forecast_config))
                forecast_data = json.loads(forecast.forecast_data)

                result.append({
                    'id': forecast.id,
                    'user_id': forecast.user_id,
                    'name': forecast.name,
                    'description': forecast.description,
                    'forecast_config': forecast_config,
                    'forecast_data': forecast_data,
                    'created_at': forecast.created_at.isoformat(),
                    'updated_at': forecast.updated_at.isoformat()
                })
            except Exception as e:
                print(f"Error parsing saved forecast {forecast.id}: {e}")
                continue

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting saved forecasts: {str(e)}")

@app.post("/saved_forecasts")
async def save_forecast(
    request: SavedForecastRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save a forecast result"""
    try:
        saved_forecast = SavedForecastResult(
            user_id=current_user.id,
            name=request.name,
            description=request.description,
            forecast_config=json.dumps(request.forecast_config.dict()),
            forecast_data=json.dumps(request.forecast_data.dict())
        )

        db.add(saved_forecast)
        db.commit()
        db.refresh(saved_forecast)

        return {
            "message": "Forecast saved successfully",
            "id": saved_forecast.id
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving forecast: {str(e)}")

@app.delete("/saved_forecasts/{forecast_id}")
async def delete_saved_forecast(
    forecast_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a saved forecast (only if it belongs to the current user)"""
    try:
        saved_forecast = db.query(SavedForecastResult).filter(
            SavedForecastResult.id == forecast_id,
            SavedForecastResult.user_id == current_user.id
        ).first()

        if not saved_forecast:
            raise HTTPException(
                status_code=404, 
                detail="Saved forecast not found or you don't have permission to delete it"
            )

        db.delete(saved_forecast)
        db.commit()

        return {"message": "Saved forecast deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting saved forecast: {str(e)}")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload and store data file in PostgreSQL database"""
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel files.")
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Validate required columns
        if 'date' not in df.columns or 'quantity' not in df.columns:
            raise HTTPException(status_code=400, detail="Data must contain 'date' and 'quantity' columns")
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Convert quantity to numeric
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df = df.dropna(subset=['quantity'])
        
        # Define string fields (all fields except date, quantity, and unit_price)
        string_fields = [
            'product', 'product_group', 'product_hierarchy', 
            'location', 'location_region', 
            'customer', 'customer_group', 'customer_region',
            'ship_to_party', 'sold_to_party', 'uom'
        ]
        
        # Convert string fields to strings
        for field in string_fields:
            if field in df.columns:
                df[field] = df[field].astype(str)
        
        # Get a direct connection to use psycopg2
        connection = db.connection().connection
        cursor = connection.cursor()
        
        # Map columns to database fields
        column_mapping = {
            'product': 'product',
            'quantity': 'quantity',
            'product_group': 'product_group',
            'product_hierarchy': 'product_hierarchy',
            'location': 'location',
            'location_region': 'location_region',
            'customer': 'customer',
            'customer_group': 'customer_group',
            'customer_region': 'customer_region',
            'ship_to_party': 'ship_to_party',
            'sold_to_party': 'sold_to_party',
            'uom': 'uom',
            'date': 'date',
            'unit_price': 'unit_price'
        }
        
        # Prepare data for insertion
        columns = [col for col in column_mapping.values() if col in df.columns]
        columns_str = ', '.join(columns)
        
        # Add created_at and updated_at
        columns_str += ', created_at, updated_at'
        
        # Create values part of the SQL statement
        values_list = []
        params = []
        
        for _, row in df.iterrows():
            values = []
            row_params = []
            
            for col in columns:
                if col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        values.append('NULL')
                    elif col == 'date':
                        values.append('%s')
                        row_params.append(value.date())
                    else:
                        values.append('%s')
                        row_params.append(value)
                else:
                    values.append('NULL')
            
            # Add created_at and updated_at
            now = datetime.utcnow()
            values.append('%s')
            values.append('%s')
            row_params.append(now)
            row_params.append(now)
            
            values_list.append('(' + ', '.join(values) + ')')
            params.extend(row_params)
        
        # Build the SQL statement with ON CONFLICT DO NOTHING
        sql = f"""
        INSERT INTO forecast_data ({columns_str})
        VALUES {', '.join(values_list)}
        ON CONFLICT (product, customer, location, date) DO NOTHING;
        """
        
        # Execute the SQL
        cursor.execute(sql, params)
        inserted_count = cursor.rowcount
        
        # Commit the transaction
        connection.commit()
        
        # Get total records count
        total_records = db.query(func.count(ForecastData.id)).scalar()
        
        return {
            "message": "File processed and stored in database successfully",
            "inserted": inserted_count,
            "duplicates": len(df) - inserted_count,
            "totalRecords": total_records,
            "filename": file.filename
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Add a global flag to enable/disable caching
ENABLE_MODEL_CACHE = True  # Enable caching to improve performance with parallel execution

@app.post("/forecast")
def generate_forecast(
    config: ForecastConfig,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate forecast using data from database"""
    try:
        # Initialize process log
        process_log = []
        process_log.append("=== Forecast Request Received ===")
        process_log.append(f"Multi-select mode: {config.multiSelect}")
        process_log.append(f"Advanced mode: {config.advancedMode}")
        process_log.append(f"Selected products: {config.selectedProducts}")
        process_log.append(f"Selected customers: {config.selectedCustomers}")
        process_log.append(f"Selected locations: {config.selectedLocations}")
        process_log.append(f"Selected items: {config.selectedItems}")

        # Single selection mode (backward compatibility)
        process_log.append("Running single selection mode")
        result = ForecastingEngine.generate_forecast(db, config, process_log)

        # Auto-save the forecast result
        try:
            auto_save_name = f"Auto-saved Forecast {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            auto_save_description = "Automatically saved forecast result"

            # Check if a forecast with this name already exists
            existing = db.query(SavedForecastResult).filter(
                SavedForecastResult.user_id == current_user.id,
                SavedForecastResult.name == auto_save_name
            ).first()

            if not existing:
                saved_forecast = SavedForecastResult(
                    user_id=current_user.id,
                    name=auto_save_name,
                    description=auto_save_description,
                    forecast_config=json.dumps(config.dict()),
                    forecast_data=json.dumps(result.dict())
                )
                db.add(saved_forecast)
                db.commit()
                process_log.append(f"Forecast automatically saved as '{auto_save_name}'")
            else:
                # If exists, append a number to make it unique
                counter = 1
                new_name = f"{auto_save_name} ({counter})"
                while db.query(SavedForecastResult).filter(
                    SavedForecastResult.user_id == current_user.id,
                    SavedForecastResult.name == new_name
                ).first():
                    counter += 1
                    new_name = f"{auto_save_name} ({counter})"

                saved_forecast = SavedForecastResult(
                    user_id=current_user.id,
                    name=new_name,
                    description=auto_save_description,
                    forecast_config=json.dumps(config.dict()),
                    forecast_data=json.dumps(result.dict())
                )
                db.add(saved_forecast)
                db.commit()
                process_log.append(f"Forecast automatically saved as '{new_name}'")
        except Exception as save_error:
            process_log.append(f"Warning: Could not auto-save forecast: {str(save_error)}")

        return result

    except Exception as e:
        if 'process_log' in locals():
            process_log.append(f"FATAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@app.post("/download_forecast_excel")
async def download_forecast_excel(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Download single forecast data as Excel"""
    try:
        result = ForecastResult(**request['forecastResult'])
        forecast_by = request.get('forecastBy', '')
        selected_item = request.get('selectedItem', '')
 
        # Prepare data with selected items
        hist = result.historicData
        fore = result.forecastData
 
        # Determine selected items
        product = ''
        customer = ''
        location = ''
 
        # Handle combination data from result
        if result.combination:
            product = result.combination.get('product', product)
            customer = result.combination.get('customer', customer)
            location = result.combination.get('location', location)
        else:
            # Handle simple mode
            if forecast_by == 'product':
                product = selected_item
            elif forecast_by == 'customer':
                customer = selected_item
            elif forecast_by == 'location':
                location = selected_item
 
        # Create comprehensive Excel data
        hist_rows = []
        fore_rows = []
 
        for d in hist:
            hist_rows.append({
                "Product": product,
                "Customer": customer,
                "Location": location,
                "Date": d.date,
                "Period": d.period,
                "Quantity": d.quantity,
                "Type": "Historical"
            })
 
        for d in fore:
            fore_rows.append({
                "Product": product,
                "Customer": customer,
                "Location": location,
                "Date": d.date,
                "Period": d.period,
                "Quantity": d.quantity,
                "Type": "Forecast"
            })
 
        all_rows = hist_rows + fore_rows
 
        # Create DataFrame
        df = pd.DataFrame(all_rows)
 
        # Add configuration details
        config_df = pd.DataFrame([{
            "Algorithm": result.selectedAlgorithm,
            "Accuracy": result.accuracy,
            "MAE": result.mae,
            "RMSE": result.rmse,
            "Trend": result.trend,
            "Historic Periods": len(result.historicData),
            "Forecast Periods": len(result.forecastData)
        }])
 
        # Write to Excel in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # Main forecast data
            df.to_excel(writer, index=False, sheet_name="Forecast Data")
 
            # Configuration details
            config_df.to_excel(writer, index=False, sheet_name="Configuration")
 
            # If multiple algorithms were compared, include them
            if result.allAlgorithms:
                algo_data = []
                for algo in result.allAlgorithms:
                    algo_data.append({
                        "Algorithm": algo.algorithm,
                        "Accuracy": algo.accuracy,
                        "MAE": algo.mae,
                        "RMSE": algo.rmse,
                        "Trend": algo.trend
                    })
                algo_df = pd.DataFrame(algo_data)
                algo_df.to_excel(writer, index=False, sheet_name="All Algorithms")
 
        output.seek(0)
 
        # Create filename with selected items - sanitize for filesystem
        filename_parts = []
        if product:
            # Remove/replace special characters
            safe_product = "".join(c for c in str(product) if c.isalnum() or c in (' ', '-', '_')).strip()
            if safe_product:
                filename_parts.append(safe_product)
        if customer:
            safe_customer = "".join(c for c in str(customer) if c.isalnum() or c in (' ', '-', '_')).strip()
            if safe_customer:
                filename_parts.append(safe_customer)
        if location:
            safe_location = "".join(c for c in str(location) if c.isalnum() or c in (' ', '-', '_')).strip()
            if safe_location:
                filename_parts.append(safe_location)
 
        filename_base = "_".join(filename_parts) if filename_parts else "forecast"
        filename = f"{filename_base}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
 
        return StreamingResponse(
            io.BytesIO(output.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Excel: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Advanced Multi-variant Forecasting API with PostgreSQL")
    print("📊 Algorithms + Best Fit Available")
    print("🗄️  PostgreSQL Database Integration")
    print("🌐 Server starting on http://localhost:8000")
    print("📈 Frontend should be available on http://localhost:5173")
    print("⏹️  Press Ctrl+C to stop the server\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)