#!/usr/bin/env python3
"""
Advanced Multi-variant Forecasting API with MySQL Database Integration
"""

from database import get_db, init_database, ForecastData, User, ExternalFactorData, ForecastConfiguration
from model_persistence import ModelPersistenceManager, SavedModel, ModelAccuracyHistory
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
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
from sqlalchemy.orm import Session # type: ignore
from sqlalchemy import func, distinct, and_, or_ # type: ignore
from database import get_db, init_database, ForecastData, User, ExternalFactorData
from database import ForecastConfiguration,SavedForecastResult
from auth import create_access_token, get_current_user, get_current_user_optional
from validation import DateRangeValidator
import requests
import os
warnings.filterwarnings('ignore')

app = FastAPI(title="Multi-variant Forecasting API with MySQL", version="3.0.0")

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

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Multi-variant Forecasting API...")
    if init_database():
        print("✅ Database initialization successful!")
        # Create model persistence tables
        try:
            from database import engine
            SavedModel.metadata.create_all(bind=engine)
            ModelAccuracyHistory.metadata.create_all(bind=engine)
            print("✅ Model persistence tables initialized!")
        except Exception as e:
            print(f"⚠️  Model persistence table creation failed: {e}")
    else:
        print("⚠️  Database initialization failed - some features may not work")
        print("Please run: python setup_database.py")

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

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

# class UserResponse(BaseModel):
#     id: int
#     username: str
#     email: str
#     full_name: Optional[str] = None
#     is_active: bool
#     created_at: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_approved: Optional[bool] = None
    is_admin: Optional[bool] = None
    created_at: str

class AdminSetActiveRequest(BaseModel):
    is_active: bool

def require_admin(user: User):
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class DataViewRequest(BaseModel):
    product: Optional[str] = None
    customer: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    page: int = 1
    page_size: int = 50

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
        """Load data from MySQL database based on forecast configuration"""
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
    def aggregate_advanced_mode(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Aggregate data for advanced mode (Product-Customer-Location-Date combinations)
        No aggregation across dimensions - keeps unique combinations intact
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Identify external factor columns
        external_factor_cols = [col for col in df.columns if col not in ['quantity', 'product', 'customer', 'location', 'date']]
        print(f"External factor columns found in aggregate_advanced_mode: {external_factor_cols}")

        # Create period groupings without aggregating across dimensions
        if interval == 'week':
            df['period_group'] = df['date'].dt.to_period('W-MON')
        elif interval == 'month':
            df['period_group'] = df['date'].dt.to_period('M')
        elif interval == 'year':
            df['period_group'] = df['date'].dt.to_period('Y')
        else:
            df['period_group'] = df['date'].dt.to_period('M')

        # Group by all dimensions plus period to preserve unique combinations
        # This ensures Product-Customer-Location-Date combinations remain distinct
        group_cols = ['product', 'customer', 'location', 'period_group']

        # Aggregate quantity (sum) and external factors (mean)
        agg_dict = {'quantity': 'sum'}
        for col in external_factor_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'  # Use mean for external factors

        aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()

        # Convert period back to timestamp
        aggregated['date'] = aggregated['period_group'].dt.start_time

        # Add period labels
        aggregated['period'] = aggregated['date'].apply(
            lambda x: ForecastingEngine.format_period(pd.Timestamp(x), interval)
        )

        # Clean up
        aggregated = aggregated.drop('period_group', axis=1)

        print(f"Advanced mode aggregated DataFrame columns: {aggregated.columns.tolist()}")
        print(f"Sample advanced mode aggregated data:")
        print(aggregated.head())

        return aggregated.reset_index(drop=True)

    @staticmethod
    def generate_simple_multi_forecast(db: Session, config: ForecastConfig, process_log: List[str] = None) -> MultiForecastResult:
        """Generate forecasts for multiple items in simple mode"""
        if process_log is not None:
            process_log.append("Generating simple multi-forecast...")
            process_log.append(f"Selected items: {config.selectedItems}")

        if not config.selectedItems or len(config.selectedItems) == 0:
            raise ValueError(f"Please select at least one {config.forecastBy} for multi-selection forecasting")

        results = []
        successful_combinations = 0
        failed_combinations = []

        for item in config.selectedItems:
            try:
                if process_log is not None:
                    process_log.append(f"Processing item: {item}")

                # Create a single-item config
                single_config = ForecastConfig(
                    forecastBy=config.forecastBy,
                    selectedItem=item,
                    algorithm=config.algorithm,
                    interval=config.interval,
                    historicPeriod=config.historicPeriod,
                    forecastPeriod=config.forecastPeriod,
                    externalFactors=config.externalFactors
                )

                # Load and aggregate data for this item
                if process_log is not None:
                    process_log.append("Loading data from database...")

                df = ForecastingEngine.load_data_from_db(db, single_config)

                if process_log is not None:
                    process_log.append(f"Data loaded: {len(df)} records")
                    process_log.append("Aggregating data by period...")

                aggregated_df = ForecastingEngine.aggregate_by_period(df, config.interval, single_config)

                if process_log is not None:
                    process_log.append(f"Data aggregated: {len(aggregated_df)} records")

                if len(aggregated_df) < 2:
                    if process_log is not None:
                        process_log.append("Insufficient data for forecasting")

                    failed_combinations.append({
                        'combination': item,
                        'error': 'Insufficient data'
                    })
                    continue

                if config.algorithm in ["best_fit", "best_statistical", "best_ml", "best_specialized"]:
                    if process_log is not None:
                        process_log.append(f"Running {config.algorithm} algorithm selection...")

                    # Define algorithm categories
                    statistical_algorithms = [
                        "linear_regression", "polynomial_regression", "exponential_smoothing", 
                        "holt_winters", "arima", "sarima", "ses", "damped_trend", 
                        "theta_method", "drift_method", "naive_seasonal", "prophet_like"
                    ]

                    ml_algorithms = [
                        "random_forest", "xgboost", "svr", "knn", "gaussian_process", 
                        "neural_network", "lstm_like"
                    ]

                    specialized_algorithms = [
                        "seasonal_decomposition", "moving_average", "croston"
                    ]

                    # Select algorithms based on category
                    if config.algorithm == "best_statistical":
                        algorithms = statistical_algorithms
                    elif config.algorithm == "best_ml":
                        algorithms = ml_algorithms
                    elif config.algorithm == "best_specialized":
                        algorithms = specialized_algorithms
                    else:  # best_fit
                        algorithms = [alg for alg in ForecastingEngine.ALGORITHMS.keys() 
                                    if alg not in ["best_fit", "best_statistical", "best_ml", "best_specialized"]]

                    algorithm_results = []  # Initialize before parallel execution

                    # Use parallel execution for algorithm evaluation
                    max_workers = min(len(algorithms), os.cpu_count() or 4)
                    if process_log is not None:
                        process_log.append(f"Running parallel evaluation with {max_workers} workers...")

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_algorithm = {
                            executor.submit(ForecastingEngine.run_algorithm, algorithm, aggregated_df, single_config, save_model=False): algorithm
                            for algorithm in algorithms
                        }

                        for future in as_completed(future_to_algorithm):
                            algorithm_name = future_to_algorithm[future]
                            try:
                                result = future.result()
                                if result:  # Check if result is not None
                                    algorithm_results.append(result)
                                    if process_log is not None:
                                        process_log.append(f"✅ Algorithm {algorithm_name} completed for {item}")
                            except Exception as exc:
                                if process_log is not None:
                                    process_log.append(f"❌ Algorithm {algorithm_name} failed for {item}: {str(exc)}")
                                continue

                    if not algorithm_results:
                        if process_log is not None:
                            process_log.append(f"No algorithms produced valid results for {item}")

                        failed_combinations.append({
                            'combination': item,
                            'error': 'No algorithms produced valid results'
                        })
                        continue

                    best_result = max(algorithm_results, key=lambda x: x.accuracy)
                    combination_dict = {config.forecastBy: item}
                    
                    forecast_result = ForecastResult(
                        combination=combination_dict,
                        selectedAlgorithm=f"{best_result.algorithm} (Best {config.algorithm.split('_')[1].capitalize()})",
                        accuracy=best_result.accuracy,
                        mae=best_result.mae,
                        rmse=best_result.rmse,
                        historicData=best_result.historicData,
                        forecastData=best_result.forecastData,
                        trend=best_result.trend,
                        allAlgorithms=algorithm_results,
                        processLog=process_log.copy() if process_log is not None else None
                    )
                else:
                    if process_log is not None:
                        process_log.append(f"Running algorithm: {config.algorithm}")

                    result = ForecastingEngine.run_algorithm(config.algorithm, aggregated_df, single_config)
                    combination_dict = {config.forecastBy: item}
                    
                    forecast_result = ForecastResult(
                        combination=combination_dict,
                        selectedAlgorithm=result.algorithm,
                        accuracy=result.accuracy,
                        mae=result.mae,
                        rmse=result.rmse,
                        historicData=result.historicData,
                        forecastData=result.forecastData,
                        trend=result.trend,
                        processLog=process_log.copy() if process_log is not None else None
                    )

                results.append(forecast_result)
                successful_combinations += 1

                if process_log is not None:
                    process_log.append(f"Item {item} processed successfully")

            except Exception as e:
                if process_log is not None:
                    process_log.append(f"Error processing item {item}: {str(e)}")

                failed_combinations.append({
                    'combination': item,
                    'error': str(e)
                })

        if not results:
            raise ValueError("No valid forecasts could be generated for any item")

        # Calculate summary statistics
        avg_accuracy = np.mean([r.accuracy for r in results])
        best_combination = max(results, key=lambda x: x.accuracy)
        worst_combination = min(results, key=lambda x: x.accuracy)

        summary = {
            'averageAccuracy': round(avg_accuracy, 2),
            'bestCombination': {
                'combination': best_combination.combination,
                'accuracy': best_combination.accuracy
            },
            'worstCombination': {
                'combination': worst_combination.combination,
                'accuracy': worst_combination.accuracy
            },
            'successfulCombinations': successful_combinations,
            'failedCombinations': len(failed_combinations),
            'failedDetails': failed_combinations
        }

        if process_log is not None:
            process_log.append(f"Multi-forecast completed. Successful: {successful_combinations}, Failed: {len(failed_combinations)}")

        return MultiForecastResult(
            results=results,
            totalCombinations=len(config.selectedItems),
            summary=summary,
            processLog=process_log
        )

    @staticmethod
    def load_data_for_combination(db: Session, product: str, customer: str, location: str) -> pd.DataFrame:
        """Load data from MySQL database for a specific combination"""
        query = db.query(ForecastData).filter(
            ForecastData.product == product,
            ForecastData.customer == customer,
            ForecastData.location == location
        )

        results = query.all()

        if not results:
            raise ValueError(f"No data found for combination: {product} + {customer} + {location}")

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
        return df

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

    # Algorithm implementations (keeping all existing algorithms)
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
    def polynomial_regression_forecast(data: pd.DataFrame, periods: int, degree: int = 2) -> tuple:
        """Polynomial regression forecasting with feature engineering and external factors"""
        y = data['quantity'].values
        n = len(y)
        window = min(5, n - 1)
        # Get external factor columns
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        if window < 1:
            x = np.arange(n).reshape(-1, 1)
            if external_factor_cols:
                x = np.hstack([x, data[external_factor_cols].values])
            best_metrics = None
            best_forecast = None
            for d in [2, 3]:
                coeffs = np.polyfit(np.arange(n), y, d)
                poly_func = np.poly1d(coeffs)
                future_x = np.arange(n, n + periods).reshape(-1, 1)
                if external_factor_cols:
                    last_factors = data[external_factor_cols].iloc[-1].values
                    future_factors = np.tile(last_factors, (periods, 1))
                    future_x = np.hstack([future_x, future_factors])
                forecast = poly_func(np.arange(n, n + periods))
                forecast = np.maximum(forecast, 0)
                predicted = poly_func(np.arange(n))
                metrics = ForecastingEngine.calculate_metrics(y, predicted)
                if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                    best_metrics = metrics
                    best_forecast = forecast
            return best_forecast, best_metrics
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
        best_metrics = None
        best_forecast = None
        for d in [2, 3]:
            coeffs = np.polyfit(np.arange(len(y_target)), y_target, d)
            poly_func = np.poly1d(coeffs)
            future_x = np.arange(len(y_target), len(y_target) + periods)
            forecast = poly_func(future_x)
            forecast = np.maximum(forecast, 0)
            predicted = poly_func(np.arange(len(y_target)))
            metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
            if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                best_metrics = metrics
                best_forecast = forecast
        return best_forecast, best_metrics


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
    def holt_winters_forecast(data: pd.DataFrame, periods: int, season_length: int = 12) -> tuple:
        """Enhanced Holt-Winters with external factors integration"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if n < 2 * season_length:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

        if external_factor_cols:
            # Enhanced Holt-Winters with external factor regression
            # First decompose the series using traditional Holt-Winters
            alpha, beta, gamma = 0.3, 0.1, 0.1

            level = np.mean(y[:season_length])
            trend = (np.mean(y[season_length:2*season_length]) - np.mean(y[:season_length])) / season_length
            seasonal = y[:season_length] - level

            levels = [level]
            trends = [trend]
            seasonals = list(seasonal)
            fitted = []
            residuals = []

            # Apply Holt-Winters to get base forecast
            for i in range(len(y)):
                if i == 0:
                    forecast_val = level + trend + seasonal[i % season_length]
                    fitted.append(forecast_val)
                    residuals.append(y[i] - forecast_val)
                else:
                    level = alpha * (y[i] - seasonals[i % season_length]) + (1 - alpha) * (levels[-1] + trends[-1])
                    trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]
                    if len(seasonals) > i:
                        seasonals[i % season_length] = gamma * (y[i] - level) + (1 - gamma) * seasonals[i % season_length]

                    levels.append(level)
                    trends.append(trend)
                    forecast_val = level + trend + seasonals[i % season_length]
                    fitted.append(forecast_val)
                    residuals.append(y[i] - forecast_val)

            # Use external factors to model residuals
            window = min(3, len(residuals) - 1)
            if window > 0 and len(residuals) > window:
                X, y_residual = [], []
                for i in range(window, len(residuals)):
                    features = list(residuals[i-window:i])  # Lag residuals
                    features.extend(data[external_factor_cols].iloc[i].values)
                    X.append(features)
                    y_residual.append(residuals[i])

                if len(X) > 1:
                    X = np.array(X)
                    y_residual = np.array(y_residual)

                    # Model residuals with external factors
                    residual_model = LinearRegression()
                    residual_model.fit(X, y_residual)

                    # Forecast with external factors
                    forecast = []
                    recent_residuals = residuals[-window:]

                    for i in range(periods):
                        # Base Holt-Winters forecast
                        hw_forecast = level + (i + 1) * trend + seasonals[(len(y) + i) % season_length]

                        # Residual correction using external factors
                        features = list(recent_residuals)
                        features.extend(data[external_factor_cols].iloc[-1].values)

                        residual_correction = residual_model.predict([features])[0]
                        final_forecast = hw_forecast + residual_correction
                        final_forecast = max(0, final_forecast)

                        forecast.append(final_forecast)
                        recent_residuals = recent_residuals[1:] + [residual_correction]

                    # Calculate metrics on corrected fitted values
                    fitted_corrected = np.array(fitted) + residual_model.predict(X)
                    metrics = ForecastingEngine.calculate_metrics(y[window:], fitted_corrected)

                    return np.array(forecast), metrics

        # Traditional Holt-Winters without external factors
        alpha, beta, gamma = 0.3, 0.1, 0.1

        level = np.mean(y[:season_length])
        trend = (np.mean(y[season_length:2*season_length]) - np.mean(y[:season_length])) / season_length
        seasonal = y[:season_length] - level

        levels = [level]
        trends = [trend]
        seasonals = list(seasonal)
        fitted = []

        for i in range(len(y)):
            if i == 0:
                fitted.append(level + trend + seasonal[i % season_length])
            else:
                level = alpha * (y[i] - seasonals[i % season_length]) + (1 - alpha) * (levels[-1] + trends[-1])
                trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]
                if len(seasonals) > i:
                    seasonals[i % season_length] = gamma * (y[i] - level) + (1 - gamma) * seasonals[i % season_length]

                levels.append(level)
                trends.append(trend)
                fitted.append(level + trend + seasonals[i % season_length])

        forecast = []
        for i in range(periods):
            forecast_value = level + (i + 1) * trend + seasonals[(len(y) + i) % season_length]
            forecast.append(max(0, forecast_value))

        metrics = ForecastingEngine.calculate_metrics(y, fitted)

        return np.array(forecast), metrics

    @staticmethod
    def arima_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Actual ARIMA/ARIMAX forecasting using statsmodels"""
        from statsmodels.tsa.arima.model import ARIMA
        from pmdarima import auto_arima # type: ignore
        import warnings

        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if len(y) < 10:  # Need sufficient data for ARIMA
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if external_factor_cols:
                    # ARIMAX: ARIMA with external regressors
                    print(f"Running ARIMAX with external factors: {external_factor_cols}")

                    # Prepare external regressors
                    exog = data[external_factor_cols].values

                    # Auto-select ARIMA parameters with external regressors
                    try:
                        auto_model = auto_arima(
                            y, 
                            exogenous=exog,
                            start_p=0, start_q=0, max_p=3, max_q=3, max_d=2,
                            seasonal=False,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=False
                        )

                        # Forecast external factors properly
                        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
                        future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)

                        # Create future exogenous matrix
                        future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
                        forecast = auto_model.predict(n_periods=periods, exogenous=future_exog)
                        forecast = np.maximum(forecast, 0)

                        # Calculate metrics
                        fitted = auto_model.fittedvalues()
                        if len(fitted) == len(y):
                            metrics = ForecastingEngine.calculate_metrics(y, fitted)
                        else:
                            # Handle case where fitted values might be shorter due to differencing
                            start_idx = len(y) - len(fitted)
                            metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                        print(f"ARIMAX model: {auto_model.order}, AIC: {auto_model.aic():.2f}")
                        return forecast, metrics

                    except Exception as e:
                        print(f"Auto ARIMAX failed: {e}, trying manual ARIMAX")

                        # Fallback to manual ARIMAX
                        for order in [(1,1,1), (2,1,1), (1,1,2), (0,1,1)]:
                            try:
                                model = ARIMA(y, exog=exog, order=order)
                                fitted_model = model.fit()

                                # Use forecasted external factors
                                future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
                                forecast = fitted_model.forecast(steps=periods, exog=future_exog)
                                forecast = np.maximum(forecast, 0)

                                fitted = fitted_model.fittedvalues
                                if len(fitted) == len(y):
                                    metrics = ForecastingEngine.calculate_metrics(y, fitted)
                                else:
                                    start_idx = len(y) - len(fitted)
                                    metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                                print(f"Manual ARIMAX model: {order}, AIC: {fitted_model.aic:.2f}")
                                return forecast, metrics

                            except:
                                continue

                else:
                    # Traditional ARIMA without external factors
                    print("Running ARIMA without external factors")

                    try:
                        # Auto-select ARIMA parameters
                        auto_model = auto_arima(
                            y,
                            start_p=0, start_q=0, max_p=3, max_q=3, max_d=2,
                            seasonal=False,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=False
                        )

                        forecast = auto_model.predict(n_periods=periods)
                        forecast = np.maximum(forecast, 0)

                        # Calculate metrics
                        fitted = auto_model.fittedvalues()
                        if len(fitted) == len(y):
                            metrics = ForecastingEngine.calculate_metrics(y, fitted)
                        else:
                            start_idx = len(y) - len(fitted)
                            metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                        print(f"ARIMA model: {auto_model.order}, AIC: {auto_model.aic():.2f}")
                        return forecast, metrics

                    except Exception as e:
                        print(f"Auto ARIMA failed: {e}, trying manual ARIMA")

                        # Fallback to manual ARIMA
                        for order in [(1,1,1), (2,1,1), (1,1,2), (0,1,1), (1,0,1)]:
                            try:
                                model = ARIMA(y, order=order)
                                fitted_model = model.fit()

                                forecast = fitted_model.forecast(steps=periods)
                                forecast = np.maximum(forecast, 0)

                                fitted = fitted_model.fittedvalues
                                if len(fitted) == len(y):
                                    metrics = ForecastingEngine.calculate_metrics(y, fitted)
                                else:
                                    start_idx = len(y) - len(fitted)
                                    metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                                print(f"Manual ARIMA model: {order}, AIC: {fitted_model.aic:.2f}")
                                return forecast, metrics

                            except:
                                continue

        except Exception as e:
            print(f"ARIMA forecasting failed: {e}")

        # Final fallback to exponential smoothing
        print("ARIMA failed, falling back to exponential smoothing")
        return ForecastingEngine.exponential_smoothing_forecast(data, periods)

    @staticmethod
    def random_forest_forecast(data: pd.DataFrame, periods: int, n_estimators_list: list = [50, 100, 200], max_depth_list: list = [3, 5, None]) -> tuple:
        """Random Forest regression forecasting with hyperparameter tuning"""
        y = data['quantity'].values
        dates = pd.to_datetime(data['date'])

        # Get external factor columns
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        # Create features
        features = []
        targets = []
        window = min(5, len(y) - 1)

        for i in range(window, len(y)):
            lags = y[i-window:i]
            trend = i
            seasonal = i % 12
            month = dates.iloc[i].month
            quarter = dates.iloc[i].quarter
            feature_vector = list(lags) + [trend, seasonal, month, quarter]
            if external_factor_cols:
                feature_vector.extend(data[external_factor_cols].iloc[i].values)
            features.append(feature_vector)
            targets.append(y[i])

        if len(features) < 3:
            return ForecastingEngine.linear_regression_forecast(data, periods)

        features = np.array(features)
        targets = np.array(targets)

        best_metrics = None
        best_forecast = None

        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                print(f"Running Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(features, targets)

                # Forecast
                forecast = []
                recent_values = list(y[-window:])
                last_date = dates.iloc[-1]

                for i in range(periods):
                    trend_val = len(y) + i
                    seasonal_val = (len(y) + i) % 12
                    next_date = last_date + pd.DateOffset(months=i+1)
                    month_val = next_date.month
                    quarter_val = next_date.quarter
                    feature_vector = recent_values + [trend_val, seasonal_val, month_val, quarter_val]
                    if external_factor_cols:
                        # Use forecasted external factors
                        if 'future_factors' not in locals():
                            future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
                        forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
                        feature_vector.extend(forecasted_factors)

                    next_value = model.predict([feature_vector])[0]
                    next_value = max(0, next_value)
                    forecast.append(next_value)
                    recent_values = recent_values[1:] + [next_value]

                predicted = model.predict(features)
                metrics = ForecastingEngine.calculate_metrics(targets, predicted)
                print(f"n_estimators={n_estimators}, max_depth={max_depth}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, Accuracy={metrics['accuracy']:.2f}")

                if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                    best_metrics = metrics
                    best_forecast = forecast

        return np.array(best_forecast), best_metrics

    @staticmethod
    def seasonal_decomposition_forecast(data: pd.DataFrame, periods: int, season_length: int = 12) -> tuple:
        """Enhanced seasonal decomposition with external factors"""
        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if len(y) < 2 * season_length:
            return ForecastingEngine.linear_regression_forecast(data, periods)

        # Simple seasonal decomposition
        trend = np.convolve(y, np.ones(season_length)/season_length, mode='same')

        # Calculate seasonal component
        detrended = y - trend
        seasonal_pattern = []
        for i in range(season_length):
            seasonal_values = [detrended[j] for j in range(i, len(detrended), season_length)]
            seasonal_pattern.append(np.mean(seasonal_values))

        if external_factor_cols:
            # Use external factors to enhance trend forecasting
            x = np.arange(len(trend))
            valid_trend = ~np.isnan(trend)

            if np.sum(valid_trend) > season_length:
                # Prepare features for trend modeling
                X_trend, y_trend = [], []
                for i in range(season_length, len(y)):
                    if not np.isnan(trend[i]):
                        features = [i]  # Time index
                        features.extend(data[external_factor_cols].iloc[i].values)
                        X_trend.append(features)
                        y_trend.append(trend[i])

                if len(X_trend) > 1:
                    X_trend = np.array(X_trend)
                    y_trend = np.array(y_trend)

                    # Model trend with external factors
                    trend_model = LinearRegression()
                    trend_model.fit(X_trend, y_trend)

                    # Forecast trend with external factors
                    future_trend = []
                    for i in range(periods):
                        features = [len(y) + i]
                        features.extend(data[external_factor_cols].iloc[-1].values)
                        trend_forecast = trend_model.predict([features])[0]
                        future_trend.append(trend_forecast)

                    # Forecast seasonal (repeat pattern)
                    future_seasonal = [seasonal_pattern[(len(y) + i) % season_length] for i in range(periods)]

                    # Combine forecast
                    forecast = np.array(future_trend) + np.array(future_seasonal)
                    forecast = np.maximum(forecast, 0)

                    # Calculate metrics using model predictions
                    trend_predicted = trend_model.predict(X_trend)
                    seasonal_full = np.tile(seasonal_pattern, len(y) // season_length + 1)[:len(y)]
                    fitted = np.full(len(y), np.nan)

                    for i, idx in enumerate(range(season_length, len(y))):
                        if not np.isnan(trend[idx]):
                            fitted[idx] = trend_predicted[X_trend[:, 0] == idx][0] if np.any(X_trend[:, 0] == idx) else trend[idx]
                            fitted[idx] += seasonal_full[idx]

                    valid_fitted = ~np.isnan(fitted)
                    if np.sum(valid_fitted) > 0:
                        metrics = ForecastingEngine.calculate_metrics(y[valid_fitted], fitted[valid_fitted])
                    else:
                        metrics = {'accuracy': 50.0, 'mae': np.std(y), 'rmse': np.std(y)}

                    return forecast, metrics

        # Traditional seasonal decomposition without external factors
        x = np.arange(len(trend))
        valid_trend = ~np.isnan(trend)
        if np.sum(valid_trend) > 1:
            slope, intercept, _, _, _ = stats.linregress(x[valid_trend], trend[valid_trend])
            future_trend = [slope * (len(y) + i) + intercept for i in range(periods)]
        else:
            future_trend = [np.nanmean(trend)] * periods

        future_seasonal = [seasonal_pattern[(len(y) + i) % season_length] for i in range(periods)]

        forecast = np.array(future_trend) + np.array(future_seasonal)
        forecast = np.maximum(forecast, 0)

        seasonal_full = np.tile(seasonal_pattern, len(y) // season_length + 1)[:len(y)]
        fitted = trend + seasonal_full
        valid_fitted = ~np.isnan(fitted)
        if np.sum(valid_fitted) > 0:
            metrics = ForecastingEngine.calculate_metrics(y[valid_fitted], fitted[valid_fitted])
        else:
            metrics = {'accuracy': 50.0, 'mae': np.std(y), 'rmse': np.std(y)}

        return forecast, metrics

    @staticmethod
    def moving_average_forecast(data: pd.DataFrame, periods: int, window: int = 3) -> tuple:
        """Enhanced moving average forecasting with external factors"""
        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        window = min(window, len(y))

        if external_factor_cols and len(y) > window + 2:
            # Moving average with external factor regression
            ma_values = []
            for i in range(len(y)):
                start_idx = max(0, i - window + 1)
                ma_values.append(np.mean(y[start_idx:i+1]))

            # Use external factors to adjust moving average
            X, y_target = [], []
            for i in range(window, len(y)):
                features = [ma_values[i]]  # Moving average as base feature
                features.extend(data[external_factor_cols].iloc[i].values)  # External factors
                X.append(features)
                y_target.append(y[i])

            if len(X) > 1:
                X = np.array(X)
                y_target = np.array(y_target)

                # Fit model to adjust moving average with external factors
                model = LinearRegression()
                model.fit(X, y_target)

                # Forecast
                forecast = []
                for i in range(periods):
                    # Calculate moving average based on recent values
                    if i == 0:
                        recent_avg = np.mean(y[-window:])
                    else:
                        # Use forecasted values for moving average
                        recent_values = list(y[-window+i:]) + forecast[:i]
                        recent_avg = np.mean(recent_values[-window:])

                    # Apply external factors
                    features = [recent_avg]
                    features.extend(data[external_factor_cols].iloc[-1].values)

                    adjusted_forecast = model.predict([features])[0]
                    adjusted_forecast = max(0, adjusted_forecast)
                    forecast.append(adjusted_forecast)

                # Calculate metrics
                predicted = model.predict(X)
                metrics = ForecastingEngine.calculate_metrics(y_target, predicted)

                return np.array(forecast), metrics

        # Traditional moving average without external factors
        moving_avg = []
        for i in range(len(y)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(np.mean(y[start_idx:i+1]))

        last_avg = np.mean(y[-window:])
        forecast = np.full(periods, last_avg)

        metrics = ForecastingEngine.calculate_metrics(y[window-1:], moving_avg[window-1:])

        return forecast, metrics

    @staticmethod
    def sarima_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Actual SARIMA/SARIMAX forecasting using statsmodels"""
        from statsmodels.tsa.arima.model import ARIMA
        from pmdarima import auto_arima # type: ignore
        import warnings

        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if len(y) < 24:  # Need at least 2 seasons for SARIMA
            return ForecastingEngine.arima_forecast(data, periods)

        # Determine seasonal period based on data frequency
        # Assume monthly data, so seasonal period = 12
        seasonal_period = 12
        if len(y) < 2 * seasonal_period:
            seasonal_period = max(4, len(y) // 3)  # Quarterly or adaptive

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if external_factor_cols:
                    # SARIMAX: SARIMA with external regressors
                    print(f"Running SARIMAX with external factors: {external_factor_cols}, seasonal_period: {seasonal_period}")

                    # Prepare external regressors
                    exog = data[external_factor_cols].values

                    # Auto-select SARIMA parameters with external regressors
                    try:
                        auto_model = auto_arima(
                            y, 
                            exogenous=exog,
                            start_p=0, start_q=0, max_p=2, max_q=2, max_d=1,
                            start_P=0, start_Q=0, max_P=2, max_Q=2, max_D=1,
                            seasonal=True, m=seasonal_period,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=False
                        )

                        # Forecast external factors properly
                        future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)

                        # Create future exogenous matrix
                        future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
                        forecast = auto_model.predict(n_periods=periods, exogenous=future_exog)
                        forecast = np.maximum(forecast, 0)

                        # Calculate metrics
                        fitted = auto_model.fittedvalues()
                        if len(fitted) == len(y):
                            metrics = ForecastingEngine.calculate_metrics(y, fitted)
                        else:
                            start_idx = len(y) - len(fitted)
                            metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                        print(f"SARIMAX model: {auto_model.order}x{auto_model.seasonal_order}, AIC: {auto_model.aic():.2f}")
                        return forecast, metrics

                    except Exception as e:
                        print(f"Auto SARIMAX failed: {e}, trying manual SARIMAX")

                        # Fallback to manual SARIMAX
                        seasonal_orders = [
                            (1, 1, 1, seasonal_period),
                            (0, 1, 1, seasonal_period),
                            (1, 1, 0, seasonal_period),
                            (2, 1, 1, seasonal_period)
                        ]

                        for seasonal_order in seasonal_orders:
                            for order in [(1,1,1), (0,1,1), (1,1,0)]:
                                try:
                                    model = ARIMA(y, exog=exog, order=order, seasonal_order=seasonal_order)
                                    fitted_model = model.fit()

                                    # Use forecasted external factors
                                    future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
                                    forecast = fitted_model.forecast(steps=periods, exog=future_exog)
                                    forecast = np.maximum(forecast, 0)

                                    fitted = fitted_model.fittedvalues
                                    if len(fitted) == len(y):
                                        metrics = ForecastingEngine.calculate_metrics(y, fitted)
                                    else:
                                        start_idx = len(y) - len(fitted)
                                        metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                                    print(f"Manual SARIMAX model: {order}x{seasonal_order}, AIC: {fitted_model.aic:.2f}")
                                    return forecast, metrics

                                except:
                                    continue

                else:
                    # Traditional SARIMA without external factors
                    print(f"Running SARIMA without external factors, seasonal_period: {seasonal_period}")

                    try:
                        # Auto-select SARIMA parameters
                        auto_model = auto_arima(
                            y,
                            start_p=0, start_q=0, max_p=2, max_q=2, max_d=1,
                            start_P=0, start_Q=0, max_P=2, max_Q=2, max_D=1,
                            seasonal=True, m=seasonal_period,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=False
                        )

                        forecast = auto_model.predict(n_periods=periods)
                        forecast = np.maximum(forecast, 0)

                        # Calculate metrics
                        fitted = auto_model.fittedvalues()
                        if len(fitted) == len(y):
                            metrics = ForecastingEngine.calculate_metrics(y, fitted)
                        else:
                            start_idx = len(y) - len(fitted)
                            metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                        print(f"SARIMA model: {auto_model.order}x{auto_model.seasonal_order}, AIC: {auto_model.aic():.2f}")
                        return forecast, metrics

                    except Exception as e:
                        print(f"Auto SARIMA failed: {e}, trying manual SARIMA")

                        # Fallback to manual SARIMA
                        seasonal_orders = [
                            (1, 1, 1, seasonal_period),
                            (0, 1, 1, seasonal_period),
                            (1, 1, 0, seasonal_period),
                            (2, 1, 1, seasonal_period)
                        ]

                        for seasonal_order in seasonal_orders:
                            for order in [(1,1,1), (0,1,1), (1,1,0)]:
                                try:
                                    model = ARIMA(y, order=order, seasonal_order=seasonal_order)
                                    fitted_model = model.fit()

                                    forecast = fitted_model.forecast(steps=periods)
                                    forecast = np.maximum(forecast, 0)

                                    fitted = fitted_model.fittedvalues
                                    if len(fitted) == len(y):
                                        metrics = ForecastingEngine.calculate_metrics(y, fitted)
                                    else:
                                        start_idx = len(y) - len(fitted)
                                        metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                                    print(f"Manual SARIMA model: {order}x{seasonal_order}, AIC: {fitted_model.aic:.2f}")
                                    return forecast, metrics

                                except:
                                    continue

        except Exception as e:
            print(f"SARIMA forecasting failed: {e}")

        # Final fallback to ARIMA
        print("SARIMA failed, falling back to ARIMA")
        return ForecastingEngine.arima_forecast(data, periods)

    @staticmethod
    def prophet_like_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Actual Facebook Prophet forecasting with external regressors support"""
        from prophet import Prophet # type: ignore
        import warnings

        y = data['quantity'].values
        dates = pd.to_datetime(data['date'])
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if len(y) < 10:  # Need sufficient data for Prophet
            return ForecastingEngine.linear_regression_forecast(data, periods)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # Prepare data for Prophet (requires 'ds' and 'y' columns)
                prophet_data = pd.DataFrame({
                    'ds': dates,
                    'y': y
                })

                # Add external regressors if available
                if external_factor_cols:
                    print(f"Running Prophet with external regressors: {external_factor_cols}")
                    for col in external_factor_cols:
                        prophet_data[col] = data[col].values

                # Initialize Prophet model
                model = Prophet(
                    yearly_seasonality=True if len(y) >= 24 else False,
                    weekly_seasonality=False,  # Assuming monthly data
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    interval_width=0.8
                )

                # Add external regressors to the model
                if external_factor_cols:
                    for col in external_factor_cols:
                        model.add_regressor(col)

                # Fit the model
                model.fit(prophet_data)

                # Create future dataframe
                future = model.make_future_dataframe(periods=periods, freq='MS')  # Monthly start

                # Add external regressor values for future periods
                if external_factor_cols:
                    # Forecast external factors properly
                    future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)

                    for col in external_factor_cols:
                        # Fill historical values
                        future[col] = np.nan
                        future.loc[:len(prophet_data)-1, col] = prophet_data[col].values
                        # Use forecasted values for future periods
                        future.loc[len(prophet_data):, col] = future_factors[col]

                # Make predictions
                forecast_df = model.predict(future)

                # Extract forecast values
                forecast = forecast_df['yhat'].iloc[-periods:].values
                forecast = np.maximum(forecast, 0)

                # Calculate metrics using fitted values
                fitted = forecast_df['yhat'].iloc[:len(y)].values
                metrics = ForecastingEngine.calculate_metrics(y, fitted)

                print(f"Prophet model fitted successfully with {len(external_factor_cols)} external regressors")
                return forecast, metrics

        except Exception as e:
            print(f"Prophet forecasting failed: {e}")

            # Fallback to simplified Prophet without external regressors
            try:
                print("Trying Prophet without external regressors...")

                prophet_data = pd.DataFrame({
                    'ds': dates,
                    'y': y
                })

                model = Prophet(
                    yearly_seasonality=True if len(y) >= 24 else False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.05
                )

                model.fit(prophet_data)
                future = model.make_future_dataframe(periods=periods, freq='MS')
                forecast_df = model.predict(future)

                forecast = forecast_df['yhat'].iloc[-periods:].values
                forecast = np.maximum(forecast, 0)

                fitted = forecast_df['yhat'].iloc[:len(y)].values
                metrics = ForecastingEngine.calculate_metrics(y, fitted)

                print("Prophet model fitted successfully without external regressors")
                return forecast, metrics

            except Exception as e2:
                print(f"Prophet fallback also failed: {e2}")

        # Final fallback to linear regression
        print("Prophet failed, falling back to linear regression")
        return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def lstm_simple_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Actual LSTM forecasting using TensorFlow/Keras with external factors support"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential # type: ignore
        from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
        from tensorflow.keras.optimizers import Adam # type: ignore
        from sklearn.preprocessing import MinMaxScaler
        import warnings

        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        n = len(y)

        if n < 20:  # Need sufficient data for LSTM
            return ForecastingEngine.linear_regression_forecast(data, periods)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # Set random seeds for reproducibility
                tf.random.set_seed(42)
                np.random.seed(42)

                # Prepare data
                window_size = min(10, n // 3)  # Use larger window for LSTM

                # Scale the data
                scaler_y = MinMaxScaler()
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

                # Handle external factors
                external_data = None
                scaler_external = None
                if external_factor_cols:
                    print(f"Running LSTM with external factors: {external_factor_cols}")
                    external_data = data[external_factor_cols].values
                    scaler_external = MinMaxScaler()
                    external_data = scaler_external.fit_transform(external_data)

                # Create sequences
                X, y_target = [], []
                for i in range(window_size, n):
                    # Time series features
                    sequence = y_scaled[i-window_size:i]

                    if external_factor_cols:
                        # Add external factors to each time step
                        external_seq = external_data[i-window_size:i]
                        # Combine time series with external factors
                        combined_seq = np.column_stack([sequence.reshape(-1, 1), external_seq])
                        X.append(combined_seq)
                    else:
                        X.append(sequence.reshape(-1, 1))

                    y_target.append(y_scaled[i])

                X = np.array(X)
                y_target = np.array(y_target)

                if len(X) < 10:  # Need minimum samples for LSTM
                    return ForecastingEngine.linear_regression_forecast(data, periods)

                # Determine input shape
                if external_factor_cols:
                    input_shape = (window_size, 1 + len(external_factor_cols))
                else:
                    input_shape = (window_size, 1)

                print(f"LSTM input shape: {input_shape}, samples: {len(X)}")

                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=input_shape),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                ])

                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

                # Train the model
                history = model.fit(
                    X, y_target,
                    epochs=50,
                    batch_size=min(32, len(X) // 2),
                    verbose=0,
                    validation_split=0.2 if len(X) > 20 else 0
                )

                # Make predictions for training data (for metrics)
                predicted_scaled = model.predict(X, verbose=0).flatten()
                predicted = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
                actual_for_metrics = scaler_y.inverse_transform(y_target.reshape(-1, 1)).flatten()

                # Generate forecasts
                forecast = []
                current_sequence = X[-1].copy()  # Last sequence from training data

                for i in range(periods):
                    # Predict next value
                    next_pred_scaled = model.predict(current_sequence.reshape(1, window_size, -1), verbose=0)[0, 0]
                    next_pred = scaler_y.inverse_transform([[next_pred_scaled]])[0, 0]
                    next_pred = max(0, next_pred)
                    forecast.append(next_pred)

                    # Update sequence for next prediction
                    if external_factor_cols:
                        # Use last known external factor values
                        last_external = external_data[-1]
                        next_scaled = scaler_y.transform([[next_pred]])[0, 0]
                        next_combined = np.array([next_scaled] + list(last_external))

                        # Shift sequence and add new prediction
                        current_sequence = np.roll(current_sequence, -1, axis=0)
                        current_sequence[-1] = next_combined
                    else:
                        next_scaled = scaler_y.transform([[next_pred]])[0, 0]
                        current_sequence = np.roll(current_sequence, -1, axis=0)
                        current_sequence[-1, 0] = next_scaled

                # Calculate metrics
                metrics = ForecastingEngine.calculate_metrics(actual_for_metrics, predicted)

                print(f"LSTM model trained successfully. Final loss: {history.history['loss'][-1]:.4f}")
                return np.array(forecast), metrics

        except Exception as e:
            print(f"LSTM forecasting failed: {e}")

            # Fallback to neural network (MLP)
            try:
                print("Trying MLP as LSTM fallback...")

                window_size = min(5, n - 1)
                X, y_target = [], []

                for i in range(window_size, n):
                    features = list(y[i-window_size:i])
                    if external_factor_cols:
                        features.extend(data[external_factor_cols].iloc[i].values)
                    X.append(features)
                    y_target.append(y[i])

                X = np.array(X)
                y_target = np.array(y_target)

                if len(X) >= 3:
                    model = MLPRegressor(
                        hidden_layer_sizes=(50, 25), 
                        max_iter=500, 
                        random_state=42, 
                        alpha=0.01
                    )
                    model.fit(X, y_target)

                    # Forecast
                    forecast = []
                    recent_values = list(y[-window_size:])

                    for _ in range(periods):
                        features = list(recent_values)
                        if external_factor_cols:
                            features.extend(data[external_factor_cols].iloc[-1].values)

                        next_pred = model.predict([features])[0]
                        next_pred = max(0, next_pred)
                        forecast.append(next_pred)
                        recent_values = recent_values[1:] + [next_pred]

                    predicted = model.predict(X)
                    metrics = ForecastingEngine.calculate_metrics(y_target, predicted)

                    print("MLP fallback successful")
                    return np.array(forecast), metrics

            except Exception as e2:
                print(f"MLP fallback also failed: {e2}")

        # Final fallback to linear regression
        print("LSTM and MLP failed, falling back to linear regression")
        return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def xgboost_forecast(data: pd.DataFrame, periods: int, n_estimators_list: list = [50, 100], learning_rate_list: list = [0.05, 0.1, 0.2], max_depth_list: list = [3, 4, 5]) -> tuple:
        """XGBoost-like forecasting with hyperparameter tuning and external factors"""
        y = data['quantity'].values
        dates = pd.to_datetime(data['date'])
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        if n < 6:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        features = []
        targets = []
        window = min(4, n - 1)
        for i in range(window, n):
            lags = list(y[i-window:i])
            date = dates.iloc[i]
            time_features = [
                i,
                date.month,
                date.quarter,
                date.dayofyear % 7,
                i % 12,
            ]
            recent_mean = np.mean(y[max(0, i-3):i])
            recent_std = np.std(y[max(0, i-3):i]) if i > 3 else 0
            feature_vector = lags + time_features + [recent_mean, recent_std]
            if external_factor_cols:
                feature_vector.extend(data[external_factor_cols].iloc[i].values)
            features.append(feature_vector)
            targets.append(y[i])
        if len(features) < 3:
            return ForecastingEngine.random_forest_forecast(data, periods)
        features = np.array(features)
        targets = np.array(targets)
        best_metrics = None
        best_forecast = None
        for n_estimators in n_estimators_list:
            for learning_rate in learning_rate_list:
                for max_depth in max_depth_list:
                    print(f"Running XGBoost with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
                    try:
                        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                        model.fit(features, targets)
                        forecast = []
                        recent_values = list(y[-window:])
                        last_date = dates.iloc[-1]
                        for i in range(periods):
                            next_date = last_date + pd.DateOffset(months=i+1)
                            time_features = [
                                n + i,
                                next_date.month,
                                next_date.quarter,
                                next_date.dayofyear % 7,
                                (n + i) % 12,
                            ]
                            recent_mean = np.mean(recent_values[-3:])
                            recent_std = np.std(recent_values[-3:]) if len(recent_values) > 1 else 0
                            feature_vector = recent_values + time_features + [recent_mean, recent_std]
                            if external_factor_cols:
                                last_factors = data[external_factor_cols].iloc[-1].values
                                feature_vector = list(feature_vector) + list(last_factors)
                            next_pred = model.predict([feature_vector])[0]
                            next_pred = max(0, next_pred)
                            forecast.append(next_pred)
                            recent_values = recent_values[1:] + [next_pred]
                        predicted = model.predict(features)
                        metrics = ForecastingEngine.calculate_metrics(targets, predicted)
                        print(f"n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, Accuracy={metrics['accuracy']:.2f}")
                        if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                            best_metrics = metrics
                            best_forecast = forecast
                    except Exception as e:
                        print(f"Error running XGBoost with params: {e}")
                        continue
        return np.array(best_forecast), best_metrics

    @staticmethod
    def svr_forecast(data: pd.DataFrame, periods: int, C_list: list = [1, 10, 100], epsilon_list: list = [0.1, 0.2]) -> tuple:
        """Support Vector Regression forecasting with hyperparameter tuning and external factors"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        if n < 4:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        window = min(3, n - 1)
        X, y_target = [], []
        for i in range(window, n):
            features = [i] + list(y[i-window:i])
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[i].values)
            X.append(features)
            y_target.append(y[i])
        if len(X) < 2:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        X = np.array(X)
        y_target = np.array(y_target)
        X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_norm = (X - X_mean) / X_std
        param_grid = {'C': C_list, 'epsilon': epsilon_list}
        model = SVR(kernel='rbf', gamma='scale')
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_norm, y_target)
        best_model = grid_search.best_estimator_
        print(f"Best SVR params: {grid_search.best_params_}")
        forecast = []
        recent_values = list(y[-window:])
        # Forecast external factors if needed
        if external_factor_cols:
            future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)

        for i in range(periods):
            features = [n + i] + recent_values
            if external_factor_cols:
                forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
                features.extend(forecasted_factors)
            features_norm = (np.array(features) - X_mean) / X_std
            next_pred = best_model.predict([features_norm])[0]
            next_pred = max(0, next_pred)
            forecast.append(next_pred)
            recent_values = recent_values[1:] + [next_pred]
        predicted = best_model.predict(X_norm)
        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
        return np.array(forecast), metrics

    @staticmethod
    def knn_forecast(data: pd.DataFrame, periods: int, n_neighbors_list: list = [7, 10]) -> tuple:
        """K-Nearest Neighbors forecasting with hyperparameter tuning and external factors"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        if n < 6:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        window = min(4, n - 1)
        X, y_target = [], []
        for i in range(window, n):
            features = list(y[i-window:i])
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[i].values)
            X.append(features)
            y_target.append(y[i])
        if len(X) < 3:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        X = np.array(X)
        y_target = np.array(y_target)
        param_grid = {'n_neighbors': n_neighbors_list}
        model = KNeighborsRegressor(weights='distance')
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X, y_target)
        best_model = grid_search.best_estimator_
        print(f"Best KNN params: {grid_search.best_params_}")
        forecast = []
        current_window = list(y[-window:])

        # Forecast external factors if needed
        if external_factor_cols:
            future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)

        for i in range(periods):
            features = list(current_window)
            if external_factor_cols:
                forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
                features.extend(forecasted_factors)
            next_pred = best_model.predict([features])[0]
            next_pred = max(0, next_pred)
            forecast.append(next_pred)
            current_window = current_window[1:] + [next_pred]
        predicted = best_model.predict(X)
        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
        return np.array(forecast), metrics

    @staticmethod
    def gaussian_process_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Improved Gaussian Process Regression forecasting with hyperparameter tuning and scaling"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV
        y = data['quantity'].values
        n = len(y)

        if n < 4:
            return ForecastingEngine.linear_regression_forecast(data, periods)

        # Create time features
        X = np.arange(n).reshape(-1, 1)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            # Define kernel with initial parameters
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

            # Create GP model
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=42, normalize_y=True)

            # Hyperparameter tuning for kernel parameters
            param_grid = {
                "kernel__k1__constant_value": [0.1, 1, 10],
                "kernel__k2__length_scale": [0.1, 1, 10]
            }
            grid_search = GridSearchCV(gp, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_scaled, y)
            best_model = grid_search.best_estimator_

            # Forecast
            future_X = np.arange(n, n + periods).reshape(-1, 1)
            future_X_scaled = scaler.transform(future_X)
            forecast, _ = best_model.predict(future_X_scaled, return_std=True)
            forecast = np.maximum(forecast, 0)

            # Calculate metrics
            predicted, _ = best_model.predict(X_scaled, return_std=True)
            metrics = ForecastingEngine.calculate_metrics(y, predicted)

        except Exception as e:
            print(f"Error in Gaussian Process forecasting: {e}")
            return ForecastingEngine.linear_regression_forecast(data, periods)

        return forecast, metrics

    @staticmethod
    def neural_network_forecast(data: pd.DataFrame, periods: int, hidden_layer_sizes_list: list = [(10,), (20, 10)], alpha_list: list = [0.001, 0.01]) -> tuple:
        """Multi-layer Perceptron Neural Network forecasting with hyperparameter tuning and external factors"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        if n < 6:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        window = min(5, n - 1)
        X, y_target = [], []
        for i in range(window, n):
            lags = list(y[i-window:i])
            trend = i / n
            seasonal = np.sin(2 * np.pi * i / 12)
            features = lags + [trend, seasonal]
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[i].values)
            X.append(features)
            y_target.append(y[i])
        if len(X) < 3:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        X = np.array(X)
        y_target = np.array(y_target)
        param_grid = {'hidden_layer_sizes': hidden_layer_sizes_list, 'alpha': alpha_list}
        model = MLPRegressor(activation='relu', solver='adam', max_iter=1000, random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X, y_target)
        best_model = grid_search.best_estimator_
        print(f"Best MLP params: {grid_search.best_params_}")
        forecast = []
        recent_values = list(y[-window:])
        for i in range(periods):
            trend = (n + i) / n
            seasonal = np.sin(2 * np.pi * (n + i) / 12)
            features = recent_values + [trend, seasonal]
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[-1].values)
            next_pred = best_model.predict([features])[0]
            next_pred = max(0, next_pred)
            forecast.append(next_pred)
            recent_values = recent_values[1:] + [next_pred]
        predicted = best_model.predict(X)
        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
        return np.array(forecast), metrics

    @staticmethod
    def theta_method_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Enhanced Theta method with external factors integration"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if n < 3:
            return ForecastingEngine.linear_regression_forecast(data, periods)

        # Decompose series into two theta lines
        theta1 = 0
        theta2 = 2

        t = np.arange(n)
        mean_y = np.mean(y)
        theta_line1 = mean_y + (y - mean_y) * theta1
        theta_line2 = mean_y + (y - mean_y) * theta2

        if external_factor_cols:
            # Enhanced theta method with external factors
            # Forecast theta_line1 with external factors
            X1 = np.column_stack([t, data[external_factor_cols].values])
            model1 = LinearRegression()
            model1.fit(X1, theta_line1)

            future_external = np.tile(data[external_factor_cols].iloc[-1].values, (periods, 1))
            future_t = np.arange(n, n + periods)
            future_X1 = np.column_stack([future_t, future_external])
            forecast1 = model1.predict(future_X1)

            # Forecast theta_line2 with external factor-adjusted smoothing
            window = min(3, n - 1)
            if window > 0:
                X2, y2_target = [], []
                for i in range(window, n):
                    features = list(theta_line2[i-window:i])  # Historical smoothed values
                    features.extend(data[external_factor_cols].iloc[i].values)
                    X2.append(features)
                    y2_target.append(theta_line2[i])

                if len(X2) > 1:
                    X2 = np.array(X2)
                    y2_target = np.array(y2_target)

                    model2 = LinearRegression()
                    model2.fit(X2, y2_target)

                    # Forecast theta_line2
                    forecast2 = []
                    recent_smoothed = list(theta_line2[-window:])

                    for i in range(periods):
                        features = list(recent_smoothed)
                        features.extend(data[external_factor_cols].iloc[-1].values)

                        next_smoothed = model2.predict([features])[0]
                        forecast2.append(next_smoothed)
                        recent_smoothed = recent_smoothed[1:] + [next_smoothed]

                    forecast2 = np.array(forecast2)
                else:
                    # Fallback to simple smoothing
                    alpha = 0.3
                    smoothed = [theta_line2[0]]
                    for i in range(1, n):
                        smoothed.append(alpha * theta_line2[i] + (1 - alpha) * smoothed[i-1])
                    forecast2 = np.full(periods, smoothed[-1])
            else:
                # Simple smoothing fallback
                alpha = 0.3
                smoothed = [theta_line2[0]]
                for i in range(1, n):
                    smoothed.append(alpha * theta_line2[i] + (1 - alpha) * smoothed[i-1])
                forecast2 = np.full(periods, smoothed[-1])

            # Combine forecasts
            forecast = (forecast1 + forecast2) / 2
            forecast = np.maximum(forecast, 0)

            # Calculate metrics
            fitted1 = model1.predict(X1)
            if len(X2) > 1:
                fitted2_partial = model2.predict(X2)
                fitted2 = np.full(n, np.mean(theta_line2))
                fitted2[window:] = fitted2_partial
            else:
                fitted2 = theta_line2

            fitted = (fitted1 + fitted2) / 2
            metrics = ForecastingEngine.calculate_metrics(y, fitted)

            return forecast, metrics

        # Traditional theta method without external factors
        X = t.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, theta_line1)
        future_t = np.arange(n, n + periods).reshape(-1, 1)
        forecast1 = model.predict(future_t)

        alpha = 0.3
        smoothed = [theta_line2[0]]
        for i in range(1, n):
            smoothed.append(alpha * theta_line2[i] + (1 - alpha) * smoothed[i-1])
        last_smoothed = smoothed[-1]
        forecast2 = np.full(periods, last_smoothed)

        forecast = (forecast1 + forecast2) / 2
        forecast = np.maximum(forecast, 0)

        fitted = (model.predict(X) + smoothed) / 2
        metrics = ForecastingEngine.calculate_metrics(y, fitted)

        return forecast, metrics

    @staticmethod
    def croston_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Enhanced Croston's method with external factors for intermittent demand"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if n < 3:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

        # Identify non-zero demands
        non_zero_indices = np.where(y > 0)[0]

        if len(non_zero_indices) < 2:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

        if external_factor_cols and len(non_zero_indices) > 3:
            # Enhanced Croston with external factors
            demand_sizes = y[non_zero_indices]
            intervals = np.diff(non_zero_indices)

            # Model demand sizes with external factors
            if len(demand_sizes) > 1:
                X_demand, y_demand = [], []
                for i, idx in enumerate(non_zero_indices[1:], 1):
                    features = [demand_sizes[i-1]]  # Previous demand size
                    features.extend(data[external_factor_cols].iloc[idx].values)
                    X_demand.append(features)
                    y_demand.append(demand_sizes[i])

                if len(X_demand) > 1:
                    X_demand = np.array(X_demand)
                    y_demand = np.array(y_demand)

                    demand_model = LinearRegression()
                    demand_model.fit(X_demand, y_demand)

                    # Model intervals with external factors
                    if len(intervals) > 1:
                        X_interval, y_interval = [], []
                        for i in range(1, len(intervals)):
                            idx = non_zero_indices[i+1]
                            features = [intervals[i-1]]  # Previous interval
                            features.extend(data[external_factor_cols].iloc[idx].values)
                            X_interval.append(features)
                            y_interval.append(intervals[i])

                        if len(X_interval) > 1:
                            X_interval = np.array(X_interval)
                            y_interval = np.array(y_interval)

                            interval_model = LinearRegression()
                            interval_model.fit(X_interval, y_interval)

                            # Forecast with external factors
                            last_demand = demand_sizes[-1]
                            last_interval = intervals[-1] if len(intervals) > 0 else 1

                            forecast = []
                            for i in range(periods):
                                # Predict demand size
                                demand_features = [last_demand]
                                demand_features.extend(data[external_factor_cols].iloc[-1].values)
                                predicted_demand = demand_model.predict([demand_features])[0]
                                predicted_demand = max(0, predicted_demand)

                                # Predict interval
                                interval_features = [last_interval]
                                interval_features.extend(data[external_factor_cols].iloc[-1].values)
                                predicted_interval = interval_model.predict([interval_features])[0]
                                predicted_interval = max(1, predicted_interval)

                                # Calculate forecast
                                forecast_value = predicted_demand / predicted_interval
                                forecast.append(max(0, forecast_value))

                                # Update for next iteration
                                last_demand = predicted_demand
                                last_interval = predicted_interval

                            # Calculate metrics
                            predicted_demands = demand_model.predict(X_demand)
                            predicted_intervals = interval_model.predict(X_interval)

                            # Simplified metrics calculation
                            avg_predicted = np.mean(predicted_demands / predicted_intervals)
                            predicted_full = np.full(n, avg_predicted)
                            metrics = ForecastingEngine.calculate_metrics(y, predicted_full)

                            return np.array(forecast), metrics

        # Traditional Croston method without external factors
        intervals = np.diff(non_zero_indices)
        if len(intervals) == 0:
            intervals = [1]

        alpha = 0.3

        demand_sizes = y[non_zero_indices]
        smoothed_demand = demand_sizes[0]
        for demand in demand_sizes[1:]:
            smoothed_demand = alpha * demand + (1 - alpha) * smoothed_demand

        smoothed_interval = intervals[0]
        for interval in intervals[1:]:
            smoothed_interval = alpha * interval + (1 - alpha) * smoothed_interval

        forecast_demand = smoothed_demand / smoothed_interval
        forecast = np.full(periods, max(0, forecast_demand))

        avg_demand = np.mean(y[y > 0]) if np.any(y > 0) else 0
        predicted = np.full(n, avg_demand / max(1, np.mean(intervals)))
        metrics = ForecastingEngine.calculate_metrics(y, predicted)

        return forecast, metrics


    @staticmethod
    def ses_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Enhanced Simple Exponential Smoothing with external factors"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if n < 2:
            return np.full(periods, y[0] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': 0, 'rmse': 0}

        if external_factor_cols and n > 5:
            # SES with external factor enhancement
            alphas = [0.1, 0.3, 0.5, 0.7]
            best_metrics = None
            best_forecast = None

            for alpha in alphas:
                # Traditional SES for base smoothing
                smoothed = [y[0]]
                for i in range(1, n):
                    smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[i-1])

                # Use external factors to adjust smoothed values
                window = min(3, n - 1)
                X, y_target = [], []

                for i in range(window, n):
                    features = [smoothed[i]]  # Base smoothed value
                    features.extend(data[external_factor_cols].iloc[i].values)
                    X.append(features)
                    y_target.append(y[i])

                if len(X) > 1:
                    X = np.array(X)
                    y_target = np.array(y_target)

                    # Model to adjust SES with external factors
                    model = LinearRegression()
                    model.fit(X, y_target)

                    # Forecast
                    forecast = []
                    current_smoothed = smoothed[-1]

                    for i in range(periods):
                        features = [current_smoothed]
                        features.extend(data[external_factor_cols].iloc[-1].values)

                        adjusted_forecast = model.predict([features])[0]
                        adjusted_forecast = max(0, adjusted_forecast)
                        forecast.append(adjusted_forecast)

                        # Update smoothed value for next prediction
                        current_smoothed = alpha * adjusted_forecast + (1 - alpha) * current_smoothed

                    # Calculate metrics
                    predicted = model.predict(X)
                    metrics = ForecastingEngine.calculate_metrics(y_target, predicted)

                    if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                        best_metrics = metrics
                        best_forecast = forecast

            if best_forecast is not None:
                return np.array(best_forecast), best_metrics

        # Traditional SES without external factors using statsmodels
        try:
            import warnings
            best_model = None
            best_aic = float('inf')
            best_forecast = None

            seasonal_periods_options = [None, 4, 6, 12]
            trend_options = [None, 'add', 'mul']
            seasonal_options = [None, 'add', 'mul']

            for seasonal_periods in seasonal_periods_options:
                for trend in trend_options:
                    for seasonal in seasonal_options:
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                model = ExponentialSmoothing(
                                    y,
                                    trend=trend,
                                    seasonal=seasonal,
                                    seasonal_periods=seasonal_periods,
                                    initialization_method="estimated"
                                )
                                fit = model.fit(optimized=True)
                            aic = fit.aic
                            if aic < best_aic:
                                best_aic = aic
                                best_model = fit
                                best_forecast = fit.forecast(periods)
                        except Exception:
                            continue

            if best_forecast is not None:
                forecast = np.maximum(best_forecast, 0)
                fitted = best_model.fittedvalues
                metrics = ForecastingEngine.calculate_metrics(y, fitted)
                return forecast, metrics
        except:
            pass

        # Simple fallback SES
        alpha = 0.3
        smoothed = [y[0]]
        for i in range(1, n):
            smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[i-1])

        forecast = np.full(periods, smoothed[-1])
        metrics = ForecastingEngine.calculate_metrics(y[1:], smoothed[1:])

        return forecast, metrics

    @staticmethod
    def damped_trend_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Damped trend exponential smoothing"""
        y = data['quantity'].values
        n = len(y)

        if n < 3:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

        # Parameters
        alpha, beta, phi = 0.3, 0.1, 0.8  # phi is damping parameter

        # Initialize
        level = y[0]
        trend = y[1] - y[0] if n > 1 else 0

        levels = [level]
        trends = [trend]
        fitted = [level + trend]

        # Apply damped trend smoothing
        for i in range(1, n):
            level = alpha * y[i] + (1 - alpha) * (levels[-1] + phi * trends[-1])
            trend = beta * (level - levels[-1]) + (1 - beta) * phi * trends[-1]

            levels.append(level)
            trends.append(trend)
            fitted.append(level + trend)

        # Forecast with damping
        forecast = []
        for h in range(1, periods + 1):
            damped_trend = trend * sum(phi**i for i in range(1, h + 1))
            forecast_value = level + damped_trend
            forecast.append(max(0, forecast_value))

        # Calculate metrics
        metrics = ForecastingEngine.calculate_metrics(y[1:], fitted[1:])

        return np.array(forecast), metrics

    @staticmethod
    def naive_seasonal_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Naive seasonal forecasting"""
        y = data['quantity'].values
        n = len(y)

        if n < 2:
            return np.full(periods, y[0] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': 0, 'rmse': 0}

        # Determine season length
        season_length = min(12, n)

        # Forecast by repeating seasonal pattern
        forecast = []
        for i in range(periods):
            seasonal_index = (n + i) % season_length
            if seasonal_index < n:
                forecast.append(y[-(season_length - seasonal_index)])
            else:
                forecast.append(y[-1])

        # Calculate metrics using naive forecast on historical data
        if n > season_length:
            predicted = []
            for i in range(season_length, n):
                predicted.append(y[i - season_length])
            actual = y[season_length:]
            metrics = ForecastingEngine.calculate_metrics(actual, predicted)
        else:
            metrics = {'accuracy': 60.0, 'mae': np.std(y), 'rmse': np.std(y)}

        return np.array(forecast), metrics

    @staticmethod
    def drift_method_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Improved Drift method forecasting with linear regression trend"""
        y = data['quantity'].values
        n = len(y)

        if n < 2:
            return np.full(periods, y[0] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': 0, 'rmse': 0}

        # Use linear regression to estimate trend and intercept
        X = np.arange(n).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)

        # Forecast
        future_X = np.arange(n, n + periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        forecast = np.maximum(forecast, 0)

        # Calculate metrics on training data
        predicted = model.predict(X)
        metrics = ForecastingEngine.calculate_metrics(y, predicted)

        return forecast, metrics

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

            # Check for cached model
            training_data = data['quantity'].values
            if training_data is not None and len(training_data) > 0:
                model_hash = ModelPersistenceManager.find_cached_model(db, algorithm, config.dict(), training_data)
            else:
                model_hash = None

            if model_hash:
                print(f"Using cached model for {algorithm}")
                cached_model = ModelPersistenceManager.load_model(db, model_hash)
                if cached_model:
                    # Use cached model for prediction
                    # Note: This is a simplified example - in practice, you'd need to adapt this
                    # based on the specific algorithm and model type
                    pass

            # Time-based train/test split for realistic metrics
            train, test = ForecastingEngine.time_based_split(data, test_ratio=0.2)

            # Train model on train set
            model = None
            if algorithm == "linear_regression":
                forecast, metrics = ForecastingEngine.linear_regression_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = LinearRegression().fit(np.arange(len(train)).reshape(-1, 1), train['quantity'].values)
            elif algorithm == "polynomial_regression":
                forecast, metrics = ForecastingEngine.polynomial_regression_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "exponential_smoothing":
                forecast, metrics = ForecastingEngine.exponential_smoothing_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "holt_winters":
                forecast, metrics = ForecastingEngine.holt_winters_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "arima":
                forecast, metrics = ForecastingEngine.arima_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "random_forest":
                forecast, metrics = ForecastingEngine.random_forest_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "seasonal_decomposition":
                forecast, metrics = ForecastingEngine.seasonal_decomposition_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "moving_average":
                forecast, metrics = ForecastingEngine.moving_average_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "sarima":
                forecast, metrics = ForecastingEngine.sarima_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "prophet_like":
                forecast, metrics = ForecastingEngine.prophet_like_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "lstm_like":
                forecast, metrics = ForecastingEngine.lstm_simple_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "xgboost":
                forecast, metrics = ForecastingEngine.xgboost_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "svr":
                forecast, metrics = ForecastingEngine.svr_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "knn":
                forecast, metrics = ForecastingEngine.knn_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "gaussian_process":
                forecast, metrics = ForecastingEngine.gaussian_process_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "neural_network":
                forecast, metrics = ForecastingEngine.neural_network_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "theta_method":
                forecast, metrics = ForecastingEngine.theta_method_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "croston":
                forecast, metrics = ForecastingEngine.croston_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "ses":
                forecast, metrics = ForecastingEngine.ses_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "damped_trend":
                forecast, metrics = ForecastingEngine.damped_trend_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "naive_seasonal":
                forecast, metrics = ForecastingEngine.naive_seasonal_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "drift_method":
                forecast, metrics = ForecastingEngine.drift_method_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # Save model/configuration to cache
            if ENABLE_MODEL_CACHE and save_model:
                try:
                    print(f"Debug run_algorithm: Saving model/configuration for algorithm {algorithm}")
                    ModelPersistenceManager.save_model(
                        db, model, algorithm, config.dict(), 
                        training_data, metrics, {'data_shape': training_data.shape}
                    )
                    print(f"Debug run_algorithm: Model/configuration saved successfully for algorithm {algorithm}")
                except Exception as e:
                    print(f"Failed to save model/configuration to cache: {e}")

            # Compute test metrics
            if test is not None and len(test) > 0:
                actual = test['quantity'].values
                predicted = forecast[:len(test)]
                metrics = ForecastingEngine.calculate_metrics(actual, predicted)
            else:
                # Fallback to training metrics
                y = train['quantity'].values
                x = np.arange(len(y)).reshape(-1, 1)
                if algorithm == "linear_regression":
                    model = LinearRegression().fit(x, y)
                    predicted = model.predict(x)
                elif algorithm == "polynomial_regression":
                    coeffs = np.polyfit(np.arange(len(y)), y, 2)
                    poly_func = np.poly1d(coeffs)
                    predicted = poly_func(np.arange(len(y)))
                elif algorithm == "exponential_smoothing" or algorithm == "ses":
                    # Use simple exponential smoothing for fallback
                    alpha = 0.3
                    smoothed = [y[0]]
                    for i in range(1, len(y)):
                        smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[i-1])
                    predicted = smoothed
                else:
                    predicted = y
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

        if config.algorithm in ["best_fit", "best_statistical", "best_ml", "best_specialized"]:
            if process_log is not None:
                if config.algorithm == "best_fit":
                    process_log.append("Running best fit algorithm selection...")
                elif config.algorithm == "best_statistical":
                    process_log.append("Running best statistical method selection...")
                elif config.algorithm == "best_ml":
                    process_log.append("Running best machine learning method selection...")
                elif config.algorithm == "best_specialized":
                    process_log.append("Running best specialized method selection...")

            # Define algorithm categories
            statistical_algorithms = [
                "linear_regression", "polynomial_regression", "exponential_smoothing", 
                "holt_winters", "arima", "sarima", "ses", "damped_trend", 
                "theta_method", "drift_method", "naive_seasonal", "prophet_like"
            ]

            ml_algorithms = [
                "random_forest", "xgboost", "svr", "knn", "gaussian_process", 
                "neural_network", "lstm_like"
            ]

            specialized_algorithms = [
                "seasonal_decomposition", "moving_average", "croston"
            ]

            # Select algorithms based on category
            if config.algorithm == "best_statistical":
                algorithms = statistical_algorithms
            elif config.algorithm == "best_ml":
                algorithms = ml_algorithms
            elif config.algorithm == "best_specialized":
                algorithms = specialized_algorithms
            else:  # best_fit
                algorithms = [alg for alg in ForecastingEngine.ALGORITHMS.keys() 
                            if alg not in ["best_fit", "best_statistical", "best_ml", "best_specialized"]]

            algorithm_results = []
            best_model = None
            best_algorithm = None
            best_metrics = None

            # Use ThreadPoolExecutor for parallel execution
            max_workers = min(len(algorithms), os.cpu_count() or 4)
            if process_log is not None:
                process_log.append(f"Starting parallel execution with {max_workers} workers for {len(algorithms)} algorithms...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all algorithm tasks
                future_to_algorithm = {
                    executor.submit(ForecastingEngine.run_algorithm, algorithm, aggregated_df, config, save_model=False): algorithm
                    for algorithm in algorithms
                }

                # Collect results as they complete
                for future in as_completed(future_to_algorithm):
                    algorithm_name = future_to_algorithm[future]
                    try:
                        result = future.result()
                        algorithm_results.append(result)
                        if process_log is not None:
                            process_log.append(f"✅ Algorithm {algorithm_name} completed with accuracy: {result.accuracy:.2f}%")

                        # Track best performing algorithm
                        if best_metrics is None or result.accuracy > best_metrics['accuracy']:
                            best_metrics = {
                                'accuracy': result.accuracy,
                                'mae': result.mae,
                                'rmse': result.rmse
                            }
                            best_model = result
                            best_algorithm = algorithm_name

                    except Exception as exc:
                        if process_log is not None:
                            process_log.append(f"❌ Algorithm {algorithm_name} failed: {str(exc)}")
                        # Create a dummy result for failed algorithms to maintain structure
                        algorithm_results.append(AlgorithmResult(
                            algorithm=ForecastingEngine.ALGORITHMS[algorithm_name],
                            accuracy=0.0,
                            mae=999.0,
                            rmse=999.0,
                            historicData=[],
                            forecastData=[],
                            trend='stable'
                        ))

            # Filter out failed results for ensemble calculation
            successful_results = [res for res in algorithm_results if res.accuracy > 0]
            if not successful_results:
                raise ValueError("All algorithms failed to produce valid results")

            if process_log is not None:
                process_log.append(f"Parallel execution completed. {len(successful_results)} algorithms succeeded, {len(algorithm_results) - len(successful_results)} failed.")

            if not algorithm_results:
                raise ValueError("No algorithms produced valid results")

            # Ensemble: average forecast of top 3 algorithms by accuracy
            top3 = sorted(successful_results, key=lambda x: -x.accuracy)[:3]
            if len(top3) >= 2:
                if process_log is not None:
                    process_log.append(f"Creating ensemble from top {len(top3)} algorithms...")

                n_forecast = len(top3[0].forecastData) if top3[0].forecastData else 0
                avg_forecast = []
                for i in range(n_forecast):
                    quantities = [algo.forecastData[i].quantity for algo in top3 if algo.forecastData]
                    if not quantities:
                        avg_qty = 0
                    else:
                        avg_qty = np.mean(quantities)
                    avg_forecast.append(DataPoint(
                        date=top3[0].forecastData[i].date,
                        quantity=avg_qty,
                        period=top3[0].forecastData[i].period
                    ))

                ensemble_result = AlgorithmResult(
                    algorithm="Ensemble (Top 3 Avg)",
                    accuracy=np.mean([algo.accuracy for algo in top3]),
                    mae=np.mean([algo.mae for algo in top3]),
                    rmse=np.mean([algo.rmse for algo in top3]),
                    historicData=top3[0].historicData,
                    forecastData=avg_forecast,
                    trend=top3[0].trend
                )
                algorithm_results.append(ensemble_result)

            # Save the best_fit model configuration
            try:
                if process_log is not None:
                    process_log.append(f"Saving best fit model configuration (best algorithm: {best_algorithm})...")

                ModelPersistenceManager.save_model(
                    db, None, "best_fit", config.dict(),
                    aggregated_df['quantity'].values, 
                    {'accuracy': ensemble_result.accuracy, 'mae': ensemble_result.mae, 'rmse': ensemble_result.rmse},
                    {'data_shape': aggregated_df.shape, 'best_algorithm': best_algorithm}
                )
            except Exception as e:
                print(f"Failed to save best_fit model: {e}")
                if process_log is not None:
                    process_log.append(f"Warning: Failed to save best_fit model: {e}")

            # Generate config hash for tracking
            config_hash = ModelPersistenceManager.generate_config_hash(config.dict())

            return ForecastResult(
                selectedAlgorithm=f"{best_model.algorithm} (Best Fit)",
                accuracy=best_model.accuracy,
                mae=best_model.mae,
                rmse=best_model.rmse,
                historicData=best_model.historicData,
                forecastData=best_model.forecastData,
                trend=best_model.trend,
                allAlgorithms=algorithm_results,
                configHash=config_hash,
                processLog=process_log
            )
        else:
            if process_log is not None:
                process_log.append(f"Running algorithm: {config.algorithm}")

            result = ForecastingEngine.run_algorithm(config.algorithm, aggregated_df, config, save_model=True)
            config_hash = ModelPersistenceManager.generate_config_hash(config.dict())
            return ForecastResult(
                selectedAlgorithm=result.algorithm,
                combination=None,
                accuracy=result.accuracy,
                mae=result.mae,
                rmse=result.rmse,
                historicData=result.historicData,
                forecastData=result.forecastData,
                trend=result.trend,
                configHash=config_hash,
                processLog=process_log
            )

    @staticmethod
    def generate_forecast_three_dimensions(db: Session, config: ForecastConfig, products: list, customers: list, locations: list, process_log: List[str] = None) -> MultiForecastResult:
        """Generate forecasts for all three dimensions selected"""
        from itertools import product as itertools_product
        combinations = [(p, c, l) for p, c, l in itertools_product(products, customers, locations)]

        if process_log is not None:
            process_log.append(f"Generated {len(combinations)} combinations for three dimensions")

        results = []
        successful_combinations = 0
        failed_combinations = []

        for product, customer, location in combinations:
            try:
                if process_log is not None:
                    process_log.append(f"Processing combination: {product} + {customer} + {location}")

                # Load data for this specific combination
                df = ForecastingEngine.load_data_for_combination(db, product, customer, location)
                aggregated_df = ForecastingEngine.aggregate_by_period(df, config.interval, config)

                if len(aggregated_df) < 2:
                    if process_log is not None:
                        process_log.append("Insufficient data for forecasting")

                    failed_combinations.append({
                        'combination': f"{product} + {customer} + {location}",
                        'error': 'Insufficient data'
                    })
                    continue

                if config.algorithm in ["best_fit", "best_statistical", "best_ml", "best_specialized"]:
                    if process_log is not None:
                        process_log.append(f"Running {config.algorithm} algorithm selection...")
                    
                    # Define algorithm categories
                    statistical_algorithms = [
                        "linear_regression", "polynomial_regression", "exponential_smoothing", 
                        "holt_winters", "arima", "sarima", "ses", "damped_trend", 
                        "theta_method", "drift_method", "naive_seasonal", "prophet_like"
                    ]

                    ml_algorithms = [
                        "random_forest", "xgboost", "svr", "knn", "gaussian_process", 
                        "neural_network", "lstm_like"
                    ]

                    specialized_algorithms = [
                        "seasonal_decomposition", "moving_average", "croston"
                    ]

                    # Select algorithms based on category
                    if config.algorithm == "best_statistical":
                        algorithms = statistical_algorithms
                    elif config.algorithm == "best_ml":
                        algorithms = ml_algorithms
                    elif config.algorithm == "best_specialized":
                        algorithms = specialized_algorithms
                    else:  # best_fit
                        algorithms = [alg for alg in ForecastingEngine.ALGORITHMS.keys() 
                                    if alg not in ["best_fit", "best_statistical", "best_ml", "best_specialized"]]
                    algorithm_results = []

                    # Use parallel execution for three-dimension best fit
                    max_workers = min(len(algorithms), os.cpu_count() or 4)
                    if process_log is not None:
                        process_log.append(f"Running parallel best fit with {max_workers} workers...")

                    # Create a copy of config for each algorithm
                    single_config = ForecastConfig(
                        forecastBy=config.forecastBy,
                        selectedProducts=[product],
                        selectedCustomers=[customer],
                        selectedLocations=[location],
                        algorithm=config.algorithm,
                        interval=config.interval,
                        historicPeriod=config.historicPeriod,
                        forecastPeriod=config.forecastPeriod,
                        multiSelect=False,
                        externalFactors=config.externalFactors
                    )

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_algorithm = {
                            executor.submit(ForecastingEngine.run_algorithm, algorithm, aggregated_df, single_config, save_model=False): algorithm
                            for algorithm in algorithms
                        }

                        for future in as_completed(future_to_algorithm):
                            algorithm_name = future_to_algorithm[future]
                            try:
                                result = future.result()
                                algorithm_results.append(result)
                                if process_log is not None:
                                    process_log.append(f"✅ Algorithm {algorithm_name} completed for {product}+{customer}+{location}")
                            except Exception as exc:
                                if process_log is not None:
                                    process_log.append(f"❌ Algorithm {algorithm_name} failed for {product}+{customer}+{location}: {str(exc)}")
                                continue

                    if not algorithm_results:
                        if process_log is not None:
                            process_log.append("No algorithms produced valid results")

                        failed_combinations.append({
                            'combination': f"{product} + {customer} + {location}",
                            'error': 'No algorithms produced valid results'
                        })
                        continue
                    best_result = max(algorithm_results, key=lambda x: x.accuracy)
                    forecast_result = ForecastResult(
                    combination={"product": product, "customer": customer, "location": location},
                    selectedAlgorithm=f"{best_result.algorithm} (Best Fit)",
                    accuracy=best_result.accuracy,
                    mae=best_result.mae,
                    rmse=best_result.rmse,
                    historicData=best_result.historicData,
                    forecastData=best_result.forecastData,
                    trend=best_result.trend,
                    allAlgorithms=algorithm_results,
                    processLog=process_log if process_log is not None else []
                )
                else:
                    if process_log is not None:
                        process_log.append(f"Running algorithm: {config.algorithm}")

                    result = ForecastingEngine.run_algorithm(config.algorithm, aggregated_df, config)
                    forecast_result = ForecastResult(
                        combination={"product": product, "customer": customer, "location": location},
                        selectedAlgorithm=result.algorithm,
                        accuracy=result.accuracy,
                        mae=result.mae,
                        rmse=result.rmse,
                        historicData=result.historicData,
                        forecastData=result.forecastData,
                        trend=result.trend,
                        processLog=process_log if process_log is not None else []
                    )
                results.append(forecast_result)
                successful_combinations += 1

                if process_log is not None:
                    process_log.append(f"Combination {product} + {customer} + {location} processed successfully")

            except Exception as e:
                if process_log is not None:
                    process_log.append(f"Error processing combination {product} + {customer} + {location}: {str(e)}")

                failed_combinations.append({
                    'combination': f"{product} + {customer} + {location}",
                    'error': str(e)
                })

        if not results:
            raise ValueError("No valid forecasts could be generated for any combination")

        avg_accuracy = np.mean([r.accuracy for r in results])
        best_combination = max(results, key=lambda x: x.accuracy)
        worst_combination = min(results, key=lambda x: x.accuracy)

        summary = {
            'averageAccuracy': round(avg_accuracy, 2),
            'bestCombination': {
                'combination': best_combination.combination,
                'accuracy': best_combination.accuracy
            },
            'worstCombination': {
                'combination': worst_combination.combination,
                'accuracy': worst_combination.accuracy
            },
            'successfulCombinations': successful_combinations,
            'failedCombinations': len(failed_combinations),
            'failedDetails': failed_combinations
        }

        if process_log is not None:
            process_log.append(f"Three-dimension forecast completed. Successful: {successful_combinations}, Failed: {len(failed_combinations)}")

        return MultiForecastResult(
            results=results,
            totalCombinations=len(combinations),
            summary=summary,
            processLog=process_log if process_log is not None else []
        )


    @staticmethod
    def aggregate_two_dimensions(df: pd.DataFrame, interval: str, combination_type: str, dim1_value: str, dim2_value: str) -> pd.DataFrame:
        """
        Aggregate data specifically for two-dimensional combinations
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Filter data for the specific combination
        if combination_type == 'product_customer':
            # Filter for specific product and customer, aggregate across all locations
            filtered_df = df[(df['product'] == dim1_value) & (df['customer'] == dim2_value)]
            group_cols = ['product', 'customer']
        elif combination_type == 'product_location':
            # Filter for specific product and location, aggregate across all customers
            filtered_df = df[(df['product'] == dim1_value) & (df['location'] == dim2_value)]
            group_cols = ['product', 'location']
        elif combination_type == 'customer_location':
            # Filter for specific customer and location, aggregate across all products
            filtered_df = df[(df['customer'] == dim1_value) & (df['location'] == dim2_value)]
            group_cols = ['customer', 'location']
        else:
            raise ValueError(f"Unknown combination type: {combination_type}")

        if len(filtered_df) == 0:
            raise ValueError(f"No data found for combination: {dim1_value} + {dim2_value}")

        print(f"Filtered data shape: {filtered_df.shape}")
        print(f"Sample filtered data:\n{filtered_df.head()}")

        # Create period groupings
        if interval == 'week':
            filtered_df['period_group'] = filtered_df['date'].dt.to_period('W-MON')
        elif interval == 'month':
            filtered_df['period_group'] = filtered_df['date'].dt.to_period('M')
        elif interval == 'year':
            filtered_df['period_group'] = filtered_df['date'].dt.to_period('Y')
        else:
            filtered_df['period_group'] = filtered_df['date'].dt.to_period('M')

        # Group by the two dimensions and period, then sum quantities
        group_cols_with_period = group_cols + ['period_group']
        aggregated = filtered_df.groupby(group_cols_with_period)['quantity'].sum().reset_index()

        # Convert period back to timestamp
        aggregated['date'] = aggregated['period_group'].dt.start_time

        # Add period labels
        aggregated['period'] = aggregated['date'].apply(
            lambda x: ForecastingEngine.format_period(pd.Timestamp(x), interval)
        )

        # Keep essential columns plus external factors for forecasting
        external_factor_cols = [col for col in aggregated.columns if col not in ['product', 'customer', 'location', 'date', 'period', 'quantity', 'period_group']]
        result_cols = group_cols + ['date', 'period', 'quantity'] + external_factor_cols
        aggregated = aggregated[result_cols]

        print(f"Aggregated data shape: {aggregated.shape}")
        print(f"Sample aggregated data:\n{aggregated.head()}")

        return aggregated.reset_index(drop=True)

    @staticmethod
    def generate_forecast_two_dimensions(db: Session, config: ForecastConfig, products: list, customers: list, locations: list, process_log: List[str] = None) -> MultiForecastResult:
        """Generate forecasts for two dimension combinations with dedicated aggregation"""
        if process_log is not None:
            process_log.append("Generating two-dimension forecast...")

        combinations = []

        # Determine which two dimensions are selected and create combinations accordingly
        if products and customers:
            # Product + Customer selected, aggregate across all locations
            for p in products:
                for c in customers:
                    combinations.append(('product_customer', p, c))
        elif products and locations:
            # Product + Location selected, aggregate across all customers
            for p in products:
                for l in locations:
                    combinations.append(('product_location', p, l))
        elif customers and locations:
            # Customer + Location selected, aggregate across all products
            for c in customers:
                for l in locations:
                    combinations.append(('customer_location', c, l))

        if process_log is not None:
            process_log.append(f"Generated {len(combinations)} two-dimension combinations")

        # Load all data once (without filtering by specific combinations)
        base_config = ForecastConfig(
            forecastBy=config.forecastBy,
            selectedProducts=products,
            selectedCustomers=customers,
            selectedLocations=locations,
            algorithm=config.algorithm,
            interval=config.interval,
            historicPeriod=config.historicPeriod,
            forecastPeriod=config.forecastPeriod,
            multiSelect=False,  # Load all data
            externalFactors=config.externalFactors
        )

        # Load the full dataset once
        if process_log is not None:
            process_log.append("Loading full dataset...")

        full_df = ForecastingEngine.load_data_from_db(db, base_config)

        if process_log is not None:
            process_log.append(f"Full dataset loaded: {full_df.shape}")

        results = []
        successful_combinations = 0
        failed_combinations = []

        for combination_type, dim1_value, dim2_value in combinations:
            try:
                if process_log is not None:
                    process_log.append(f"\nProcessing combination: {combination_type} - {dim1_value} + {dim2_value}")

                # Use dedicated two-dimensional aggregation
                aggregated_df = ForecastingEngine.aggregate_two_dimensions(
                    full_df, config.interval, combination_type, dim1_value, dim2_value
                )

                if len(aggregated_df) < 2:
                    if process_log is not None:
                        process_log.append("Insufficient data after aggregation")

                    failed_combinations.append({
                        'combination': f"{combination_type}: {dim1_value} + {dim2_value}",
                        'error': 'Insufficient data after aggregation'
                    })
                    continue

                # Create combination dictionary for result
                if combination_type == 'product_customer':
                    combination_dict = {"product": dim1_value, "customer": dim2_value}
                elif combination_type == 'product_location':
                    combination_dict = {"product": dim1_value, "location": dim2_value}
                elif combination_type == 'customer_location':
                    combination_dict = {"customer": dim1_value, "location": dim2_value}

                # Create a simplified config for the algorithm (it doesn't need the multi-select complexity)
                algo_config = ForecastConfig(
                    forecastBy=config.forecastBy,
                    selectedProducts=None,
                    selectedCustomers=None,
                    selectedLocations=None,
                    algorithm=config.algorithm,
                    interval=config.interval,
                    historicPeriod=config.historicPeriod,
                    forecastPeriod=config.forecastPeriod,
                    multiSelect=False,
                    externalFactors=config.externalFactors
                )

                # Run forecasting algorithm
                if config.algorithm in ["best_fit", "best_statistical", "best_ml", "best_specialized"]:
                    if process_log is not None:
                        if config.algorithm == "best_fit":
                            process_log.append("Running best fit algorithm selection...")
                        elif config.algorithm == "best_statistical":
                            process_log.append("Running best statistical method selection...")
                        elif config.algorithm == "best_ml":
                            process_log.append("Running best machine learning method selection...")
                        elif config.algorithm == "best_specialized":
                            process_log.append("Running best specialized method selection...")

                    # Define algorithm categories
                    statistical_algorithms = [
                        "linear_regression", "polynomial_regression", "exponential_smoothing", 
                        "holt_winters", "arima", "sarima", "ses", "damped_trend", 
                        "theta_method", "drift_method", "naive_seasonal", "prophet_like"
                    ]

                    ml_algorithms = [
                        "random_forest", "xgboost", "svr", "knn", "gaussian_process", 
                        "neural_network", "lstm_like"
                    ]

                    specialized_algorithms = [
                        "seasonal_decomposition", "moving_average", "croston"
                    ]

                    # Select algorithms based on category
                    if config.algorithm == "best_statistical":
                        algorithms = statistical_algorithms
                    elif config.algorithm == "best_ml":
                        algorithms = ml_algorithms
                    elif config.algorithm == "best_specialized":
                        algorithms = specialized_algorithms
                    else:  # best_fit
                        algorithms = [alg for alg in ForecastingEngine.ALGORITHMS.keys() 
                                    if alg not in ["best_fit", "best_statistical", "best_ml", "best_specialized"]]

                    algorithm_results = []
                    # Use parallel execution for two-dimension best fit
                    max_workers = min(len(algorithms), os.cpu_count() or 4)
                    if process_log is not None:
                        process_log.append(f"Running parallel best fit with {max_workers} workers...")

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_algorithm = {
                            executor.submit(ForecastingEngine.run_algorithm, algorithm, aggregated_df, algo_config, save_model=False): algorithm
                            for algorithm in algorithms
                        }

                        for future in as_completed(future_to_algorithm):
                            algorithm_name = future_to_algorithm[future]
                            try:
                                result = future.result()
                                algorithm_results.append(result)
                                if process_log is not None:
                                    process_log.append(f"✅ Algorithm {algorithm_name} completed for {combination_type}: {dim1_value}+{dim2_value}")
                            except Exception as exc:
                                if process_log is not None:
                                    process_log.append(f"❌ Algorithm {algorithm_name} failed for {combination_type}: {dim1_value}+{dim2_value}: {str(exc)}")
                                continue

                    if not algorithm_results:
                        if process_log is not None:
                            process_log.append("No algorithms produced valid results")

                        failed_combinations.append({
                            'combination': f"{combination_type}: {dim1_value} + {dim2_value}",
                            'error': 'No algorithms produced valid results'
                        })
                        continue
                    best_result = max(algorithm_results, key=lambda x: x.accuracy)
                    forecast_result = ForecastResult(
                        combination=combination_dict,
                        selectedAlgorithm=f"{best_result.algorithm} (Best Fit)",
                        accuracy=best_result.accuracy,
                        mae=best_result.mae,
                        rmse=best_result.rmse,
                        historicData=best_result.historicData,
                        forecastData=best_result.forecastData,
                        trend=best_result.trend,
                        allAlgorithms=algorithm_results,
                        processLog=process_log if process_log is not None else []
                    )
                else:
                    if process_log is not None:
                        process_log.append(f"Running algorithm: {config.algorithm}")

                    result = ForecastingEngine.run_algorithm(config.algorithm, aggregated_df, algo_config)
                    forecast_result = ForecastResult(
                        combination=combination_dict,
                        selectedAlgorithm=result.algorithm,
                        accuracy=result.accuracy,
                        mae=result.mae,
                        rmse=result.rmse,
                        historicData=result.historicData,
                        forecastData=result.forecastData,
                        trend=result.trend,
                        processLog=process_log if process_log is not None else []
                    )

                results.append(forecast_result)
                successful_combinations += 1

                if process_log is not None:
                    process_log.append(f"Successfully processed: {combination_type} - {dim1_value} + {dim2_value}")

            except Exception as e:
                if process_log is not None:
                    process_log.append(f"Error processing combination {dim1_value} + {dim2_value}: {e}")

                failed_combinations.append({
                    'combination': f"{combination_type}: {dim1_value} + {dim2_value}",
                    'error': str(e)
                })

        if not results:
            raise ValueError("No valid forecasts could be generated for any combination")

        avg_accuracy = np.mean([r.accuracy for r in results])
        best_combination = max(results, key=lambda x: x.accuracy)
        worst_combination = min(results, key=lambda x: x.accuracy)

        summary = {
            'averageAccuracy': round(avg_accuracy, 2),
            'bestCombination': {
                'combination': best_combination.combination,
                'accuracy': best_combination.accuracy
            },
            'worstCombination': {
                'combination': worst_combination.combination,
                'accuracy': worst_combination.accuracy
            },
            'successfulCombinations': successful_combinations,
            'failedCombinations': len(failed_combinations),
            'failedDetails': failed_combinations
        }

        if process_log is not None:
            process_log.append(f"Two-dimension forecast completed. Successful: {successful_combinations}, Failed: {len(failed_combinations)}")

        return MultiForecastResult(
            results=results,
            totalCombinations=len(combinations),
            summary=summary,
            processLog=process_log if process_log is not None else []
        )

    @staticmethod
    def generate_multi_forecast(db: Session, config: ForecastConfig, process_log: List[str] = None) -> MultiForecastResult:
        """Generate forecasts for multiple combinations"""
        if process_log is not None:
            process_log.append("Generating multi-forecast...")

        products = config.selectedProducts or []
        customers = config.selectedCustomers or []
        locations = config.selectedLocations or []

        if process_log is not None:
            process_log.append(f"Multi-forecast: products={len(products)}, customers={len(customers)}, locations={len(locations)}")

        selected_dimensions = 0
        if products: selected_dimensions += 1
        if customers: selected_dimensions += 1
        if locations: selected_dimensions += 1

        if selected_dimensions < 2:
            raise ValueError("Please select at least two dimensions (Product, Customer, or Location) for multi-selection forecasting")

        if selected_dimensions == 3:
            if process_log is not None:
                process_log.append("Running three-dimension forecast")
                # Check if all three dimensions are selected for advanced mode
                if not (config.selectedProducts and config.selectedCustomers and config.selectedLocations):
                    process_log.append("ERROR: Advanced mode requires all three dimensions to be selected")
                    raise ValueError("Advanced mode requires selection of Products, Customers, and Locations")
                
            return ForecastingEngine.generate_forecast_three_dimensions(db, config, 
                config.selectedProducts, config.selectedCustomers, config.selectedLocations, process_log)
        else:
            if process_log is not None:
                process_log.append("Running two-dimension forecast")
            return ForecastingEngine.generate_forecast_two_dimensions(db, config, products, customers, locations, process_log)

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"message": "Advanced Multi-variant Forecasting API with MySQL is running", "algorithms": list(ForecastingEngine.ALGORITHMS.values())}

# Authentication endpoints
# @app.post("/auth/register", response_model=UserResponse)
# async def register(user_data: UserCreate, db: Session = Depends(get_db)):
#     """Register a new user"""
#     # Check if username already exists
#     existing_user = db.query(User).filter(
#         or_(User.username == user_data.username, User.email == user_data.email)
#     ).first()

#     if existing_user:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Username or email already registered"
#         )

#     # Create new user
#     hashed_password = User.hash_password(user_data.password)
#     new_user = User(
#         username=user_data.username,
#         email=user_data.email,
#         hashed_password=hashed_password,
#         full_name=user_data.full_name,
#         is_active=1
#     )

#     db.add(new_user)
#     db.commit()
#     db.refresh(new_user)

#     return UserResponse(
#         id=new_user.id,
#         username=new_user.username,
#         email=new_user.email,
#         full_name=new_user.full_name,
#         is_active=bool(new_user.is_active),
#         created_at=new_user.created_at.isoformat()
#     )

# @app.post("/auth/login", response_model=Token)
# async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
#     """Login user and return access token"""
#     user = db.query(User).filter(User.username == user_credentials.username).first()

#     if not user or not user.verify_password(user_credentials.password):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

#     if not user.is_active:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Inactive user"
#         )

#     access_token_expires = timedelta(minutes=30)
#     access_token = create_access_token(
#         data={"sub": user.username}, expires_delta=access_token_expires
#     )

#     return Token(
#         access_token=access_token,
#         token_type="bearer",
#         user=UserResponse(
#             id=user.id,
#             username=user.username,
#             email=user.email,
#             full_name=user.full_name,
#             is_active=bool(user.is_active),
#             created_at=user.created_at.isoformat()
#         )
#     )

# @app.get("/auth/me", response_model=UserResponse)
# async def get_current_user_info(current_user: User = Depends(get_current_user)):
#     """Get current user information"""
#     return UserResponse(
#         id=current_user.id,
#         username=current_user.username,
#         email=current_user.email,
#         full_name=current_user.full_name,
#         is_active=bool(current_user.is_active),
#         created_at=current_user.created_at.isoformat()
#     )

@app.get("/admin/users", response_model=List[UserResponse])
async def list_users(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all users (admin only)"""
    require_admin(current_user)
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            email=u.email,
            full_name=u.full_name,
            is_active=bool(u.is_active),
            is_approved=bool(u.is_approved),
            created_at=u.created_at.isoformat()
        ) for u in users
    ]

@app.post("/admin/users/{user_id}/approve", response_model=UserResponse)
async def approve_user(user_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Approve a pending user (admin only)"""
    require_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    user.is_approved = True
    db.commit()
    db.refresh(user)
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=bool(user.is_active),
        is_approved=bool(user.is_approved),
        created_at=user.created_at.isoformat()
    )


@app.post("/admin/users/{user_id}/active", response_model=UserResponse)
async def set_user_active(user_id: int, payload: AdminSetActiveRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Activate/Deactivate a user (admin only)"""
    require_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.is_active = bool(payload.is_active)
    db.commit()
    db.refresh(user)

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=bool(user.is_active),
        is_approved=bool(user.is_approved),
        is_admin=bool(user.is_admin),
        created_at=user.created_at.isoformat()
    )
@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if username already exists
    existing_user = db.query(User).filter(
        or_(User.username == user_data.username, User.email == user_data.email)
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )

    # Create new user
    hashed_password = User.hash_password(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=True,
        is_approved=False  # Set is_approved to False
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return UserResponse(
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        full_name=new_user.full_name,
        is_active=bool(new_user.is_active),
        created_at=new_user.created_at.isoformat()
    )


@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Login user and return access token"""
    user = db.query(User).filter(User.username == user_credentials.username).first()

    if not user or not user.verify_password(user_credentials.password):
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

    if not user.is_approved:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not approved by admin"
        )

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=bool(user.is_active),
            is_approved=bool(user.is_approved),
            created_at=user.created_at.isoformat()
        )
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=bool(current_user.is_active),
        is_approved=bool(current_user.is_approved),
        is_admin=bool(current_user.is_admin),
        created_at=current_user.created_at.isoformat()
    )

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
    """Upload and store external factor data file in MySQL database"""
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

@app.get("/accuracy_history/{config_hash}")
async def get_accuracy_history(
    config_hash: str,
    days_back: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get accuracy history for a configuration"""
    try:
        history = ModelPersistenceManager.get_accuracy_history(db, config_hash, days_back)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting accuracy history: {str(e)}")

@app.get("/model_cache_info")
async def get_model_cache_info(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get information about cached models"""
    try:
        models = db.query(SavedModel).order_by(SavedModel.last_used.desc()).limit(50).all()

        result = []
        for model in models:
            result.append({
                'model_hash': model.model_hash,
                'algorithm': model.algorithm,
                'accuracy': model.accuracy,
                'created_at': model.created_at.isoformat(),
                'last_used': model.last_used.isoformat(),
                'use_count': model.use_count
            })

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model cache info: {str(e)}")

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

@app.post("/clear_all_model_cache")
async def clear_all_model_cache(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Clear ALL cached models immediately"""
    try:
        # Count models before cleanup
        before_count = db.query(SavedModel).count()

        # Delete all models
        db.query(SavedModel).delete()
        db.query(ModelAccuracyHistory).delete()
        db.commit()

        # Count models after cleanup
        after_count = db.query(SavedModel).count()
        cleared_count = before_count - after_count

        return {
            "message": f"All model cache cleared successfully",
            "cleared_count": cleared_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing all model cache: {str(e)}")

@app.post("/clear_model_cache")
async def clear_model_cache(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Clear old cached models"""
    try:
        # Count models before cleanup
        before_count = db.query(SavedModel).count()

        # Cleanup old models
        ModelPersistenceManager.cleanup_old_models(db, days_old=7, max_models=50)

        # Count models after cleanup
        after_count = db.query(SavedModel).count()
        cleared_count = before_count - after_count

        return {
            "message": f"Model cache cleanup completed",
            "cleared_count": cleared_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing model cache: {str(e)}")


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

        if config.multiSelect:
            if config.advancedMode:
                # Advanced mode: precise Product-Customer-Location combinations
                process_log.append("Running advanced mode (precise combinations)")
                # Check if all three dimensions are selected for advanced mode
                if not (config.selectedProducts and config.selectedCustomers and config.selectedLocations):
                    process_log.append("ERROR: Advanced mode requires all three dimensions")
                    raise ValueError("Advanced mode requires selection of Products, Customers, and Locations")
                result = ForecastingEngine.generate_forecast_three_dimensions(db, config, 
                    config.selectedProducts, config.selectedCustomers, config.selectedLocations, process_log)

                # Auto-save the forecast result
                try:
                    auto_save_name = f"Auto-saved Multi-Forecast {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    auto_save_description = "Automatically saved multi-forecast result"

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
                        process_log.append(f"Multi-forecast automatically saved as '{auto_save_name}'")
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
                        process_log.append(f"Multi-forecast automatically saved as '{new_name}'")
                except Exception as save_error:
                    process_log.append(f"Warning: Could not auto-save multi-forecast: {str(save_error)}")

                return result
            else:
                # Multi-selection mode - check if it's flexible multi-select or simple multi-select
                selected_dimensions = 0
                if config.selectedProducts: selected_dimensions += 1
                if config.selectedCustomers: selected_dimensions += 1
                if config.selectedLocations: selected_dimensions += 1

                process_log.append(f"Multi-select mode: {selected_dimensions} dimensions selected")

                if selected_dimensions >= 2:
                    # Flexible multi-select mode (2 or 3 dimensions with aggregation)
                    result = ForecastingEngine.generate_multi_forecast(db, config, process_log)

                    # Auto-save the forecast result
                    try:
                        auto_save_name = f"Auto-saved Multi-Forecast {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        auto_save_description = "Automatically saved multi-forecast result"

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
                            process_log.append(f"Multi-forecast automatically saved as '{auto_save_name}'")
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
                            process_log.append(f"Multi-forecast automatically saved as '{new_name}'")
                    except Exception as save_error:
                        process_log.append(f"Warning: Could not auto-save multi-forecast: {str(save_error)}")

                    return result
                else:
                    # This shouldn't happen with proper frontend validation
                    process_log.append("ERROR: Multi-select mode requires at least 2 dimensions")
                    raise ValueError("Multi-select mode requires at least 2 dimensions")

        elif config.selectedItems and len(config.selectedItems) > 1:
            # Simple mode multi-select
            process_log.append("Running simple multi-select mode")
            result = ForecastingEngine.generate_simple_multi_forecast(db, config, process_log)

            # Auto-save the forecast result
            try:
                auto_save_name = f"Auto-saved Multi-Forecast {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                auto_save_description = "Automatically saved multi-forecast result"

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
                    process_log.append(f"Multi-forecast automatically saved as '{auto_save_name}'")
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
                    process_log.append(f"Multi-forecast automatically saved as '{new_name}'")
            except Exception as save_error:
                process_log.append(f"Warning: Could not auto-save multi-forecast: {str(save_error)}")

            return result
        else:
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

from fastapi.responses import StreamingResponse # type: ignore

from fastapi.responses import StreamingResponse # type: ignore
 
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
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Excel: {str(e)}")
 
@app.post("/download_multi_forecast_excel")
async def download_multi_forecast_excel(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Download multi-forecast data as Excel with all combinations"""
    try:
        multi_result = MultiForecastResult(**request['multiForecastResult'])
 
        # Prepare comprehensive data for all combinations
        all_data = []
        summary_data = []
 
        for result in multi_result.results:
            combination = result.combination or {}
            product = combination.get('product', '')
            customer = combination.get('customer', '')
            location = combination.get('location', '')
 
            # Fix: For simple multi-select mode, combination may be a single dimension with multiple items
            # If combination is empty or missing keys, try to get from selectedItems in summary or elsewhere
            if not product and not customer and not location:
                # Try to get dimension name and items from result.combination keys and values
                if result.combination and len(result.combination) == 1:
                    dim_name = list(result.combination.keys())[0]
                    dim_value = result.combination[dim_name]
                    if isinstance(dim_value, list):
                        # Multiple items selected, join them as string
                        dim_value_str = ", ".join(dim_value)
                    else:
                        dim_value_str = str(dim_value)
                    if dim_name == 'product':
                        product = dim_value_str
                    elif dim_name == 'customer':
                        customer = dim_value_str
                    elif dim_name == 'location':
                        location = dim_value_str
 
            # Add summary row for this combination
            summary_data.append({
                "Product": product,
                "Customer": customer,
                "Location": location,
                "Algorithm": result.selectedAlgorithm,
                "Accuracy": result.accuracy,
                "MAE": result.mae,
                "RMSE": result.rmse,
                "Trend": result.trend,
                "Historic_Periods": len(result.historicData),
                "Forecast_Periods": len(result.forecastData)
            })
 
            # Add historical data
            for d in result.historicData:
                all_data.append({
                    "Product": product,
                    "Customer": customer,
                    "Location": location,
                    "Date": d.date,
                    "Period": d.period,
                    "Quantity": d.quantity,
                    "Type": "Historical",
                    "Algorithm": result.selectedAlgorithm,
                    "Accuracy": result.accuracy
                })
 
            # Add forecast data
            for d in result.forecastData:
                all_data.append({
                    "Product": product,
                    "Customer": customer,
                    "Location": location,
                    "Date": d.date,
                    "Period": d.period,
                    "Quantity": d.quantity,
                    "Type": "Forecast",
                    "Algorithm": result.selectedAlgorithm,
                    "Accuracy": result.accuracy
                })
 
        # Create DataFrames
        all_df = pd.DataFrame(all_data)
        summary_df = pd.DataFrame(summary_data)
 
        # Create overall summary
        overall_summary = pd.DataFrame([{
            "Total_Combinations": multi_result.totalCombinations,
            "Successful_Combinations": multi_result.summary['successfulCombinations'],
            "Failed_Combinations": multi_result.summary['failedCombinations'],
            "Average_Accuracy": multi_result.summary['averageAccuracy'],
            "Best_Combination": f"{multi_result.summary['bestCombination']['combination'].get('product', '')} → {multi_result.summary['bestCombination']['combination'].get('customer', '')} → {multi_result.summary['bestCombination']['combination'].get('location', '')}",
            "Best_Accuracy": multi_result.summary['bestCombination']['accuracy'],
            "Worst_Combination": f"{multi_result.summary['worstCombination']['combination'].get('product', '')} → {multi_result.summary['worstCombination']['combination'].get('customer', '')} → {multi_result.summary['worstCombination']['combination'].get('location', '')}",
            "Worst_Accuracy": multi_result.summary['worstCombination']['accuracy']
        }])
 
        # Write to Excel in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # All forecast data
            all_df.to_excel(writer, index=False, sheet_name="All Forecast Data")
 
            # Summary by combination
            summary_df.to_excel(writer, index=False, sheet_name="Combination Summary")
 
            # Overall summary
            overall_summary.to_excel(writer, index=False, sheet_name="Overall Summary")
 
            # Failed combinations if any
            if multi_result.summary['failedCombinations'] > 0:
                failed_df = pd.DataFrame(multi_result.summary['failedDetails'])
                failed_df.to_excel(writer, index=False, sheet_name="Failed Combinations")
 
        output.seek(0)
 
        # Create filename
        filename = f"multi_forecast_{multi_result.totalCombinations}combinations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
 
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating multi-forecast Excel: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn # type: ignore
    print("🚀 Advanced Multi-variant Forecasting API with MySQL")
    print("📊 23 Algorithms + Best Fit Available")
    print("🗄️  PostgreSQL Database Integration")
    print("🌐 Server starting on http://localhost:8000")
    print("📈 Frontend should be available on http://localhost:5173")
    print("⏹️  Press Ctrl+C to stop the server\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)