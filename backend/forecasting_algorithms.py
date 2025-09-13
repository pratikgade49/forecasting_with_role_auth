#!/usr/bin/env python3
"""
Advanced Forecasting Algorithms Engine

This module provides comprehensive forecasting capabilities with multiple algorithms,
model persistence, and advanced features.
"""

import logging
import traceback
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

# Import database models
from database import ForecastData, ProductCustomerLocationCombination, ExternalFactorData
from database_utils import get_aggregated_data_optimized
from model_persistence import ModelPersistenceManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForecastPoint:
    """Data class for forecast points"""
    period: str
    quantity: float
    date: Optional[str] = None

@dataclass
class AlgorithmResult:
    """Data class for algorithm results"""
    algorithm: str
    accuracy: float
    mae: float
    rmse: float
    historic_data: List[ForecastPoint]
    forecast_data: List[ForecastPoint]
    trend: str
    model: Any = None

class ForecastingEngine:
    """Main forecasting engine with multiple algorithms"""
    
    def __init__(self, db: Session):
        self.db = db
        self.process_log = []
        
        # Algorithm registry
        self.algorithms = {
            'linear_regression': self._linear_regression,
            'polynomial_regression': self._polynomial_regression,
            'exponential_smoothing': self._exponential_smoothing,
            'holt_winters': self._holt_winters,
            'arima': self._arima,
            'random_forest': self._random_forest,
            'seasonal_decomposition': self._seasonal_decomposition,
            'moving_average': self._moving_average,
        }
    
    def log(self, message: str):
        """Add message to process log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.process_log.append(log_message)
        logger.info(log_message)
    
    def generate_single_forecast(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forecast for a single item/combination"""
        try:
            self.log("Starting single forecast generation")
            self.log(f"Configuration: {config}")
            
            # Get data based on configuration
            data = self._get_forecast_data(config)
            
            if not data:
                raise ValueError("No data found for the selected criteria")
            
            self.log(f"Retrieved {len(data)} data points")
            
            # Prepare time series data
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Aggregate by interval
            df_agg = self._aggregate_by_interval(df, config.get('interval', 'month'))
            
            if len(df_agg) < config.get('historicPeriod', 12):
                raise ValueError(f"Insufficient data. Need at least {config.get('historicPeriod', 12)} periods, got {len(df_agg)}")
            
            # Get external factors if specified
            external_factors = self._get_external_factors(config, df_agg)
            
            # Generate forecast
            algorithm = config.get('algorithm', 'best_fit')
            
            if algorithm == 'best_fit':
                result = self._best_fit_forecast(df_agg, config, external_factors)
            else:
                result = self._single_algorithm_forecast(df_agg, config, algorithm, external_factors)
            
            self.log("Forecast generation completed successfully")
            
            # Add process log to result
            result['processLog'] = self.process_log.copy()
            
            return result
            
        except Exception as e:
            self.log(f"Error in forecast generation: {str(e)}")
            logger.error(f"Single forecast error: {e}")
            logger.error(traceback.format_exc())
            raise e
    
    def generate_multi_forecast(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forecasts for multiple combinations"""
        try:
            self.log("Starting multi-forecast generation")
            
            # Get all combinations to forecast
            combinations = self._get_forecast_combinations(config)
            
            if not combinations:
                raise ValueError("No valid combinations found for the selected criteria")
            
            self.log(f"Found {len(combinations)} combinations to forecast")
            
            results = []
            successful_count = 0
            failed_count = 0
            failed_details = []
            total_accuracy = 0
            
            for i, combination in enumerate(combinations):
                try:
                    self.log(f"Processing combination {i+1}/{len(combinations)}: {combination}")
                    
                    # Create single forecast config for this combination
                    single_config = config.copy()
                    single_config.update({
                        'selectedProduct': combination.get('product'),
                        'selectedCustomer': combination.get('customer'),
                        'selectedLocation': combination.get('location'),
                        'multiSelect': False
                    })
                    
                    # Generate forecast for this combination
                    result = self.generate_single_forecast(single_config)
                    
                    # Add combination info to result
                    result['combination'] = combination
                    results.append(result)
                    
                    successful_count += 1
                    total_accuracy += result.get('accuracy', 0)
                    
                    self.log(f"Successfully processed combination {i+1}")
                    
                except Exception as e:
                    failed_count += 1
                    error_msg = str(e)
                    combination_str = f"{combination.get('product', 'N/A')} - {combination.get('customer', 'N/A')} - {combination.get('location', 'N/A')}"
                    
                    failed_details.append({
                        'combination': combination_str,
                        'error': error_msg
                    })
                    
                    self.log(f"Failed to process combination {i+1}: {error_msg}")
                    continue
            
            if successful_count == 0:
                raise ValueError("No forecasts could be generated successfully")
            
            # Calculate summary statistics
            average_accuracy = total_accuracy / successful_count if successful_count > 0 else 0
            
            # Find best and worst combinations
            best_combination = max(results, key=lambda x: x.get('accuracy', 0)) if results else None
            worst_combination = min(results, key=lambda x: x.get('accuracy', 0)) if results else None
            
            summary = {
                'averageAccuracy': round(average_accuracy, 2),
                'bestCombination': {
                    'combination': best_combination.get('combination', {}),
                    'accuracy': best_combination.get('accuracy', 0)
                } if best_combination else None,
                'worstCombination': {
                    'combination': worst_combination.get('combination', {}),
                    'accuracy': worst_combination.get('accuracy', 0)
                } if worst_combination else None,
                'successfulCombinations': successful_count,
                'failedCombinations': failed_count,
                'failedDetails': failed_details
            }
            
            self.log(f"Multi-forecast completed: {successful_count} successful, {failed_count} failed")
            
            return {
                'results': results,
                'totalCombinations': len(combinations),
                'summary': summary,
                'processLog': self.process_log.copy()
            }
            
        except Exception as e:
            self.log(f"Error in multi-forecast generation: {str(e)}")
            logger.error(f"Multi-forecast error: {e}")
            logger.error(traceback.format_exc())
            raise e
    
    def _get_forecast_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get forecast data based on configuration"""
        try:
            # Determine the selection mode
            if config.get('selectedProduct') and config.get('selectedCustomer') and config.get('selectedLocation'):
                # Precise combination mode
                return self._get_precise_combination_data(config)
            elif config.get('selectedItems') and len(config.get('selectedItems', [])) > 1:
                # Multiple items of same type
                return self._get_multiple_items_data(config)
            elif config.get('selectedItem'):
                # Single item aggregated
                return self._get_single_item_data(config)
            else:
                raise ValueError("Invalid configuration: no valid selection criteria found")
                
        except Exception as e:
            logger.error(f"Error getting forecast data: {e}")
            raise e
    
    def _get_precise_combination_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get data for a precise product-customer-location combination"""
        try:
            # Find the combination
            combination = self.db.query(ProductCustomerLocationCombination).filter(
                and_(
                    ProductCustomerLocationCombination.product == config['selectedProduct'],
                    ProductCustomerLocationCombination.customer == config['selectedCustomer'],
                    ProductCustomerLocationCombination.location == config['selectedLocation']
                )
            ).first()
            
            if not combination:
                raise ValueError(f"No data found for combination: {config['selectedProduct']} - {config['selectedCustomer']} - {config['selectedLocation']}")
            
            # Get forecast data for this combination
            forecast_data = self.db.query(ForecastData).filter(
                ForecastData.combination_id == combination.id
            ).order_by(ForecastData.date).all()
            
            return [
                {
                    'date': fd.date.isoformat(),
                    'quantity': float(fd.quantity),
                    'product': combination.product,
                    'customer': combination.customer,
                    'location': combination.location
                }
                for fd in forecast_data
            ]
            
        except Exception as e:
            logger.error(f"Error getting precise combination data: {e}")
            raise e
    
    def _get_single_item_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get aggregated data for a single item"""
        try:
            forecast_by = config.get('forecastBy', 'product')
            selected_item = config.get('selectedItem')
            
            if not selected_item:
                raise ValueError("No item selected")
            
            # Build query based on forecast dimension
            query = self.db.query(
                ForecastData.date,
                func.sum(ForecastData.quantity).label('total_quantity')
            ).join(
                ProductCustomerLocationCombination,
                ForecastData.combination_id == ProductCustomerLocationCombination.id
            )
            
            # Apply filter based on forecast dimension
            if forecast_by == 'product':
                query = query.filter(ProductCustomerLocationCombination.product == selected_item)
            elif forecast_by == 'customer':
                query = query.filter(ProductCustomerLocationCombination.customer == selected_item)
            elif forecast_by == 'location':
                query = query.filter(ProductCustomerLocationCombination.location == selected_item)
            else:
                raise ValueError(f"Invalid forecast dimension: {forecast_by}")
            
            # Group by date and order
            results = query.group_by(ForecastData.date).order_by(ForecastData.date).all()
            
            return [
                {
                    'date': result.date.isoformat(),
                    'quantity': float(result.total_quantity)
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting single item data: {e}")
            raise e
    
    def _get_multiple_items_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get aggregated data for multiple items of the same type"""
        try:
            forecast_by = config.get('forecastBy', 'product')
            selected_items = config.get('selectedItems', [])
            
            if not selected_items:
                raise ValueError("No items selected")
            
            # Build query based on forecast dimension
            query = self.db.query(
                ForecastData.date,
                func.sum(ForecastData.quantity).label('total_quantity')
            ).join(
                ProductCustomerLocationCombination,
                ForecastData.combination_id == ProductCustomerLocationCombination.id
            )
            
            # Apply filter based on forecast dimension
            if forecast_by == 'product':
                query = query.filter(ProductCustomerLocationCombination.product.in_(selected_items))
            elif forecast_by == 'customer':
                query = query.filter(ProductCustomerLocationCombination.customer.in_(selected_items))
            elif forecast_by == 'location':
                query = query.filter(ProductCustomerLocationCombination.location.in_(selected_items))
            else:
                raise ValueError(f"Invalid forecast dimension: {forecast_by}")
            
            # Group by date and order
            results = query.group_by(ForecastData.date).order_by(ForecastData.date).all()
            
            return [
                {
                    'date': result.date.isoformat(),
                    'quantity': float(result.total_quantity)
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting multiple items data: {e}")
            raise e
    
    def _get_forecast_combinations(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all combinations for multi-forecast"""
        try:
            combinations = []
            
            if config.get('multiSelect'):
                # Multi-selection mode
                selected_products = config.get('selectedProducts', [])
                selected_customers = config.get('selectedCustomers', [])
                selected_locations = config.get('selectedLocations', [])
                
                # Build query for combinations
                query = self.db.query(ProductCustomerLocationCombination)
                
                # Apply filters
                filters = []
                if selected_products:
                    filters.append(ProductCustomerLocationCombination.product.in_(selected_products))
                if selected_customers:
                    filters.append(ProductCustomerLocationCombination.customer.in_(selected_customers))
                if selected_locations:
                    filters.append(ProductCustomerLocationCombination.location.in_(selected_locations))
                
                if filters:
                    query = query.filter(and_(*filters))
                
                # Get combinations that have data
                query = query.join(ForecastData, ProductCustomerLocationCombination.id == ForecastData.combination_id)
                query = query.distinct()
                
                results = query.all()
                
                for combination in results:
                    combinations.append({
                        'product': combination.product,
                        'customer': combination.customer,
                        'location': combination.location
                    })
            
            return combinations
            
        except Exception as e:
            logger.error(f"Error getting forecast combinations: {e}")
            raise e
    
    def _get_external_factors(self, config: Dict[str, Any], df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get external factors data if specified"""
        try:
            external_factors = config.get('externalFactors', [])
            if not external_factors:
                return None
            
            self.log(f"Loading external factors: {external_factors}")
            
            # Get date range from main data
            start_date = df['date'].min()
            end_date = df['date'].max()
            
            # Query external factors
            factors_data = self.db.query(ExternalFactorData).filter(
                and_(
                    ExternalFactorData.factor_name.in_(external_factors),
                    ExternalFactorData.date >= start_date,
                    ExternalFactorData.date <= end_date
                )
            ).all()
            
            if not factors_data:
                self.log("No external factors data found in date range")
                return None
            
            # Convert to DataFrame
            factors_df = pd.DataFrame([
                {
                    'date': fd.date,
                    'factor_name': fd.factor_name,
                    'factor_value': float(fd.factor_value)
                }
                for fd in factors_data
            ])
            
            # Pivot to have factors as columns
            factors_pivot = factors_df.pivot(index='date', columns='factor_name', values='factor_value')
            factors_pivot = factors_pivot.reset_index()
            
            self.log(f"Loaded external factors data: {factors_pivot.shape}")
            return factors_pivot
            
        except Exception as e:
            self.log(f"Error loading external factors: {str(e)}")
            return None
    
    def _aggregate_by_interval(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Aggregate data by specified interval"""
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            if interval == 'week':
                df['period'] = df['date'].dt.to_period('W')
            elif interval == 'month':
                df['period'] = df['date'].dt.to_period('M')
            elif interval == 'year':
                df['period'] = df['date'].dt.to_period('Y')
            else:
                raise ValueError(f"Unsupported interval: {interval}")
            
            # Aggregate by period
            agg_df = df.groupby('period').agg({
                'quantity': 'sum',
                'date': 'first'
            }).reset_index()
            
            # Convert period to string for JSON serialization
            agg_df['period_str'] = agg_df['period'].astype(str)
            
            return agg_df.sort_values('date')
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            raise e
    
    def _best_fit_forecast(self, df: pd.DataFrame, config: Dict[str, Any], external_factors: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run all algorithms and select the best one"""
        try:
            self.log("Running Best Fit analysis - testing all algorithms")
            
            results = []
            training_data = df['quantity'].values
            
            # Test all algorithms
            for algorithm_name, algorithm_func in self.algorithms.items():
                try:
                    self.log(f"Testing {algorithm_name}")
                    
                    # Check for cached model
                    cached_model_hash = ModelPersistenceManager.find_cached_model(
                        self.db, algorithm_name, config, training_data
                    )
                    
                    if cached_model_hash:
                        self.log(f"Found cached model for {algorithm_name}")
                        cached_model = ModelPersistenceManager.load_model(self.db, cached_model_hash)
                        if cached_model:
                            # Use cached model for prediction
                            result = self._use_cached_model(cached_model, df, config, algorithm_name)
                            if result:
                                results.append(result)
                                continue
                    
                    # Run algorithm
                    result = algorithm_func(df, config, external_factors)
                    if result:
                        results.append(result)
                        
                        # Save model to cache
                        if result.model:
                            metrics = {
                                'accuracy': result.accuracy,
                                'mae': result.mae,
                                'rmse': result.rmse
                            }
                            ModelPersistenceManager.save_model(
                                self.db, result.model, algorithm_name, config, training_data, metrics
                            )
                    
                except Exception as e:
                    self.log(f"Algorithm {algorithm_name} failed: {str(e)}")
                    continue
            
            if not results:
                raise ValueError("All algorithms failed to generate forecasts")
            
            # Select best algorithm based on accuracy
            best_result = max(results, key=lambda x: x.accuracy)
            
            self.log(f"Best algorithm: {best_result.algorithm} with {best_result.accuracy:.2f}% accuracy")
            
            # Prepare response
            response = {
                'selectedAlgorithm': f"{best_result.algorithm} (Best Fit)",
                'accuracy': best_result.accuracy,
                'mae': best_result.mae,
                'rmse': best_result.rmse,
                'trend': best_result.trend,
                'historicData': [
                    {'period': point.period, 'quantity': point.quantity}
                    for point in best_result.historic_data
                ],
                'forecastData': [
                    {'period': point.period, 'quantity': point.quantity}
                    for point in best_result.forecast_data
                ],
                'allAlgorithms': [
                    {
                        'algorithm': result.algorithm,
                        'accuracy': result.accuracy,
                        'mae': result.mae,
                        'rmse': result.rmse,
                        'trend': result.trend
                    }
                    for result in sorted(results, key=lambda x: x.accuracy, reverse=True)
                ]
            }
            
            # Generate config hash for tracking
            config_str = json.dumps(config, sort_keys=True)
            response['configHash'] = hashlib.sha256(config_str.encode()).hexdigest()
            
            return response
            
        except Exception as e:
            logger.error(f"Best fit forecast error: {e}")
            raise e
    
    def _single_algorithm_forecast(self, df: pd.DataFrame, config: Dict[str, Any], algorithm: str, external_factors: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run a single algorithm"""
        try:
            self.log(f"Running {algorithm} algorithm")
            
            if algorithm not in self.algorithms:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            training_data = df['quantity'].values
            
            # Check for cached model
            cached_model_hash = ModelPersistenceManager.find_cached_model(
                self.db, algorithm, config, training_data
            )
            
            if cached_model_hash:
                self.log(f"Found cached model for {algorithm}")
                cached_model = ModelPersistenceManager.load_model(self.db, cached_model_hash)
                if cached_model:
                    result = self._use_cached_model(cached_model, df, config, algorithm)
                    if result:
                        response = {
                            'selectedAlgorithm': result.algorithm,
                            'accuracy': result.accuracy,
                            'mae': result.mae,
                            'rmse': result.rmse,
                            'trend': result.trend,
                            'historicData': [
                                {'period': point.period, 'quantity': point.quantity}
                                for point in result.historic_data
                            ],
                            'forecastData': [
                                {'period': point.period, 'quantity': point.quantity}
                                for point in result.forecast_data
                            ]
                        }
                        
                        # Generate config hash
                        config_str = json.dumps(config, sort_keys=True)
                        response['configHash'] = hashlib.sha256(config_str.encode()).hexdigest()
                        
                        return response
            
            # Run algorithm
            algorithm_func = self.algorithms[algorithm]
            result = algorithm_func(df, config, external_factors)
            
            if not result:
                raise ValueError(f"Algorithm {algorithm} failed to generate forecast")
            
            # Save model to cache
            if result.model:
                metrics = {
                    'accuracy': result.accuracy,
                    'mae': result.mae,
                    'rmse': result.rmse
                }
                ModelPersistenceManager.save_model(
                    self.db, result.model, algorithm, config, training_data, metrics
                )
            
            # Prepare response
            response = {
                'selectedAlgorithm': result.algorithm,
                'accuracy': result.accuracy,
                'mae': result.mae,
                'rmse': result.rmse,
                'trend': result.trend,
                'historicData': [
                    {'period': point.period, 'quantity': point.quantity}
                    for point in result.historic_data
                ],
                'forecastData': [
                    {'period': point.period, 'quantity': point.quantity}
                    for point in result.forecast_data
                ]
            }
            
            # Generate config hash
            config_str = json.dumps(config, sort_keys=True)
            response['configHash'] = hashlib.sha256(config_str.encode()).hexdigest()
            
            return response
            
        except Exception as e:
            logger.error(f"Single algorithm forecast error: {e}")
            raise e
    
    def _use_cached_model(self, model: Any, df: pd.DataFrame, config: Dict[str, Any], algorithm: str) -> Optional[AlgorithmResult]:
        """Use a cached model for prediction"""
        try:
            # This is a simplified implementation
            # In a real scenario, you'd need to handle different model types appropriately
            
            historic_periods = config.get('historicPeriod', 12)
            forecast_periods = config.get('forecastPeriod', 6)
            
            # Use last N periods for historic data
            historic_data = df.tail(historic_periods)
            
            # Generate forecast periods
            last_date = pd.to_datetime(df['date'].iloc[-1])
            interval = config.get('interval', 'month')
            
            forecast_dates = []
            for i in range(1, forecast_periods + 1):
                if interval == 'month':
                    next_date = last_date + pd.DateOffset(months=i)
                elif interval == 'week':
                    next_date = last_date + pd.DateOffset(weeks=i)
                elif interval == 'year':
                    next_date = last_date + pd.DateOffset(years=i)
                else:
                    next_date = last_date + pd.DateOffset(months=i)
                
                forecast_dates.append(next_date)
            
            # Simple forecast using trend (this would be replaced with actual model prediction)
            recent_values = historic_data['quantity'].values
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            forecast_values = []
            for i in range(forecast_periods):
                forecast_value = recent_values[-1] + trend * (i + 1)
                forecast_values.append(max(0, forecast_value))  # Ensure non-negative
            
            # Calculate metrics (simplified)
            mae = np.mean(np.abs(recent_values - np.mean(recent_values)))
            rmse = np.sqrt(np.mean((recent_values - np.mean(recent_values)) ** 2))
            accuracy = max(0, 100 - (mae / np.mean(recent_values) * 100))
            
            # Determine trend
            if trend > 0.1:
                trend_direction = 'increasing'
            elif trend < -0.1:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
            
            # Create result
            historic_points = [
                ForecastPoint(
                    period=row['period_str'],
                    quantity=row['quantity']
                )
                for _, row in historic_data.iterrows()
            ]
            
            forecast_points = [
                ForecastPoint(
                    period=date.strftime('%Y-%m') if interval == 'month' else date.strftime('%Y-%W') if interval == 'week' else date.strftime('%Y'),
                    quantity=value
                )
                for date, value in zip(forecast_dates, forecast_values)
            ]
            
            return AlgorithmResult(
                algorithm=algorithm,
                accuracy=accuracy,
                mae=mae,
                rmse=rmse,
                historic_data=historic_points,
                forecast_data=forecast_points,
                trend=trend_direction,
                model=model
            )
            
        except Exception as e:
            logger.error(f"Error using cached model: {e}")
            return None
    
    # Algorithm implementations
    def _linear_regression(self, df: pd.DataFrame, config: Dict[str, Any], external_factors: Optional[pd.DataFrame] = None) -> Optional[AlgorithmResult]:
        """Linear regression forecasting"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            historic_periods = config.get('historicPeriod', 12)
            forecast_periods = config.get('forecastPeriod', 6)
            
            # Prepare data
            data = df.tail(historic_periods + forecast_periods).copy()
            
            if len(data) < historic_periods:
                return None
            
            # Use last historic_periods for training
            train_data = data.head(historic_periods)
            
            # Prepare features
            X = np.arange(len(train_data)).reshape(-1, 1)
            y = train_data['quantity'].values
            
            # Add external factors if available
            if external_factors is not None:
                # Merge external factors with training data
                train_data_with_factors = train_data.merge(
                    external_factors, on='date', how='left'
                )
                
                # Fill missing values
                factor_columns = [col for col in external_factors.columns if col != 'date']
                for col in factor_columns:
                    if col in train_data_with_factors.columns:
                        train_data_with_factors[col] = train_data_with_factors[col].fillna(
                            train_data_with_factors[col].mean()
                        )
                        X = np.column_stack([X, train_data_with_factors[col].values])
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate predictions for forecast periods
            forecast_X = np.arange(len(train_data), len(train_data) + forecast_periods).reshape(-1, 1)
            
            # Add external factors for forecast if available
            if external_factors is not None and len(factor_columns) > 0:
                # Use last known values for external factors (simplified approach)
                last_factors = []
                for col in factor_columns:
                    if col in train_data_with_factors.columns:
                        last_value = train_data_with_factors[col].iloc[-1]
                        last_factors.extend([last_value] * forecast_periods)
                
                if last_factors:
                    factor_matrix = np.array(last_factors).reshape(forecast_periods, len(factor_columns))
                    forecast_X = np.column_stack([forecast_X, factor_matrix])
            
            forecast_values = model.predict(forecast_X)
            forecast_values = np.maximum(forecast_values, 0)  # Ensure non-negative
            
            # Calculate metrics
            train_pred = model.predict(X)
            mae = mean_absolute_error(y, train_pred)
            rmse = np.sqrt(mean_squared_error(y, train_pred))
            accuracy = max(0, 100 - (mae / np.mean(y) * 100))
            
            # Determine trend
            slope = model.coef_[0] if len(model.coef_) > 0 else 0
            if slope > 0.1:
                trend = 'increasing'
            elif slope < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Generate periods
            last_date = pd.to_datetime(train_data['date'].iloc[-1])
            interval = config.get('interval', 'month')
            
            forecast_periods_list = []
            for i in range(1, forecast_periods + 1):
                if interval == 'month':
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                elif interval == 'week':
                    next_date = last_date + pd.DateOffset(weeks=i)
                    period_str = next_date.strftime('%Y-W%U')
                elif interval == 'year':
                    next_date = last_date + pd.DateOffset(years=i)
                    period_str = next_date.strftime('%Y')
                else:
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                
                forecast_periods_list.append(period_str)
            
            # Create result
            historic_points = [
                ForecastPoint(period=row['period_str'], quantity=row['quantity'])
                for _, row in train_data.iterrows()
            ]
            
            forecast_points = [
                ForecastPoint(period=period, quantity=value)
                for period, value in zip(forecast_periods_list, forecast_values)
            ]
            
            return AlgorithmResult(
                algorithm='Linear Regression',
                accuracy=accuracy,
                mae=mae,
                rmse=rmse,
                historic_data=historic_points,
                forecast_data=forecast_points,
                trend=trend,
                model=model
            )
            
        except Exception as e:
            logger.error(f"Linear regression error: {e}")
            return None
    
    def _polynomial_regression(self, df: pd.DataFrame, config: Dict[str, Any], external_factors: Optional[pd.DataFrame] = None) -> Optional[AlgorithmResult]:
        """Polynomial regression forecasting"""
        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            historic_periods = config.get('historicPeriod', 12)
            forecast_periods = config.get('forecastPeriod', 6)
            
            # Prepare data
            data = df.tail(historic_periods + forecast_periods).copy()
            
            if len(data) < historic_periods:
                return None
            
            train_data = data.head(historic_periods)
            
            # Prepare features
            X = np.arange(len(train_data)).reshape(-1, 1)
            y = train_data['quantity'].values
            
            # Create polynomial pipeline
            degree = min(3, len(train_data) // 3)  # Avoid overfitting
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            
            model.fit(X, y)
            
            # Generate predictions
            forecast_X = np.arange(len(train_data), len(train_data) + forecast_periods).reshape(-1, 1)
            forecast_values = model.predict(forecast_X)
            forecast_values = np.maximum(forecast_values, 0)
            
            # Calculate metrics
            train_pred = model.predict(X)
            mae = mean_absolute_error(y, train_pred)
            rmse = np.sqrt(mean_squared_error(y, train_pred))
            accuracy = max(0, 100 - (mae / np.mean(y) * 100))
            
            # Determine trend (simplified)
            recent_slope = (forecast_values[-1] - y[-1]) / forecast_periods
            if recent_slope > np.std(y) * 0.1:
                trend = 'increasing'
            elif recent_slope < -np.std(y) * 0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Generate periods
            last_date = pd.to_datetime(train_data['date'].iloc[-1])
            interval = config.get('interval', 'month')
            
            forecast_periods_list = []
            for i in range(1, forecast_periods + 1):
                if interval == 'month':
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                elif interval == 'week':
                    next_date = last_date + pd.DateOffset(weeks=i)
                    period_str = next_date.strftime('%Y-W%U')
                elif interval == 'year':
                    next_date = last_date + pd.DateOffset(years=i)
                    period_str = next_date.strftime('%Y')
                else:
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                
                forecast_periods_list.append(period_str)
            
            # Create result
            historic_points = [
                ForecastPoint(period=row['period_str'], quantity=row['quantity'])
                for _, row in train_data.iterrows()
            ]
            
            forecast_points = [
                ForecastPoint(period=period, quantity=value)
                for period, value in zip(forecast_periods_list, forecast_values)
            ]
            
            return AlgorithmResult(
                algorithm='Polynomial Regression',
                accuracy=accuracy,
                mae=mae,
                rmse=rmse,
                historic_data=historic_points,
                forecast_data=forecast_points,
                trend=trend,
                model=model
            )
            
        except Exception as e:
            logger.error(f"Polynomial regression error: {e}")
            return None
    
    def _exponential_smoothing(self, df: pd.DataFrame, config: Dict[str, Any], external_factors: Optional[pd.DataFrame] = None) -> Optional[AlgorithmResult]:
        """Exponential smoothing forecasting"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            historic_periods = config.get('historicPeriod', 12)
            forecast_periods = config.get('forecastPeriod', 6)
            
            # Prepare data
            data = df.tail(historic_periods + forecast_periods).copy()
            
            if len(data) < historic_periods:
                return None
            
            train_data = data.head(historic_periods)
            y = train_data['quantity'].values
            
            # Fit exponential smoothing model
            model = ExponentialSmoothing(y, trend='add', seasonal=None)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_values = fitted_model.forecast(forecast_periods)
            forecast_values = np.maximum(forecast_values, 0)
            
            # Calculate metrics
            fitted_values = fitted_model.fittedvalues
            mae = np.mean(np.abs(y[1:] - fitted_values))  # Skip first value as it's not fitted
            rmse = np.sqrt(np.mean((y[1:] - fitted_values) ** 2))
            accuracy = max(0, 100 - (mae / np.mean(y) * 100))
            
            # Determine trend
            if hasattr(fitted_model, 'slope') and fitted_model.slope:
                if fitted_model.slope > 0.1:
                    trend = 'increasing'
                elif fitted_model.slope < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                # Fallback trend calculation
                recent_trend = np.polyfit(range(len(y)), y, 1)[0]
                if recent_trend > 0.1:
                    trend = 'increasing'
                elif recent_trend < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            
            # Generate periods
            last_date = pd.to_datetime(train_data['date'].iloc[-1])
            interval = config.get('interval', 'month')
            
            forecast_periods_list = []
            for i in range(1, forecast_periods + 1):
                if interval == 'month':
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                elif interval == 'week':
                    next_date = last_date + pd.DateOffset(weeks=i)
                    period_str = next_date.strftime('%Y-W%U')
                elif interval == 'year':
                    next_date = last_date + pd.DateOffset(years=i)
                    period_str = next_date.strftime('%Y')
                else:
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                
                forecast_periods_list.append(period_str)
            
            # Create result
            historic_points = [
                ForecastPoint(period=row['period_str'], quantity=row['quantity'])
                for _, row in train_data.iterrows()
            ]
            
            forecast_points = [
                ForecastPoint(period=period, quantity=value)
                for period, value in zip(forecast_periods_list, forecast_values)
            ]
            
            return AlgorithmResult(
                algorithm='Exponential Smoothing',
                accuracy=accuracy,
                mae=mae,
                rmse=rmse,
                historic_data=historic_points,
                forecast_data=forecast_points,
                trend=trend,
                model=fitted_model
            )
            
        except Exception as e:
            logger.error(f"Exponential smoothing error: {e}")
            return None
    
    def _holt_winters(self, df: pd.DataFrame, config: Dict[str, Any], external_factors: Optional[pd.DataFrame] = None) -> Optional[AlgorithmResult]:
        """Holt-Winters forecasting"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            historic_periods = config.get('historicPeriod', 12)
            forecast_periods = config.get('forecastPeriod', 6)
            
            # Prepare data
            data = df.tail(historic_periods + forecast_periods).copy()
            
            if len(data) < historic_periods:
                return None
            
            train_data = data.head(historic_periods)
            y = train_data['quantity'].values
            
            # Determine seasonality
            seasonal_periods = 12 if config.get('interval') == 'month' else 4 if config.get('interval') == 'quarter' else None
            
            if len(y) >= 2 * seasonal_periods if seasonal_periods else 6:
                # Fit Holt-Winters model with seasonality
                model = ExponentialSmoothing(
                    y, 
                    trend='add', 
                    seasonal='add' if seasonal_periods else None,
                    seasonal_periods=seasonal_periods
                )
            else:
                # Fallback to simple exponential smoothing
                model = ExponentialSmoothing(y, trend='add', seasonal=None)
            
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_values = fitted_model.forecast(forecast_periods)
            forecast_values = np.maximum(forecast_values, 0)
            
            # Calculate metrics
            fitted_values = fitted_model.fittedvalues
            mae = np.mean(np.abs(y[1:] - fitted_values))
            rmse = np.sqrt(np.mean((y[1:] - fitted_values) ** 2))
            accuracy = max(0, 100 - (mae / np.mean(y) * 100))
            
            # Determine trend
            if hasattr(fitted_model, 'slope') and fitted_model.slope:
                if fitted_model.slope > 0.1:
                    trend = 'increasing'
                elif fitted_model.slope < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                recent_trend = np.polyfit(range(len(y)), y, 1)[0]
                if recent_trend > 0.1:
                    trend = 'increasing'
                elif recent_trend < -0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            
            # Generate periods
            last_date = pd.to_datetime(train_data['date'].iloc[-1])
            interval = config.get('interval', 'month')
            
            forecast_periods_list = []
            for i in range(1, forecast_periods + 1):
                if interval == 'month':
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                elif interval == 'week':
                    next_date = last_date + pd.DateOffset(weeks=i)
                    period_str = next_date.strftime('%Y-W%U')
                elif interval == 'year':
                    next_date = last_date + pd.DateOffset(years=i)
                    period_str = next_date.strftime('%Y')
                else:
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                
                forecast_periods_list.append(period_str)
            
            # Create result
            historic_points = [
                ForecastPoint(period=row['period_str'], quantity=row['quantity'])
                for _, row in train_data.iterrows()
            ]
            
            forecast_points = [
                ForecastPoint(period=period, quantity=value)
                for period, value in zip(forecast_periods_list, forecast_values)
            ]
            
            return AlgorithmResult(
                algorithm='Holt-Winters',
                accuracy=accuracy,
                mae=mae,
                rmse=rmse,
                historic_data=historic_points,
                forecast_data=forecast_points,
                trend=trend,
                model=fitted_model
            )
            
        except Exception as e:
            logger.error(f"Holt-Winters error: {e}")
            return None
    
    def _arima(self, df: pd.DataFrame, config: Dict[str, Any], external_factors: Optional[pd.DataFrame] = None) -> Optional[AlgorithmResult]:
        """ARIMA forecasting"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
            
            historic_periods = config.get('historicPeriod', 12)
            forecast_periods = config.get('forecastPeriod', 6)
            
            # Prepare data
            data = df.tail(historic_periods + forecast_periods).copy()
            
            if len(data) < historic_periods:
                return None
            
            train_data = data.head(historic_periods)
            y = train_data['quantity'].values
            
            # Simple ARIMA parameter selection
            # Check stationarity
            adf_result = adfuller(y)
            d = 0 if adf_result[1] < 0.05 else 1
            
            # Use simple parameters
            p, d, q = 1, d, 1
            
            # Fit ARIMA model
            model = ARIMA(y, order=(p, d, q))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_result = fitted_model.forecast(steps=forecast_periods)
            forecast_values = np.maximum(forecast_result, 0)
            
            # Calculate metrics
            fitted_values = fitted_model.fittedvalues
            mae = np.mean(np.abs(y - fitted_values))
            rmse = np.sqrt(np.mean((y - fitted_values) ** 2))
            accuracy = max(0, 100 - (mae / np.mean(y) * 100))
            
            # Determine trend
            recent_trend = np.polyfit(range(len(y)), y, 1)[0]
            if recent_trend > 0.1:
                trend = 'increasing'
            elif recent_trend < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Generate periods
            last_date = pd.to_datetime(train_data['date'].iloc[-1])
            interval = config.get('interval', 'month')
            
            forecast_periods_list = []
            for i in range(1, forecast_periods + 1):
                if interval == 'month':
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                elif interval == 'week':
                    next_date = last_date + pd.DateOffset(weeks=i)
                    period_str = next_date.strftime('%Y-W%U')
                elif interval == 'year':
                    next_date = last_date + pd.DateOffset(years=i)
                    period_str = next_date.strftime('%Y')
                else:
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                
                forecast_periods_list.append(period_str)
            
            # Create result
            historic_points = [
                ForecastPoint(period=row['period_str'], quantity=row['quantity'])
                for _, row in train_data.iterrows()
            ]
            
            forecast_points = [
                ForecastPoint(period=period, quantity=value)
                for period, value in zip(forecast_periods_list, forecast_values)
            ]
            
            return AlgorithmResult(
                algorithm='ARIMA',
                accuracy=accuracy,
                mae=mae,
                rmse=rmse,
                historic_data=historic_points,
                forecast_data=forecast_points,
                trend=trend,
                model=fitted_model
            )
            
        except Exception as e:
            logger.error(f"ARIMA error: {e}")
            return None
    
    def _random_forest(self, df: pd.DataFrame, config: Dict[str, Any], external_factors: Optional[pd.DataFrame] = None) -> Optional[AlgorithmResult]:
        """Random Forest forecasting"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            historic_periods = config.get('historicPeriod', 12)
            forecast_periods = config.get('forecastPeriod', 6)
            
            # Prepare data
            data = df.tail(historic_periods + forecast_periods).copy()
            
            if len(data) < historic_periods:
                return None
            
            train_data = data.head(historic_periods)
            y = train_data['quantity'].values
            
            # Create features (time index, lags, etc.)
            X = []
            for i in range(len(y)):
                features = [i]  # Time index
                
                # Add lag features
                for lag in [1, 2, 3]:
                    if i >= lag:
                        features.append(y[i - lag])
                    else:
                        features.append(0)
                
                # Add moving averages
                if i >= 3:
                    features.append(np.mean(y[max(0, i-3):i]))
                else:
                    features.append(y[0] if i == 0 else np.mean(y[:i]))
                
                X.append(features)
            
            X = np.array(X)
            
            # Add external factors if available
            if external_factors is not None:
                train_data_with_factors = train_data.merge(
                    external_factors, on='date', how='left'
                )
                
                factor_columns = [col for col in external_factors.columns if col != 'date']
                for col in factor_columns:
                    if col in train_data_with_factors.columns:
                        factor_values = train_data_with_factors[col].fillna(
                            train_data_with_factors[col].mean()
                        ).values
                        X = np.column_stack([X, factor_values])
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Generate forecast
            forecast_values = []
            last_values = y.copy()
            
            for i in range(forecast_periods):
                # Create features for next prediction
                time_idx = len(y) + i
                features = [time_idx]
                
                # Add lag features
                for lag in [1, 2, 3]:
                    if len(last_values) >= lag:
                        features.append(last_values[-lag])
                    else:
                        features.append(0)
                
                # Add moving average
                features.append(np.mean(last_values[-3:]) if len(last_values) >= 3 else np.mean(last_values))
                
                # Add external factors (use last known values)
                if external_factors is not None:
                    for col in factor_columns:
                        if col in train_data_with_factors.columns:
                            last_factor_value = train_data_with_factors[col].iloc[-1]
                            features.append(last_factor_value)
                
                # Predict
                pred = model.predict([features])[0]
                pred = max(0, pred)  # Ensure non-negative
                
                forecast_values.append(pred)
                last_values = np.append(last_values, pred)
            
            # Calculate metrics
            train_pred = model.predict(X)
            mae = mean_absolute_error(y, train_pred)
            rmse = np.sqrt(mean_squared_error(y, train_pred))
            accuracy = max(0, 100 - (mae / np.mean(y) * 100))
            
            # Determine trend
            recent_trend = np.polyfit(range(len(y)), y, 1)[0]
            if recent_trend > 0.1:
                trend = 'increasing'
            elif recent_trend < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Generate periods
            last_date = pd.to_datetime(train_data['date'].iloc[-1])
            interval = config.get('interval', 'month')
            
            forecast_periods_list = []
            for i in range(1, forecast_periods + 1):
                if interval == 'month':
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                elif interval == 'week':
                    next_date = last_date + pd.DateOffset(weeks=i)
                    period_str = next_date.strftime('%Y-W%U')
                elif interval == 'year':
                    next_date = last_date + pd.DateOffset(years=i)
                    period_str = next_date.strftime('%Y')
                else:
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                
                forecast_periods_list.append(period_str)
            
            # Create result
            historic_points = [
                ForecastPoint(period=row['period_str'], quantity=row['quantity'])
                for _, row in train_data.iterrows()
            ]
            
            forecast_points = [
                ForecastPoint(period=period, quantity=value)
                for period, value in zip(forecast_periods_list, forecast_values)
            ]
            
            return AlgorithmResult(
                algorithm='Random Forest',
                accuracy=accuracy,
                mae=mae,
                rmse=rmse,
                historic_data=historic_points,
                forecast_data=forecast_points,
                trend=trend,
                model=model
            )
            
        except Exception as e:
            logger.error(f"Random Forest error: {e}")
            return None
    
    def _seasonal_decomposition(self, df: pd.DataFrame, config: Dict[str, Any], external_factors: Optional[pd.DataFrame] = None) -> Optional[AlgorithmResult]:
        """Seasonal decomposition forecasting"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            from sklearn.linear_model import LinearRegression
            
            historic_periods = config.get('historicPeriod', 12)
            forecast_periods = config.get('forecastPeriod', 6)
            
            # Prepare data
            data = df.tail(historic_periods + forecast_periods).copy()
            
            if len(data) < historic_periods:
                return None
            
            train_data = data.head(historic_periods)
            y = train_data['quantity'].values
            
            # Perform seasonal decomposition
            if len(y) >= 8:  # Minimum for decomposition
                ts = pd.Series(y, index=pd.date_range(start='2020-01-01', periods=len(y), freq='M'))
                decomposition = seasonal_decompose(ts, model='additive', period=min(12, len(y)//2))
                
                trend = decomposition.trend.dropna().values
                seasonal = decomposition.seasonal.values
                residual = decomposition.resid.dropna().values
            else:
                # Fallback for short series
                trend = y
                seasonal = np.zeros_like(y)
                residual = np.zeros_like(y)
            
            # Forecast trend using linear regression
            if len(trend) > 2:
                X_trend = np.arange(len(trend)).reshape(-1, 1)
                trend_model = LinearRegression()
                trend_model.fit(X_trend, trend)
                
                # Forecast trend
                forecast_X = np.arange(len(trend), len(trend) + forecast_periods).reshape(-1, 1)
                forecast_trend = trend_model.predict(forecast_X)
            else:
                # Simple trend extrapolation
                if len(y) > 1:
                    trend_slope = (y[-1] - y[0]) / (len(y) - 1)
                    forecast_trend = [y[-1] + trend_slope * (i + 1) for i in range(forecast_periods)]
                else:
                    forecast_trend = [y[-1]] * forecast_periods
            
            # Forecast seasonal component (repeat pattern)
            if len(seasonal) > 0:
                seasonal_period = min(12, len(seasonal))
                forecast_seasonal = [seasonal[i % seasonal_period] for i in range(forecast_periods)]
            else:
                forecast_seasonal = [0] * forecast_periods
            
            # Combine trend and seasonal
            forecast_values = np.array(forecast_trend) + np.array(forecast_seasonal)
            forecast_values = np.maximum(forecast_values, 0)
            
            # Calculate metrics
            if len(trend) == len(y):
                fitted_values = trend + seasonal
            else:
                fitted_values = y  # Fallback
            
            mae = np.mean(np.abs(y - fitted_values))
            rmse = np.sqrt(np.mean((y - fitted_values) ** 2))
            accuracy = max(0, 100 - (mae / np.mean(y) * 100))
            
            # Determine trend
            if len(trend) > 1:
                trend_slope = np.polyfit(range(len(trend)), trend, 1)[0]
                if trend_slope > 0.1:
                    trend_direction = 'increasing'
                elif trend_slope < -0.1:
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'stable'
            
            # Generate periods
            last_date = pd.to_datetime(train_data['date'].iloc[-1])
            interval = config.get('interval', 'month')
            
            forecast_periods_list = []
            for i in range(1, forecast_periods + 1):
                if interval == 'month':
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                elif interval == 'week':
                    next_date = last_date + pd.DateOffset(weeks=i)
                    period_str = next_date.strftime('%Y-W%U')
                elif interval == 'year':
                    next_date = last_date + pd.DateOffset(years=i)
                    period_str = next_date.strftime('%Y')
                else:
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                
                forecast_periods_list.append(period_str)
            
            # Create result
            historic_points = [
                ForecastPoint(period=row['period_str'], quantity=row['quantity'])
                for _, row in train_data.iterrows()
            ]
            
            forecast_points = [
                ForecastPoint(period=period, quantity=value)
                for period, value in zip(forecast_periods_list, forecast_values)
            ]
            
            return AlgorithmResult(
                algorithm='Seasonal Decomposition',
                accuracy=accuracy,
                mae=mae,
                rmse=rmse,
                historic_data=historic_points,
                forecast_data=forecast_points,
                trend=trend_direction,
                model=None  # Decomposition doesn't have a single model object
            )
            
        except Exception as e:
            logger.error(f"Seasonal decomposition error: {e}")
            return None
    
    def _moving_average(self, df: pd.DataFrame, config: Dict[str, Any], external_factors: Optional[pd.DataFrame] = None) -> Optional[AlgorithmResult]:
        """Moving average forecasting"""
        try:
            historic_periods = config.get('historicPeriod', 12)
            forecast_periods = config.get('forecastPeriod', 6)
            
            # Prepare data
            data = df.tail(historic_periods + forecast_periods).copy()
            
            if len(data) < historic_periods:
                return None
            
            train_data = data.head(historic_periods)
            y = train_data['quantity'].values
            
            # Calculate moving average window
            window = min(6, len(y) // 2)
            if window < 2:
                window = 2
            
            # Calculate moving averages
            moving_avg = pd.Series(y).rolling(window=window).mean()
            
            # Generate forecast (use last moving average value with trend)
            last_ma = moving_avg.iloc[-1]
            
            # Calculate trend from recent data
            recent_data = y[-window:]
            if len(recent_data) > 1:
                trend_slope = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            else:
                trend_slope = 0
            
            # Generate forecast values
            forecast_values = []
            for i in range(forecast_periods):
                forecast_value = last_ma + trend_slope * (i + 1)
                forecast_values.append(max(0, forecast_value))
            
            # Calculate metrics
            fitted_values = moving_avg.fillna(method='bfill').values
            mae = np.mean(np.abs(y - fitted_values))
            rmse = np.sqrt(np.mean((y - fitted_values) ** 2))
            accuracy = max(0, 100 - (mae / np.mean(y) * 100))
            
            # Determine trend
            if trend_slope > 0.1:
                trend = 'increasing'
            elif trend_slope < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Generate periods
            last_date = pd.to_datetime(train_data['date'].iloc[-1])
            interval = config.get('interval', 'month')
            
            forecast_periods_list = []
            for i in range(1, forecast_periods + 1):
                if interval == 'month':
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                elif interval == 'week':
                    next_date = last_date + pd.DateOffset(weeks=i)
                    period_str = next_date.strftime('%Y-W%U')
                elif interval == 'year':
                    next_date = last_date + pd.DateOffset(years=i)
                    period_str = next_date.strftime('%Y')
                else:
                    next_date = last_date + pd.DateOffset(months=i)
                    period_str = next_date.strftime('%Y-%m')
                
                forecast_periods_list.append(period_str)
            
            # Create result
            historic_points = [
                ForecastPoint(period=row['period_str'], quantity=row['quantity'])
                for _, row in train_data.iterrows()
            ]
            
            forecast_points = [
                ForecastPoint(period=period, quantity=value)
                for period, value in zip(forecast_periods_list, forecast_values)
            ]
            
            return AlgorithmResult(
                algorithm='Moving Average',
                accuracy=accuracy,
                mae=mae,
                rmse=rmse,
                historic_data=historic_points,
                forecast_data=forecast_points,
                trend=trend,
                model={'window': window, 'trend_slope': trend_slope}
            )
            
        except Exception as e:
            logger.error(f"Moving average error: {e}")
            return None