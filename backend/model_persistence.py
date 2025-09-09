
#!/usr/bin/env python3
"""
Model persistence utilities for saving and loading trained forecasting models
"""

import pickle
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from database import Base
import hashlib
import numpy as np

class SavedModel(Base):
    """Model for storing trained forecasting models"""
    __tablename__ = "saved_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_hash = Column(String(64), unique=True, nullable=False, index=True)
    algorithm = Column(String(100), nullable=False)
    config_hash = Column(String(64), nullable=False, index=True)
    model_data = Column(LargeBinary, nullable=False)
    model_metadata = Column(Text, nullable=True)
    accuracy = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, default=datetime.utcnow)
    use_count = Column(Integer, default=0)

class ModelAccuracyHistory(Base):
    """Model for tracking forecast accuracy over time"""
    __tablename__ = "model_accuracy_history"
    
    id = Column(Integer, primary_key=True, index=True)
    model_hash = Column(String(64), nullable=False, index=True)
    algorithm = Column(String(100), nullable=False)
    config_hash = Column(String(64), nullable=False)
    forecast_date = Column(DateTime, nullable=False)
    actual_values = Column(Text, nullable=True)  # JSON array
    predicted_values = Column(Text, nullable=True)  # JSON array
    accuracy = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelPersistenceManager:
    """Manager for saving and loading trained models"""
    
    @staticmethod
    def generate_config_hash(config: Dict[str, Any]) -> str:
        """Generate a hash for the configuration to identify similar setups"""
        if not config or not isinstance(config, dict):
            return "default_config_hash"
        
        # Helper function to safely get and normalize list values
        def get_safe_list(value):
            if value is None:
                return []
            elif isinstance(value, list):
                return sorted(value)
            else:
                return [value]
        
        # Create a normalized config for hashing
        normalized_config = {
            'forecastBy': config.get('forecastBy', ''),
            'selectedItem': config.get('selectedItem', ''),
            'selectedProduct': config.get('selectedProduct', ''),
            'selectedCustomer': config.get('selectedCustomer', ''),
            'selectedLocation': config.get('selectedLocation', ''),
            'algorithm': config.get('algorithm', ''),
            'interval': config.get('interval', ''),
            'historicPeriod': config.get('historicPeriod', 0),
            'forecastPeriod': config.get('forecastPeriod', 0),
            'externalFactors': get_safe_list(config.get('externalFactors'))
        }
        
        config_str = json.dumps(normalized_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    @staticmethod
    def generate_data_hash(data: np.ndarray) -> str:
        """Generate a hash for the training data"""
        if data is None or not hasattr(data, 'tobytes'):
            return "default_data_hash"
        data_str = str(data.tobytes())
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]  # Shorter hash for data
    
    @staticmethod
    def generate_model_hash(algorithm: str, config_hash: str, data_hash: str) -> str:
        """Generate a unique hash for the model"""
        combined = f"{algorithm}_{config_hash}_{data_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    @staticmethod
    def save_model(
        db: Session,
        model: Any,
        algorithm: str,
        config: Dict[str, Any],
        training_data: np.ndarray,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a trained model to the database"""
        try:
            print(f"Debug save_model: Starting to save model")
            print(f"Debug save_model: algorithm={algorithm}")
            print(f"Debug save_model: config={config}")
            print(f"Debug save_model: training_data type={type(training_data)}, length={len(training_data) if training_data is not None else 'None'}")
            print(f"Debug save_model: metrics={metrics}")
            
            config_hash = ModelPersistenceManager.generate_config_hash(config)
            data_hash = ModelPersistenceManager.generate_data_hash(training_data)
            model_hash = ModelPersistenceManager.generate_model_hash(algorithm, config_hash, data_hash)
            
            print(f"Debug save_model: config_hash={config_hash}")
            print(f"Debug save_model: data_hash={data_hash}")
            print(f"Debug save_model: model_hash={model_hash}")
            
            # Check if model already exists
            existing_model = db.query(SavedModel).filter(SavedModel.model_hash == model_hash).first()
            print(f"Debug save_model: existing_model={existing_model}")
            
            if existing_model:
                print(f"Debug save_model: Found existing model, updating use count")
                # Update last used time and use count
                existing_model.last_used = datetime.utcnow()
                existing_model.use_count += 1
                db.commit()
                print(f"Debug save_model: Updated existing model, returning hash={model_hash}")
                return model_hash
            
            print(f"Debug save_model: No existing model found, creating new one")
            
            # Serialize the model
            print(f"Debug save_model: Serializing model with pickle")
            model_data = pickle.dumps(model)
            print(f"Debug save_model: Model serialized, size={len(model_data)} bytes")
            
            # Create new saved model record
            saved_model = SavedModel(
                model_hash=model_hash,
                algorithm=algorithm,
                config_hash=config_hash,
                model_data=model_data,
                model_metadata=json.dumps(metadata) if metadata else None,
                accuracy=metrics.get('accuracy'),
                mae=metrics.get('mae'),
                rmse=metrics.get('rmse'),
                use_count=1
            )
            
            print(f"Debug save_model: Created SavedModel object")
            
            db.add(saved_model)
            print(f"Debug save_model: Added model to session")
            
            try:
                db.commit()
                print(f"Debug save_model: Committed to database")
            except Exception as commit_error:
                print(f"Error committing to database: {commit_error}")
                db.rollback()
                raise commit_error
            
            # Verify the model was saved
            verification = db.query(SavedModel).filter(SavedModel.model_hash == model_hash).first()
            print(f"Debug save_model: Verification query result={verification}")
            
            print(f"Debug save_model: Successfully saved model, returning hash={model_hash}")
            return model_hash
            
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def load_model(db: Session, model_hash: str) -> Optional[Any]:
        """Load a trained model from the database"""
        try:
            saved_model = db.query(SavedModel).filter(SavedModel.model_hash == model_hash).first()
            if not saved_model:
                return None
            
            # Update last used time and use count
            saved_model.last_used = datetime.utcnow()
            saved_model.use_count += 1
            db.commit()
            
            # Deserialize the model
            model = pickle.loads(saved_model.model_data)
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def find_cached_model(
        db: Session,
        algorithm: str,
        config: Dict[str, Any],
        training_data: np.ndarray
    ) -> Optional[str]:
        """Find if a cached model exists for the given parameters"""
        try:
            print(f"Debug find_cached_model: algorithm={algorithm}")
            print(f"Debug find_cached_model: config={config}")
            print(f"Debug find_cached_model: training_data type={type(training_data)}, length={len(training_data) if training_data is not None else 'None'}")
            
            config_hash = ModelPersistenceManager.generate_config_hash(config)
            data_hash = ModelPersistenceManager.generate_data_hash(training_data)
            model_hash = ModelPersistenceManager.generate_model_hash(algorithm, config_hash, data_hash)
            
            print(f"Debug find_cached_model: config_hash={config_hash}")
            print(f"Debug find_cached_model: data_hash={data_hash}")
            print(f"Debug find_cached_model: model_hash={model_hash}")
            
            # Check how many models exist in total
            total_models = db.query(SavedModel).count()
            print(f"Debug find_cached_model: total models in database={total_models}")
            
            # Check if there are any models with the same algorithm
            same_algorithm_models = db.query(SavedModel).filter(SavedModel.algorithm == algorithm).count()
            print(f"Debug find_cached_model: models with same algorithm ({algorithm})={same_algorithm_models}")
            
            # Check if there are any models with the same config_hash
            same_config_models = db.query(SavedModel).filter(SavedModel.config_hash == config_hash).count()
            print(f"Debug find_cached_model: models with same config_hash={same_config_models}")
            
            # Query for the specific model
            saved_model = db.query(SavedModel).filter(SavedModel.model_hash == model_hash).first()
            print(f"Debug find_cached_model: query result saved_model={saved_model}")
            
            # If no exact match, show some existing models for comparison
            if not saved_model and total_models > 0:
                print("Debug find_cached_model: No exact match found. Showing existing models:")
                existing_models = db.query(SavedModel).limit(5).all()
                for i, model in enumerate(existing_models):
                    print(f"  Model {i+1}: hash={model.model_hash}, algorithm={model.algorithm}, config_hash={model.config_hash}")
            
            return model_hash if saved_model else None
            
        except Exception as e:
            print(f"Error finding cached model: {e}")
            return None
    
    @staticmethod
    def record_accuracy_history(
        db: Session,
        model_hash: str,
        algorithm: str,
        config_hash: str,
        actual_values: List[float],
        predicted_values: List[float],
        metrics: Dict[str, float]
    ):
        """Record forecast accuracy for historical tracking"""
        try:
            history_record = ModelAccuracyHistory(
                model_hash=model_hash,
                algorithm=algorithm,
                config_hash=config_hash,
                forecast_date=datetime.utcnow(),
                actual_values=json.dumps(actual_values),
                predicted_values=json.dumps(predicted_values),
                accuracy=metrics.get('accuracy'),
                mae=metrics.get('mae'),
                rmse=metrics.get('rmse')
            )
            
            db.add(history_record)
            db.commit()
            
        except Exception as e:
            print(f"Error recording accuracy history: {e}")
    
    @staticmethod
    def get_accuracy_history(
        db: Session,
        config_hash: str,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get accuracy history for a configuration"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            history = db.query(ModelAccuracyHistory).filter(
                ModelAccuracyHistory.config_hash == config_hash,
                ModelAccuracyHistory.created_at >= cutoff_date
            ).order_by(ModelAccuracyHistory.created_at.desc()).all()
            
            result = []
            for record in history:
                result.append({
                    'algorithm': record.algorithm,
                    'forecast_date': record.forecast_date.isoformat(),
                    'accuracy': record.accuracy,
                    'mae': record.mae,
                    'rmse': record.rmse,
                    'actual_values': json.loads(record.actual_values) if record.actual_values else [],
                    'predicted_values': json.loads(record.predicted_values) if record.predicted_values else []
                })
            
            return result
            
        except Exception as e:
            print(f"Error getting accuracy history: {e}")
            return []
    
    @staticmethod
    def cleanup_old_models(db: Session, days_old: int = 30, max_models: int = 100):
        """Clean up old unused models"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Delete models that haven't been used recently
            old_models = db.query(SavedModel).filter(
                SavedModel.last_used < cutoff_date
            ).order_by(SavedModel.last_used.asc()).all()
            
            # Keep only the most recent models if we exceed max_models
            all_models = db.query(SavedModel).order_by(SavedModel.last_used.desc()).all()
            if len(all_models) > max_models:
                models_to_delete = all_models[max_models:]
                for model in models_to_delete:
                    db.delete(model)
            
            # Delete old models
            for model in old_models:
                db.delete(model)
            
            db.commit()
            
        except Exception as e:
            print(f"Error cleaning up old models: {e}")