#!/usr/bin/env python3
"""
Forecast scheduling system for automated forecast generation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from database import Base, SessionLocal, ForecastConfiguration
import json
import threading
import time
from enum import Enum as PyEnum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScheduleFrequency(PyEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class ScheduleStatus(PyEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class ScheduledForecast(Base):
    """Model for storing scheduled forecast configurations"""
    __tablename__ = "scheduled_forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    forecast_config = Column(Text, nullable=False)  # JSON string of ForecastConfig
    frequency = Column(Enum(ScheduleFrequency), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)  # Optional end date
    next_run = Column(DateTime, nullable=False)
    last_run = Column(DateTime, nullable=True)
    status = Column(Enum(ScheduleStatus), default=ScheduleStatus.ACTIVE)
    run_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ForecastExecution(Base):
    """Model for storing forecast execution history"""
    __tablename__ = "forecast_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    scheduled_forecast_id = Column(Integer, nullable=False, index=True)
    execution_time = Column(DateTime, nullable=False)
    status = Column(String(50), nullable=False)  # success, failed, running
    duration_seconds = Column(Integer, nullable=True)
    result_summary = Column(Text, nullable=True)  # JSON summary of results
    error_message = Column(Text, nullable=True)
    forecast_data = Column(Text, nullable=True)  # JSON of forecast results
    created_at = Column(DateTime, default=datetime.utcnow)

class ForecastScheduler:
    """Main scheduler class for managing automated forecasts"""
    
    def __init__(self):
        self.running = False
        self.scheduler_thread = None
        self.check_interval = 60  # Check every minute
        
    def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
            
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Forecast scheduler started")
        
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Forecast scheduler stopped")
        
    def _run_scheduler(self):
        """Main scheduler loop"""
        while self.running:
            try:
                self._check_and_execute_forecasts()
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
            
            time.sleep(self.check_interval)
            
    def _check_and_execute_forecasts(self):
        """Check for forecasts that need to be executed"""
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            
            # Get all active scheduled forecasts that are due
            due_forecasts = db.query(ScheduledForecast).filter(
                ScheduledForecast.status == ScheduleStatus.ACTIVE,
                ScheduledForecast.next_run <= now
            ).all()
            
            for scheduled_forecast in due_forecasts:
                try:
                    self._execute_forecast(db, scheduled_forecast)
                except Exception as e:
                    logger.error(f"Error executing forecast {scheduled_forecast.id}: {e}")
                    self._record_execution_failure(db, scheduled_forecast, str(e))
                    
        finally:
            db.close()
            
    def _execute_forecast(self, db: Session, scheduled_forecast: ScheduledForecast):
        """Execute a single scheduled forecast"""
        logger.info(f"Executing scheduled forecast: {scheduled_forecast.name}")
        
        start_time = datetime.utcnow()
        execution = ForecastExecution(
            scheduled_forecast_id=scheduled_forecast.id,
            execution_time=start_time,
            status="running"
        )
        db.add(execution)
        db.commit()
        
        try:
            # Parse forecast configuration
            forecast_config = json.loads(scheduled_forecast.forecast_config)
            
            # Import and execute forecast generation
            from main import generate_forecast_internal
            
            # Execute the forecast
            result = generate_forecast_internal(forecast_config, db)
            
            # Calculate execution duration
            end_time = datetime.utcnow()
            duration = int((end_time - start_time).total_seconds())
            
            # Create result summary
            if isinstance(result, dict) and 'results' in result:
                # Multi-forecast result
                summary = {
                    "type": "multi_forecast",
                    "total_combinations": result.get("totalCombinations", 0),
                    "successful": result.get("summary", {}).get("successfulCombinations", 0),
                    "failed": result.get("summary", {}).get("failedCombinations", 0),
                    "average_accuracy": result.get("summary", {}).get("averageAccuracy", 0)
                }
            else:
                # Single forecast result
                summary = {
                    "type": "single_forecast",
                    "accuracy": result.get("accuracy", 0),
                    "algorithm": result.get("selectedAlgorithm", "unknown"),
                    "mae": result.get("mae", 0),
                    "rmse": result.get("rmse", 0)
                }
            
            # Update execution record
            execution.status = "success"
            execution.duration_seconds = duration
            execution.result_summary = json.dumps(summary)
            execution.forecast_data = json.dumps(result)
            
            # Update scheduled forecast
            scheduled_forecast.last_run = start_time
            scheduled_forecast.run_count += 1
            scheduled_forecast.success_count += 1
            scheduled_forecast.next_run = self._calculate_next_run(
                scheduled_forecast.frequency, 
                start_time
            )
            scheduled_forecast.last_error = None
            
            # Check if schedule should be completed
            if (scheduled_forecast.end_date and 
                scheduled_forecast.next_run > scheduled_forecast.end_date):
                scheduled_forecast.status = ScheduleStatus.COMPLETED
            
            db.commit()
            logger.info(f"Successfully executed forecast: {scheduled_forecast.name}")
            
        except Exception as e:
            # Update execution record with failure
            execution.status = "failed"
            execution.error_message = str(e)
            execution.duration_seconds = int((datetime.utcnow() - start_time).total_seconds())
            
            # Update scheduled forecast
            scheduled_forecast.last_run = start_time
            scheduled_forecast.run_count += 1
            scheduled_forecast.failure_count += 1
            scheduled_forecast.last_error = str(e)
            scheduled_forecast.next_run = self._calculate_next_run(
                scheduled_forecast.frequency, 
                start_time
            )
            
            db.commit()
            raise e
            
    def _record_execution_failure(self, db: Session, scheduled_forecast: ScheduledForecast, error_message: str):
        """Record a forecast execution failure"""
        execution = ForecastExecution(
            scheduled_forecast_id=scheduled_forecast.id,
            execution_time=datetime.utcnow(),
            status="failed",
            error_message=error_message
        )
        db.add(execution)
        
        scheduled_forecast.failure_count += 1
        scheduled_forecast.last_error = error_message
        
        db.commit()
        
    def _calculate_next_run(self, frequency: ScheduleFrequency, last_run: datetime) -> datetime:
        """Calculate the next run time based on frequency"""
        if frequency == ScheduleFrequency.DAILY:
            return last_run + timedelta(days=1)
        elif frequency == ScheduleFrequency.WEEKLY:
            return last_run + timedelta(weeks=1)
        elif frequency == ScheduleFrequency.MONTHLY:
            # Add one month (approximate)
            if last_run.month == 12:
                return last_run.replace(year=last_run.year + 1, month=1)
            else:
                return last_run.replace(month=last_run.month + 1)
        else:
            raise ValueError(f"Unknown frequency: {frequency}")

# Global scheduler instance
scheduler = ForecastScheduler()

def start_scheduler():
    """Start the global scheduler"""
    scheduler.start()

def stop_scheduler():
    """Stop the global scheduler"""
    scheduler.stop()

def get_scheduler_status():
    """Get the current scheduler status"""
    return {
        "running": scheduler.running,
        "check_interval": scheduler.check_interval
    }