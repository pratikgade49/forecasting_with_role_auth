#!/usr/bin/env python3
"""
Validation utilities for external factors and data alignment
"""

from datetime import datetime, date
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from database import ForecastData, ExternalFactorData
import pandas as pd

class DateRangeValidator:
    """Validates date ranges between main data and external factors"""

    @staticmethod
    def get_main_data_date_range(db: Session) -> Tuple[Optional[date], Optional[date]]:
        """Get the date range of main forecast data"""
        min_date = db.query(func.min(ForecastData.date)).scalar()
        max_date = db.query(func.max(ForecastData.date)).scalar()
        return min_date, max_date

    @staticmethod
    def get_external_factor_date_range(db: Session, factor_name: str) -> Tuple[Optional[date], Optional[date]]:
        """Get the date range for a specific external factor"""
        min_date = db.query(func.min(ExternalFactorData.date)).filter(
            ExternalFactorData.factor_name == factor_name
        ).scalar()
        max_date = db.query(func.max(ExternalFactorData.date)).filter(
            ExternalFactorData.factor_name == factor_name
        ).scalar()
        return min_date, max_date

    @staticmethod
    def validate_external_factors(db: Session, factor_names: List[str]) -> Dict[str, any]:
        """Validate external factors against main data date range"""
        validation_results = {}

        # Get main data date range
        main_min, main_max = DateRangeValidator.get_main_data_date_range(db)

        if not main_min or not main_max:
            return {
                "error": "No main forecast data found in database",
                "factor_validations": {}
            }

        for factor_name in factor_names:
            factor_min, factor_max = DateRangeValidator.get_external_factor_date_range(db, factor_name)

            if not factor_min or not factor_max:
                validation_results[factor_name] = {
                    "status": "error",
                    "message": f"No data found for factor '{factor_name}'",
                    "coverage_percentage": 0,
                    "missing_dates": []
                }
                continue

            # Check coverage
            coverage_start = max(main_min, factor_min)
            coverage_end = min(main_max, factor_max)

            if coverage_start > coverage_end:
                # No overlap
                validation_results[factor_name] = {
                    "status": "error",
                    "message": f"No date overlap between main data ({main_min} to {main_max}) and factor '{factor_name}' ({factor_min} to {factor_max})",
                    "coverage_percentage": 0,
                    "main_data_range": {"start": main_min.isoformat(), "end": main_max.isoformat()},
                    "factor_range": {"start": factor_min.isoformat(), "end": factor_max.isoformat()}
                }
                continue

            # Calculate coverage percentage
            total_days = (main_max - main_min).days + 1
            covered_days = (coverage_end - coverage_start).days + 1
            coverage_percentage = (covered_days / total_days) * 100

            # Find missing dates within main data range
            missing_dates = DateRangeValidator.find_missing_dates(db, factor_name, main_min, main_max)

            # Determine status
            if coverage_percentage >= 90 and len(missing_dates) <= total_days * 0.1:
                status = "good"
                message = f"Good coverage: {coverage_percentage:.1f}% of main data range"
            elif coverage_percentage >= 70:
                status = "warning"
                message = f"Partial coverage: {coverage_percentage:.1f}% of main data range. {len(missing_dates)} missing dates."
            else:
                status = "error"
                message = f"Poor coverage: {coverage_percentage:.1f}% of main data range. {len(missing_dates)} missing dates."

            validation_results[factor_name] = {
                "status": status,
                "message": message,
                "coverage_percentage": round(coverage_percentage, 1),
                "main_data_range": {"start": main_min.isoformat(), "end": main_max.isoformat()},
                "factor_range": {"start": factor_min.isoformat(), "end": factor_max.isoformat()},
                "coverage_range": {"start": coverage_start.isoformat(), "end": coverage_end.isoformat()},
                "missing_dates": [d.isoformat() for d in missing_dates[:10]],  # Limit to first 10
                "total_missing": len(missing_dates)
            }

        return validation_results

    @staticmethod
    def find_missing_dates(db: Session, factor_name: str, start_date: date, end_date: date) -> List[date]:
        """Find dates within range where external factor data is missing"""
        # Get all dates where main data exists
        main_dates = db.query(ForecastData.date).filter(
            and_(ForecastData.date >= start_date, ForecastData.date <= end_date)
        ).distinct().all()
        main_dates_set = {d[0] for d in main_dates}

        # Get all dates where external factor exists
        factor_dates = db.query(ExternalFactorData.date).filter(
            and_(
                ExternalFactorData.factor_name == factor_name,
                ExternalFactorData.date >= start_date,
                ExternalFactorData.date <= end_date
            )
        ).distinct().all()
        factor_dates_set = {d[0] for d in factor_dates}

        # Find missing dates
        missing_dates = sorted(main_dates_set - factor_dates_set)
        return missing_dates

    @staticmethod
    def validate_upload_data(df: pd.DataFrame, db: Session) -> Dict[str, any]:
        """Validate external factor data before upload"""
        validation_result = {
            "status": "success",
            "warnings": [],
            "errors": [],
            "summary": {}
        }

        # Get main data date range
        main_min, main_max = DateRangeValidator.get_main_data_date_range(db)

        if not main_min or not main_max:
            validation_result["warnings"].append("No main forecast data found. External factors can still be uploaded.")
            return validation_result

        # Analyze uploaded data
        df['date'] = pd.to_datetime(df['date']).dt.date
        upload_min = df['date'].min()
        upload_max = df['date'].max()

        # Check date range overlap
        if upload_max < main_min or upload_min > main_max:
            validation_result["errors"].append(
                f"Uploaded data date range ({upload_min} to {upload_max}) does not overlap with main data range ({main_min} to {main_max})"
            )
            validation_result["status"] = "error"

        # Check coverage
        overlap_start = max(main_min, upload_min)
        overlap_end = min(main_max, upload_max)

        if overlap_start <= overlap_end:
            total_main_days = (main_max - main_min).days + 1
            overlap_days = (overlap_end - overlap_start).days + 1
            coverage_percentage = (overlap_days / total_main_days) * 100

            validation_result["summary"]["coverage_percentage"] = round(coverage_percentage, 1)
            validation_result["summary"]["overlap_range"] = {
                "start": overlap_start.isoformat(),
                "end": overlap_end.isoformat()
            }

            if coverage_percentage < 50:
                validation_result["warnings"].append(
                    f"Low coverage: Only {coverage_percentage:.1f}% of main data date range is covered"
                )
            elif coverage_percentage < 80:
                validation_result["warnings"].append(
                    f"Partial coverage: {coverage_percentage:.1f}% of main data date range is covered"
                )

        # Analyze by factor
        factor_summary = {}
        for factor_name in df['factor_name'].unique():
            factor_df = df[df['factor_name'] == factor_name]
            factor_min = factor_df['date'].min()
            factor_max = factor_df['date'].max()
            factor_count = len(factor_df)

            factor_summary[factor_name] = {
                "date_range": {"start": factor_min.isoformat(), "end": factor_max.isoformat()},
                "record_count": factor_count,
                "duplicate_dates": len(factor_df) - len(factor_df['date'].unique())
            }

            # Check for duplicates
            if factor_summary[factor_name]["duplicate_dates"] > 0:
                validation_result["warnings"].append(
                    f"Factor '{factor_name}' has {factor_summary[factor_name]['duplicate_dates']} duplicate dates"
                )

        validation_result["summary"]["factors"] = factor_summary
        validation_result["summary"]["main_data_range"] = {
            "start": main_min.isoformat(),
            "end": main_max.isoformat()
        }
        validation_result["summary"]["upload_range"] = {
            "start": upload_min.isoformat(),
            "end": upload_max.isoformat()
        }

        return validation_result