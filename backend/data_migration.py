#!/usr/bin/env python3
"""
Data migration utilities for database schema changes
"""

from sqlalchemy.orm import Session
from sqlalchemy import text
from database import SessionLocal, ForecastData, ProductCustomerLocationCombination
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_existing_data():
    """
    Migrate existing forecast data to use the new combination table structure
    """
    db = SessionLocal()
    try:
        logger.info("Starting data migration...")
        
        # Check if the old columns still exist
        try:
            result = db.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'forecast_data' AND column_name IN ('product', 'customer', 'location')"))
            old_columns = [row[0] for row in result.fetchall()]
            
            if not old_columns:
                logger.info("Old columns not found. Migration may have already been completed.")
                return True
                
        except Exception as e:
            logger.error(f"Error checking for old columns: {e}")
            return False
        
        # Get all existing forecast data with old structure
        try:
            result = db.execute(text("""
                SELECT DISTINCT 
                    product, customer, location, product_group, product_hierarchy,
                    location_region, customer_group, customer_region, 
                    ship_to_party, sold_to_party
                FROM forecast_data 
                WHERE product IS NOT NULL OR customer IS NOT NULL OR location IS NOT NULL
            """))
            
            unique_combinations = result.fetchall()
            logger.info(f"Found {len(unique_combinations)} unique combinations to migrate")
            
        except Exception as e:
            logger.error(f"Error fetching existing data: {e}")
            return False
        
        # Create combination records
        combination_map = {}
        for combo in unique_combinations:
            try:
                # Check if combination already exists
                existing_combo = db.query(ProductCustomerLocationCombination).filter(
                    ProductCustomerLocationCombination.product == combo[0],
                    ProductCustomerLocationCombination.customer == combo[1],
                    ProductCustomerLocationCombination.location == combo[2]
                ).first()
                
                if existing_combo:
                    combination_map[(combo[0], combo[1], combo[2])] = existing_combo.id
                else:
                    # Create new combination
                    new_combo = ProductCustomerLocationCombination(
                        product=combo[0],
                        customer=combo[1],
                        location=combo[2],
                        product_group=combo[3],
                        product_hierarchy=combo[4],
                        location_region=combo[5],
                        customer_group=combo[6],
                        customer_region=combo[7],
                        ship_to_party=combo[8],
                        sold_to_party=combo[9]
                    )
                    db.add(new_combo)
                    db.flush()  # Get the ID without committing
                    combination_map[(combo[0], combo[1], combo[2])] = new_combo.id
                    
            except Exception as e:
                logger.error(f"Error creating combination for {combo}: {e}")
                db.rollback()
                return False
        
        db.commit()
        logger.info(f"Created {len(combination_map)} combination records")
        
        # Update forecast_data records to use combination_id
        try:
            # Add combination_id column if it doesn't exist
            db.execute(text("ALTER TABLE forecast_data ADD COLUMN IF NOT EXISTS combination_id INTEGER"))
            db.commit()
            
            # Update each record with the appropriate combination_id
            for (product, customer, location), combo_id in combination_map.items():
                db.execute(text("""
                    UPDATE forecast_data 
                    SET combination_id = :combo_id 
                    WHERE (product = :product OR (product IS NULL AND :product IS NULL))
                      AND (customer = :customer OR (customer IS NULL AND :customer IS NULL))
                      AND (location = :location OR (location IS NULL AND :location IS NULL))
                      AND combination_id IS NULL
                """), {
                    'combo_id': combo_id,
                    'product': product,
                    'customer': customer,
                    'location': location
                })
            
            db.commit()
            logger.info("Updated forecast_data records with combination_id")
            
        except Exception as e:
            logger.error(f"Error updating forecast_data: {e}")
            db.rollback()
            return False
        
        # Add foreign key constraint
        try:
            db.execute(text("ALTER TABLE forecast_data ADD CONSTRAINT fk_forecast_data_combination FOREIGN KEY (combination_id) REFERENCES product_customer_location_combinations(id)"))
            db.commit()
            logger.info("Added foreign key constraint")
        except Exception as e:
            logger.warning(f"Could not add foreign key constraint (may already exist): {e}")
        
        # Drop old columns (commented out for safety - uncomment after verifying migration)
        # try:
        #     db.execute(text("ALTER TABLE forecast_data DROP COLUMN IF EXISTS product"))
        #     db.execute(text("ALTER TABLE forecast_data DROP COLUMN IF EXISTS customer"))
        #     db.execute(text("ALTER TABLE forecast_data DROP COLUMN IF EXISTS location"))
        #     db.execute(text("ALTER TABLE forecast_data DROP COLUMN IF EXISTS product_group"))
        #     db.execute(text("ALTER TABLE forecast_data DROP COLUMN IF EXISTS product_hierarchy"))
        #     db.execute(text("ALTER TABLE forecast_data DROP COLUMN IF EXISTS location_region"))
        #     db.execute(text("ALTER TABLE forecast_data DROP COLUMN IF EXISTS customer_group"))
        #     db.execute(text("ALTER TABLE forecast_data DROP COLUMN IF EXISTS customer_region"))
        #     db.execute(text("ALTER TABLE forecast_data DROP COLUMN IF EXISTS ship_to_party"))
        #     db.execute(text("ALTER TABLE forecast_data DROP COLUMN IF EXISTS sold_to_party"))
        #     db.commit()
        #     logger.info("Dropped old columns")
        # except Exception as e:
        #     logger.warning(f"Could not drop old columns: {e}")
        
        logger.info("Data migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def verify_migration():
    """
    Verify that the migration was successful
    """
    db = SessionLocal()
    try:
        # Check combination table
        combo_count = db.query(ProductCustomerLocationCombination).count()
        logger.info(f"Combination table has {combo_count} records")
        
        # Check forecast data
        forecast_count = db.query(ForecastData).count()
        forecast_with_combo = db.query(ForecastData).filter(ForecastData.combination_id.isnot(None)).count()
        
        logger.info(f"Forecast data table has {forecast_count} records")
        logger.info(f"Forecast data with combination_id: {forecast_with_combo}")
        
        if forecast_count == forecast_with_combo:
            logger.info("✅ Migration verification successful!")
            return True
        else:
            logger.error("❌ Migration verification failed!")
            return False
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("Starting database migration...")
    if migrate_existing_data():
        print("Migration completed. Verifying...")
        if verify_migration():
            print("✅ Migration and verification successful!")
        else:
            print("❌ Verification failed!")
    else:
        print("❌ Migration failed!")