#!/usr/bin/env python3
"""
Database table setup script for PostgreSQL
Assumes the database already exists and only creates tables
"""

import os
from database import init_database
from data_migration import migrate_existing_data, verify_migration

def setup_tables():
    """Setup database tables (assumes database already exists)"""
    try:
        print("🗄️  Setting up database tables...")
        print("=" * 60)
        
        # Check environment variables
        print("Database Configuration:")
        print(f"Host: {os.getenv('DB_HOST', 'localhost')}") #172.17.0.1
        print(f"Port: {os.getenv('DB_PORT', '5433')}") #'5432'
        print(f"User: {os.getenv('DB_USER', 'postgres')}")
        print(f"Database: {os.getenv('DB_NAME', 'forecasting_db')}")
        print()
        
        print("📋 Note: This script assumes the database already exists.")
        print("📋 It will only create the required tables if they don't exist.")
        print()
        
        # Initialize tables
        if init_database():
            print("\n🎉 Database tables setup completed!")
            print("\nTables created/verified:")
            print("- forecast_data (stores uploaded forecast data)")
            print("- external_factor_data (stores external factor data)")
            print("- forecast_configurations (stores saved configurations)")
            print("\nYou can now start the application with: python main.py")
            return True
        else:
            print("\n❌ Database setup failed. Please check the error messages above.")
            print("\nTroubleshooting:")
            print("1. Ensure PostgreSQL server is running")
            print("2. Verify the database 'forecasting_db' exists")
            print("3. Check database credentials in .env file")
            print("4. Ensure user has CREATE TABLE privileges")
            return False
            
            # Run data migration
            print("\n🔄 Running data migration...")
            if migrate_existing_data():
                print("✅ Data migration completed successfully!")
                if verify_migration():
                    print("✅ Migration verification passed!")
                else:
                    print("⚠️  Migration verification failed!")
            else:
                print("❌ Data migration failed!")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    setup_tables()