#!/bin/bash

# Wait for PostgreSQL to be ready
# echo "Waiting for PostgreSQL to be ready..."
# while ! pg_isready -h postgres -p 5432 -q; do
#     sleep 1
# done

# # Setup the database
# echo "Setting up the database..."
# python setup_database.py

# Start the application
echo "Starting the application..."
uvicorn main:app --host 0.0.0.0 --port 8000
