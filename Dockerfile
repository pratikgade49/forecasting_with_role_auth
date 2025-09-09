# Stage 1: Frontend Dev Server
FROM node:20-alpine AS frontend-dev

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy all frontend source code
COPY . .

# Expose frontend dev server port
EXPOSE 5173

# Stage 2: Python Backend
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./

# Make the start script executable
COPY backend/start.sh ./
RUN chmod +x start.sh

# Expose backend port
EXPOSE 8000

# Start backend
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "&&", "./start.sh"]
