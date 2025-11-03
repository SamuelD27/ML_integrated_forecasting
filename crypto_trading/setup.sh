#!/bin/bash
set -e

echo "================================================"
echo "  Crypto Trading MVP - Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
if [ "$python_version" != "3.11" ]; then
    echo "⚠️  Warning: Python 3.11 required, found $python_version"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi
echo "✅ Python version OK"

# Check PostgreSQL
echo ""
echo "Checking PostgreSQL..."
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL not found. Please install PostgreSQL 14+."
    exit 1
fi
echo "✅ PostgreSQL found"

# Check TimescaleDB
echo ""
echo "Checking TimescaleDB..."
if ! psql -U postgres -c "SELECT extname FROM pg_extension WHERE extname='timescaledb';" &> /dev/null; then
    echo "⚠️  TimescaleDB extension not found. Install it? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
    fi
fi
echo "✅ TimescaleDB ready"

# Create database
echo ""
echo "Creating database..."
if psql -U postgres -lqt | cut -d \| -f 1 | grep -qw crypto_trading; then
    echo "⚠️  Database 'crypto_trading' already exists. Drop and recreate? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        psql -U postgres -c "DROP DATABASE crypto_trading;"
        psql -U postgres -c "CREATE DATABASE crypto_trading;"
    fi
else
    psql -U postgres -c "CREATE DATABASE crypto_trading;"
fi
echo "✅ Database created"

# Run schema
echo ""
echo "Creating schema..."
psql -U postgres -d crypto_trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
psql -U postgres -d crypto_trading -f schema.sql
echo "✅ Schema created"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Setup .env
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your credentials"
else
    echo "✅ .env already exists"
fi

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env with your Kraken API credentials"
echo "2. Run: python -m crypto_trading.main"
echo ""
