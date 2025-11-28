# Makefile for Stock Analysis Project
# ====================================
# Common commands for development, testing, and deployment

.PHONY: help install install-dev install-training test test-unit test-integration \
        lint format type-check clean dashboard train fetch-data setup-env \
        docker-build docker-run runpod-setup crypto-trading

# Default target
.DEFAULT_GOAL := help

# Project configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
PROJECT_ROOT := $(shell pwd)
PYTHONPATH := $(PROJECT_ROOT)

# Colors for terminal output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help: ## Show this help message
	@echo "Stock Analysis Project - Available Commands"
	@echo "============================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------

install: ## Install basic dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements_basic.txt

install-dev: install ## Install development dependencies
	$(PIP) install pytest pytest-cov pytest-timeout hypothesis
	$(PIP) install black flake8 isort mypy bandit safety
	$(PIP) install pre-commit

install-training: ## Install full training dependencies (requires Python 3.11)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements_training.txt

install-runpod: ## Install RunPod-optimized dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements_runpod.txt

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

test: ## Run all tests
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v --tb=short

test-unit: ## Run unit tests only
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v \
		-m "not integration and not requires_api and not requires_gpu and not slow" \
		--tb=short --timeout=60

test-integration: ## Run integration tests only
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/test_integration.py -v \
		--tb=short --timeout=120

test-ml: ## Run ML model tests
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/test_ml_models.py tests/test_ensemble.py -v \
		--tb=short --timeout=300

test-portfolio: ## Run portfolio tests
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/test_enhanced_portfolio.py tests/test_hrp_optimizer.py -v \
		--tb=short

test-cov: ## Run tests with coverage report
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v \
		--cov=. --cov-report=html --cov-report=term-missing \
		--ignore=tests/test_dashboard_live.py

test-fast: ## Run fast tests only (skip slow tests)
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v \
		-m "not slow and not requires_api and not requires_gpu" \
		--tb=short --timeout=30 -x

#------------------------------------------------------------------------------
# Code Quality
#------------------------------------------------------------------------------

lint: ## Run linters (flake8)
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=15 --max-line-length=120 --statistics

format: ## Format code with Black and isort
	black .
	isort .

format-check: ## Check code formatting without making changes
	black --check --diff .
	isort --check-only --diff .

type-check: ## Run type checking with mypy
	mypy --ignore-missing-imports --no-error-summary \
		data_providers/ ml_models/ portfolio/ utils/ || true

security: ## Run security checks
	bandit -r . -x ./tests,./venv,./.venv -ll -ii || true
	@echo "Checking for hardcoded secrets..."
	@grep -rn "API_KEY\s*=\s*['\"][a-zA-Z0-9]" --include="*.py" --exclude-dir=.git --exclude-dir=venv 2>/dev/null | \
		grep -v "your_\|example\|placeholder\|_here" || echo "No obvious secrets found"

#------------------------------------------------------------------------------
# Dashboard
#------------------------------------------------------------------------------

dashboard: ## Launch Streamlit dashboard
	@echo "$(GREEN)Starting Stock Analysis Dashboard...$(NC)"
	PYTHONPATH=$(PYTHONPATH) streamlit run dashboard/Home.py --server.port 8501

dashboard-dev: ## Launch dashboard in development mode (auto-reload)
	PYTHONPATH=$(PYTHONPATH) streamlit run dashboard/Home.py \
		--server.port 8501 \
		--server.runOnSave true

#------------------------------------------------------------------------------
# Training
#------------------------------------------------------------------------------

train: ## Run model training
	@echo "$(GREEN)Starting model training...$(NC)"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) training/train_hybrid.py \
		--config configs/default_config.yaml

train-regime: ## Train regime classifier
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) train_regime_classifier.py

train-validate: ## Validate training setup
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) validate_training_setup.py

#------------------------------------------------------------------------------
# Data
#------------------------------------------------------------------------------

fetch-data: ## Fetch market data (use TICKERS env var or default)
	@echo "$(GREEN)Fetching market data...$(NC)"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) data_fetching.py \
		--tickers "$${TICKERS:-AAPL,MSFT,GOOGL,AMZN}" \
		--start "$${START_DATE:-2020-01-01}" \
		--end "$${END_DATE:-$(shell date +%Y-%m-%d)}"

prepare-data: ## Prepare training data
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) prepare_training_data.py

#------------------------------------------------------------------------------
# Portfolio
#------------------------------------------------------------------------------

portfolio: ## Run portfolio optimization
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run_portfolio.py

stock-report: ## Generate quick stock report (use TICKER env var)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) quick_stock_report.py --ticker "$${TICKER:-AAPL}"

#------------------------------------------------------------------------------
# Crypto Trading
#------------------------------------------------------------------------------

crypto-trading: ## Start crypto paper trading system
	@echo "$(YELLOW)Starting crypto paper trading system...$(NC)"
	@echo "$(YELLOW)Ensure .env is configured with database credentials$(NC)"
	cd crypto_trading && PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m crypto_trading.main

crypto-setup: ## Setup crypto trading database
	@echo "Setting up crypto trading database..."
	cd crypto_trading && ./setup.sh

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------

setup-env: ## Create .env from example
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)Created .env from .env.example$(NC)"; \
		echo "$(YELLOW)Please edit .env with your API keys$(NC)"; \
	else \
		echo "$(YELLOW).env already exists, not overwriting$(NC)"; \
	fi

setup-pythonpath: ## Setup PYTHONPATH for current session
	@echo "Run this command to set PYTHONPATH:"
	@echo "  export PYTHONPATH=$(PROJECT_ROOT)"

venv: ## Create virtual environment
	$(PYTHON) -m venv venv
	@echo "$(GREEN)Virtual environment created$(NC)"
	@echo "Activate with: source venv/bin/activate"

#------------------------------------------------------------------------------
# Cleanup
#------------------------------------------------------------------------------

clean: ## Clean build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf build/ dist/
	@echo "$(GREEN)Cleaned build artifacts$(NC)"

clean-data: ## Clean cached data files
	rm -rf data/cache/*
	rm -rf cache/*
	@echo "$(GREEN)Cleaned cached data$(NC)"

clean-all: clean clean-data ## Clean everything
	rm -rf checkpoints/*
	rm -rf logs/*
	rm -rf reports/*
	@echo "$(GREEN)Cleaned all generated files$(NC)"

#------------------------------------------------------------------------------
# CI/CD Helpers
#------------------------------------------------------------------------------

ci-test: ## Run CI test suite (mimics GitHub Actions)
	$(MAKE) lint
	$(MAKE) test-fast
	@echo "$(GREEN)CI tests passed$(NC)"

pre-commit: ## Run pre-commit checks
	pre-commit run --all-files

#------------------------------------------------------------------------------
# RunPod Deployment
#------------------------------------------------------------------------------

runpod-setup: ## Prepare files for RunPod deployment
	@echo "$(GREEN)Preparing RunPod deployment package...$(NC)"
	mkdir -p runpod_deploy
	cp requirements_runpod.txt runpod_deploy/
	cp deployment/runpod_setup.sh runpod_deploy/
	cp deployment/train_on_runpod.py runpod_deploy/
	cp deployment/serve_model.py runpod_deploy/
	cp -r configs/ runpod_deploy/configs/
	cp -r ml_models/ runpod_deploy/ml_models/
	cp -r training/ runpod_deploy/training/
	cp -r utils/ runpod_deploy/utils/
	@echo "$(GREEN)RunPod package ready in runpod_deploy/$(NC)"

runpod-launch: ## Launch RunPod instance (requires RUNPOD_API_KEY)
	@if [ -z "$$RUNPOD_API_KEY" ]; then \
		echo "$(RED)Error: RUNPOD_API_KEY not set$(NC)"; \
		exit 1; \
	fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) launch_runpod.py
