#!/bin/bash
# Test Runner Script for Stock Analysis System
# ============================================

set -e  # Exit on error

echo "========================================"
echo "Stock Analysis System - Test Suite"
echo "========================================"
echo ""

# Check if pytest is installed
if ! python3 -c "import pytest" 2>/dev/null; then
    echo "❌ pytest not installed. Installing..."
    pip install pytest pytest-cov hypothesis
fi

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
TEST_TYPE="${1:-all}"
COVERAGE="${2:-yes}"

# Function to run tests
run_tests() {
    local test_path=$1
    local description=$2

    echo ""
    echo "========================================"
    echo "Running: $description"
    echo "========================================"
    echo ""

    if [ "$COVERAGE" = "yes" ]; then
        pytest "$test_path" -v --tb=short --cov=. --cov-report=term-missing
    else
        pytest "$test_path" -v --tb=short
    fi
}

# Main test execution
case $TEST_TYPE in
    "all")
        echo "Running ALL tests..."
        run_tests "tests/" "All Tests"
        ;;

    "factor")
        echo "Running factor model tests..."
        run_tests "tests/test_factor_models.py" "Factor Models"
        ;;

    "longshort")
        echo "Running long/short strategy tests..."
        run_tests "tests/test_long_short_strategy.py" "Long/Short Strategy"
        ;;

    "hrp")
        echo "Running HRP optimizer tests..."
        run_tests "tests/test_hrp_optimizer.py" "HRP Optimizer"
        ;;

    "ensemble")
        echo "Running ensemble tests..."
        run_tests "tests/test_ensemble.py" "Model Ensemble"
        ;;

    "unit")
        echo "Running unit tests only..."
        pytest tests/ -v -m "unit" --tb=short
        ;;

    "integration")
        echo "Running integration tests only..."
        pytest tests/ -v -m "integration" --tb=short
        ;;

    "property")
        echo "Running property-based tests only..."
        pytest tests/ -v -m "property" --tb=short
        ;;

    "fast")
        echo "Running fast tests (excluding slow tests)..."
        pytest tests/ -v -m "not slow" --tb=short
        ;;

    "coverage")
        echo "Running tests with detailed coverage report..."
        pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
        echo ""
        echo "✅ Coverage report generated in htmlcov/index.html"
        ;;

    *)
        echo "❌ Unknown test type: $TEST_TYPE"
        echo ""
        echo "Usage: ./run_tests.sh [test_type] [coverage]"
        echo ""
        echo "Test types:"
        echo "  all          - Run all tests (default)"
        echo "  factor       - Factor model tests"
        echo "  longshort    - Long/short strategy tests"
        echo "  hrp          - HRP optimizer tests"
        echo "  ensemble     - Model ensemble tests"
        echo "  unit         - Unit tests only"
        echo "  integration  - Integration tests only"
        echo "  property     - Property-based tests only"
        echo "  fast         - Fast tests (exclude slow)"
        echo "  coverage     - Generate detailed coverage report"
        echo ""
        echo "Coverage: yes (default) or no"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh all yes         # All tests with coverage"
        echo "  ./run_tests.sh factor no       # Factor tests without coverage"
        echo "  ./run_tests.sh fast            # Fast tests with coverage"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================"
    echo "✅ Tests PASSED"
    echo -e "========================================${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}========================================"
    echo "❌ Tests FAILED"
    echo -e "========================================${NC}"
    echo ""
    exit 1
fi

# Display coverage summary
if [ "$COVERAGE" = "yes" ] && [ "$TEST_TYPE" != "coverage" ]; then
    echo ""
    echo "📊 Coverage Summary:"
    echo "  - Detailed report: htmlcov/index.html"
    echo "  - XML report: coverage.xml"
    echo ""
fi

echo "✅ Test run complete!"
