import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from portfolio.chart_generator import ChartGenerator
        print("  ✓ ChartGenerator imported")
    except Exception as e:
        print(f"  ✗ ChartGenerator failed: {e}")
        return False

    try:
        from portfolio.monte_carlo import MonteCarloSimulator
        print("  ✓ MonteCarloSimulator imported")
    except Exception as e:
        print(f"  ✗ MonteCarloSimulator failed: {e}")
        return False

    try:
        from portfolio.derivatives_engine import DerivativesEngine
        print("  ✓ DerivativesEngine imported")
    except Exception as e:
        print(f"  ✗ DerivativesEngine failed: {e}")
        return False

    try:
        import portfolio_creation_enhanced
        print("  ✓ portfolio_creation_enhanced imported")
    except Exception as e:
        print(f"  ✗ portfolio_creation_enhanced failed: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality with sample data."""
    print("\nTesting basic functionality...")

    try:
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta

        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)

        prices_df = pd.DataFrame({
            'AAPL': 150 + np.random.randn(len(dates)).cumsum(),
            'MSFT': 300 + np.random.randn(len(dates)).cumsum(),
            'GOOG': 100 + np.random.randn(len(dates)).cumsum(),
        }, index=dates)

        returns = prices_df.pct_change().dropna()

        holdings_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOG'],
            'weight': [0.4, 0.35, 0.25],
            'dollars': [40000, 35000, 25000],
            'shares': [266, 116, 250],
            'price': [150.0, 300.0, 100.0],
        })

        # Test ChartGenerator
        from portfolio.chart_generator import ChartGenerator

        chart_gen = ChartGenerator(prices_df, returns, holdings_df, 'AAPL')
        print("  ✓ ChartGenerator initialized")

        # Test chart generation (don't display)
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        fig, insights = chart_gen.generate_price_performance_chart()
        print(f"  ✓ Price performance chart generated ({len(insights)} insights)")

        fig, insights = chart_gen.generate_correlation_heatmap()
        print(f"  ✓ Correlation heatmap generated ({len(insights)} insights)")

        # Test MonteCarloSimulator
        from portfolio.monte_carlo import MonteCarloSimulator

        weights = {'AAPL': 0.4, 'MSFT': 0.35, 'GOOG': 0.25}
        mc_sim = MonteCarloSimulator(returns, weights, 100000)
        print("  ✓ MonteCarloSimulator initialized")

        # Run small simulation
        results = mc_sim.run_simulation(num_simulations=1000, time_horizon=252)
        print(f"  ✓ Monte Carlo simulation completed (1000 paths)")
        print(f"    - Expected return: {results['expected_return']*100:.2f}%")
        print(f"    - Probability of profit: {results['probability_profit']*100:.1f}%")

        # Test DerivativesEngine
        from portfolio.derivatives_engine import DerivativesEngine

        deriv_engine = DerivativesEngine('AAPL', 150.0, 100000, 0.05)
        print("  ✓ DerivativesEngine initialized")

        # Test variance swap
        var_swap = deriv_engine.variance_swap_analysis(0.25, 0.30)
        print(f"  ✓ Variance swap analysis completed")

        # Test total return swap
        trs = deriv_engine.total_return_swap()
        print(f"  ✓ Total return swap analysis completed")

        return True

    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scenario_analysis():
    """Test scenario analysis."""
    print("\nTesting scenario analysis...")

    try:
        import numpy as np
        import pandas as pd
        from portfolio.monte_carlo import MonteCarloSimulator

        # Create sample returns
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        returns = pd.DataFrame({
            'AAPL': np.random.randn(252) * 0.02,
            'MSFT': np.random.randn(252) * 0.018,
        }, index=dates)

        weights = {'AAPL': 0.6, 'MSFT': 0.4}
        mc_sim = MonteCarloSimulator(returns, weights, 100000)

        scenarios = {
            'Bull Market': 0.15,
            'Bear Market': -0.20,
            'Crash': -0.40,
        }

        scenario_df = mc_sim.scenario_analysis(scenarios)
        print(f"  ✓ Scenario analysis completed")
        print(f"    - Scenarios tested: {len(scenario_df)}")
        print(f"\n{scenario_df.to_string()}\n")

        return True

    except Exception as e:
        print(f"  ✗ Scenario analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("ENHANCED PORTFOLIO SYSTEM - TEST SUITE")
    print("="*80)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False

    # Test scenario analysis
    if not test_scenario_analysis():
        all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print("\nThe enhanced portfolio system is ready to use!")
        print("\nTry running:")
        print("  python portfolio_creation_enhanced.py AAPL --capital 100000 --rrr 0.6")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease review the errors above and ensure all dependencies are installed.")
        print("\nInstall dependencies with:")
        print("  pip install numpy pandas yfinance matplotlib seaborn scipy xlsxwriter")

    print()


if __name__ == "__main__":
    main()
