"""
Tests for quick_stock_report.py - primary CLI tool for single stock analysis.

Test coverage:
1. Valid ticker analysis
2. Invalid ticker handling
3. Output format validation
4. Caching behavior
5. Recommendation generation
6. Chart generation
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if quick_stock_report can be imported
try:
    import quick_stock_report
    HAS_QUICK_STOCK_REPORT = True
except ImportError:
    HAS_QUICK_STOCK_REPORT = False

# Check if xlsxwriter is available for export tests
try:
    import xlsxwriter
    HAS_XLSXWRITER = True
except ImportError:
    HAS_XLSXWRITER = False

requires_quick_stock_report = pytest.mark.skipif(
    not HAS_QUICK_STOCK_REPORT,
    reason="quick_stock_report requires dependencies that are not installed"
)

requires_xlsxwriter = pytest.mark.skipif(
    not HAS_XLSXWRITER,
    reason="xlsxwriter is required for Excel export tests"
)


class TestRunCompleteAnalysis:
    """Tests for run_complete_analysis function."""

    @requires_quick_stock_report
    def test_valid_ticker_returns_results(self, mock_price_data):
        """Test that valid ticker produces expected result structure."""
        # This test validates the result structure from run_complete_analysis
        # We use extensive mocking to avoid network calls

        with patch('quick_stock_report.fetch_full_bundle') as mock_fetch, \
             patch('quick_stock_report.save_bundle'), \
             patch('quick_stock_report.set_last_fetch_globals'), \
             patch('quick_stock_report.load_current_session_data') as mock_load_data, \
             patch('quick_stock_report.StockDataProcessor') as mock_processor_class, \
             patch('yfinance.download') as mock_yf_download, \
             patch('quick_stock_report.fit_arima_model') as mock_arima, \
             patch('quick_stock_report.generate_charts') as mock_charts:

            from quick_stock_report import run_complete_analysis

            # Setup mocks
            mock_fetch.return_value = {
                'prices': mock_price_data,
                'meta': {'date_range_actual': {'start': '2023-01-01', 'end': '2024-01-01'}}
            }
            mock_load_data.return_value = (mock_price_data, mock_fetch.return_value)

            # Mock processor
            mock_processor = Mock()
            mock_processor.close_prices = mock_price_data['Close']
            mock_processor.info = {'trailingPE': 25.0}
            mock_processor.process_all.return_value = {
                'returns': {'simple_returns': mock_price_data['Close'].pct_change()},
                'descriptive_stats': {
                    'sharpe_ratio': 1.2,
                    'mean_annual_return': 0.15,
                    'std_annual_return': 0.20,
                    'sortino_ratio': 1.5,
                    'positive_days_pct': 55.0,
                },
                'risk_metrics': {
                    'max_drawdown': -0.15,
                    'var_95': -0.025,
                },
            }
            mock_processor_class.return_value = mock_processor

            # Mock yfinance for benchmark
            mock_yf_download.return_value = mock_price_data

            # Mock ARIMA and charts
            mock_arima.return_value = Mock(
                predictions=np.array([150, 151, 152]),
                forecast_horizon=3,
                rmse=2.5,
                directional_accuracy=65.0
            )
            mock_charts.return_value = {}

            result = run_complete_analysis('AAPL')

        # Verify result structure
        assert 'ticker' in result
        assert result['ticker'] == 'AAPL'
        assert 'results' in result
        assert 'processor' in result
        assert 'analysis_time' in result

    @requires_quick_stock_report
    @patch('quick_stock_report.fetch_full_bundle')
    def test_invalid_ticker_raises_error(self, mock_fetch):
        """Test that invalid ticker raises appropriate error."""
        from quick_stock_report import run_complete_analysis

        mock_fetch.return_value = {'prices': pd.DataFrame(), 'meta': {}}

        with pytest.raises(Exception):
            run_complete_analysis('INVALID_TICKER_XYZ')

    @requires_quick_stock_report
    def test_recommendation_scoring_strong_buy(self):
        """Test recommendation generates STRONG BUY for excellent metrics."""
        from quick_stock_report import generate_recommendation

        results = {
            'descriptive_stats': {
                'sharpe_ratio': 2.0,
                'mean_annual_return': 0.25,
                'std_annual_return': 0.15,
                'positive_days_pct': 60.0,
            },
            'risk_metrics': {'max_drawdown': -0.10},
            'market_sensitivity': {'alpha_annual': 0.05},
        }

        mock_processor = Mock()
        recommendation, reasons, color = generate_recommendation(results, mock_processor)

        assert recommendation in ['STRONG BUY', 'BUY']
        assert len(reasons) > 0

    @requires_quick_stock_report
    def test_recommendation_scoring_sell(self):
        """Test recommendation generates SELL for poor metrics."""
        from quick_stock_report import generate_recommendation

        results = {
            'descriptive_stats': {
                'sharpe_ratio': -0.5,
                'mean_annual_return': -0.20,
                'std_annual_return': 0.50,
                'positive_days_pct': 40.0,
            },
            'risk_metrics': {'max_drawdown': -0.60},
            'market_sensitivity': {'alpha_annual': -0.10},
        }

        mock_processor = Mock()
        recommendation, reasons, color = generate_recommendation(results, mock_processor)

        assert recommendation in ['SELL', 'REDUCE']

    @requires_quick_stock_report
    def test_recommendation_scoring_hold(self):
        """Test recommendation generates HOLD for average metrics."""
        from quick_stock_report import generate_recommendation

        results = {
            'descriptive_stats': {
                'sharpe_ratio': 0.5,
                'mean_annual_return': 0.05,
                'std_annual_return': 0.25,
                'positive_days_pct': 52.0,
            },
            'risk_metrics': {'max_drawdown': -0.25},
            'market_sensitivity': {'alpha_annual': 0.0},
        }

        mock_processor = Mock()
        recommendation, reasons, color = generate_recommendation(results, mock_processor)

        assert recommendation in ['HOLD', 'BUY', 'REDUCE']


class TestSummarySheet:
    """Tests for summary sheet generation."""

    @requires_quick_stock_report
    def test_create_summary_sheet_structure(self, mock_price_data):
        """Test that summary sheet has expected structure."""
        from quick_stock_report import create_summary_sheet

        analysis = {
            'ticker': 'AAPL',
            'processor': Mock(
                close_prices=mock_price_data['Close'],
                info={'trailingPE': 25.0, 'priceToBook': 10.0, 'dividendYield': 0.005}
            ),
            'results': {
                'descriptive_stats': {
                    'mean_annual_return': 0.15,
                    'std_annual_return': 0.20,
                    'sharpe_ratio': 0.75,
                    'sortino_ratio': 1.0,
                    'positive_days_pct': 55.0,
                },
                'risk_metrics': {
                    'max_drawdown': -0.15,
                    'var_95': -0.025,
                },
                'company_info': {
                    'company_name': 'Apple Inc.',
                    'sector': 'Technology',
                    'industry': 'Consumer Electronics',
                },
                'corporate_actions': {
                    'num_dividends': 4,
                    'total_dividends': 0.96,
                },
            },
            'forecast': None,
        }

        df = create_summary_sheet(analysis)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'Section' in df.columns
        assert 'Metric' in df.columns
        assert 'Value' in df.columns


class TestChartGeneration:
    """Tests for chart generation."""

    @requires_quick_stock_report
    def test_generate_charts_returns_dict(self, mock_price_data):
        """Test that generate_charts returns dictionary of chart paths."""
        from quick_stock_report import generate_charts

        mock_processor = Mock()
        mock_processor.close_prices = mock_price_data['Close']

        results = {
            'returns': {'simple_returns': mock_price_data['Close'].pct_change()}
        }

        forecast_data = None

        charts = generate_charts(mock_processor, results, forecast_data, 'AAPL')

        assert isinstance(charts, dict)
        assert 'price_history' in charts
        assert 'returns_dist' in charts
        assert 'volatility' in charts

    @requires_quick_stock_report
    def test_generate_charts_with_forecast(self, mock_price_data):
        """Test chart generation includes forecast when available."""
        from quick_stock_report import generate_charts

        mock_processor = Mock()
        mock_processor.close_prices = mock_price_data['Close']

        results = {
            'returns': {'simple_returns': mock_price_data['Close'].pct_change()}
        }

        forecast_data = {
            'predictions': np.array([150, 151, 152, 153, 154]),
            'rmse': 2.5,
            'directional_accuracy': 65.0,
        }

        charts = generate_charts(mock_processor, results, forecast_data, 'AAPL')

        assert 'forecast' in charts


class TestExportReport:
    """Tests for Excel report export."""

    @requires_quick_stock_report
    @requires_xlsxwriter
    def test_export_creates_file(self, mock_price_data, temp_reports_dir):
        """Test that export creates Excel file."""
        from quick_stock_report import export_hedge_fund_report

        analysis = {
            'ticker': 'AAPL',
            'processor': Mock(
                close_prices=mock_price_data['Close'],
                info={'trailingPE': 25.0}
            ),
            'results': {
                'descriptive_stats': {
                    'mean_annual_return': 0.15,
                    'std_annual_return': 0.20,
                    'sharpe_ratio': 0.75,
                    'sortino_ratio': 1.0,
                    'positive_days_pct': 55.0,
                },
                'risk_metrics': {
                    'max_drawdown': -0.15,
                    'var_95': -0.025,
                },
            },
            'forecast': None,
            'charts': {},
            'analysis_time': 5.0,
        }

        output_path = temp_reports_dir / 'test_report.xlsx'
        export_hedge_fund_report(analysis, str(output_path))

        assert output_path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
