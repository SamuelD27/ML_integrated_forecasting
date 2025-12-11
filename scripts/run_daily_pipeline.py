#!/usr/bin/env python
"""
Daily Trading Pipeline
======================
Main orchestration script for the ML-enhanced trading pipeline.

Runs the complete pipeline:
1. Universe Building - Filter and score stocks
2. Regime Classification - Determine market regime
3. Forecasting - Generate ensemble forecasts
4. Meta-Labeling - Filter signals by meta-model
5. Instrument Selection - Choose equity vs options
6. Allocation - Optimize portfolio weights
7. Order Generation - Create executable orders

Usage:
    python scripts/run_daily_pipeline.py --config config/trading.yaml
    python scripts/run_daily_pipeline.py --dry-run
    python scripts/run_daily_pipeline.py --output orders.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import pipeline components
try:
    from pipeline.core_types import StockSnapshot, TradeSignal, Forecast
    from pipeline.universe_builder import UniverseBuilder
    from pipeline.regime_utils import classify_regime, get_regime_parameters, RegimeDetector
    from pipeline.trade_filter import filter_signals, rank_signals, select_top_signals
    from pipeline.instrument_selector import (
        select_instruments_batch,
        apply_regime_adjustments,
        summarize_selections,
        InstrumentSelection,
    )
    from pipeline.allocation_utils import (
        allocate_from_signals,
        rebalance_portfolio,
        validate_allocation,
    )
    HAS_PIPELINE = True
except ImportError as e:
    logger.error(f"Pipeline imports failed: {e}")
    HAS_PIPELINE = False

# Import data providers
try:
    from data.macro_loader import load_macro_features
    from data_fetching import fetch_data
    HAS_DATA = True
except ImportError:
    HAS_DATA = False
    logger.warning("Data fetching modules not available")

# Import ML models
try:
    from ml_models.ensemble_forecaster import EnsembleForecaster, get_forecaster
    from ml_models.meta_model import get_meta_model, MetaModel
    from ml_models.meta_features import build_meta_features
    HAS_ML = True
except ImportError:
    HAS_ML = False
    logger.warning("ML models not available")

# Import config loader
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class DailyPipeline:
    """
    Main pipeline orchestrator.

    Coordinates all pipeline stages and manages state.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.as_of_date = pd.Timestamp.now()
        self.regime = None
        self.regime_info = None
        self.signals = []
        self.allocations = None
        self.orders = []

        # Initialize components
        self._init_components()

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'universe': {
                'min_market_cap': 10e9,
                'min_avg_volume': 1e6,
                'sectors': None,
                'exclude_tickers': [],
            },
            'regime': {
                'smoothing_window': 3,
            },
            'forecaster': {
                'horizon_days': 10,
                'models': [
                    {'name': 'nbeats', 'weight': 0.35, 'enabled': True},
                    {'name': 'lstm', 'weight': 0.35, 'enabled': True},
                    {'name': 'tft', 'weight': 0.30, 'enabled': True},
                ],
            },
            'meta_labeling': {
                'meta_prob_threshold': 0.65,
            },
            'filters': {
                'blocked_regimes': [3],
                'long_only': True,
                'min_expected_return': 0.01,
                'max_volatility': 0.50,
            },
            'allocation': {
                'portfolio_value': 100000,
                'max_positions': 15,
                'max_weight': 0.15,
                'min_weight': 0.02,
            },
            'instruments': {
                'use_options': False,
                'max_position_pct': 0.05,
            },
            'execution': {
                'dry_run': True,
            },
        }

    def _init_components(self):
        """Initialize pipeline components."""
        if HAS_PIPELINE:
            self.universe_builder = UniverseBuilder(config=self.config)
            self.regime_detector = RegimeDetector(config=self.config)

        if HAS_ML:
            self.forecaster = get_forecaster(config=self.config)
            self.meta_model = get_meta_model()

    def run(self, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the full daily pipeline.

        Args:
            tickers: Optional list of tickers (otherwise builds universe)

        Returns:
            Dict with pipeline results
        """
        logger.info(f"=" * 60)
        logger.info(f"Daily Pipeline Run - {self.as_of_date.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"=" * 60)

        results = {
            'as_of_date': self.as_of_date.isoformat(),
            'status': 'started',
            'stages': {},
        }

        try:
            # Stage 1: Universe Building
            logger.info("\n[Stage 1/7] Building Universe...")
            universe = self._build_universe(tickers)
            results['stages']['universe'] = {
                'n_tickers': len(universe),
                'tickers': universe,
            }
            logger.info(f"  Universe: {len(universe)} tickers")

            if not universe:
                raise ValueError("Empty universe")

            # Stage 2: Regime Classification
            logger.info("\n[Stage 2/7] Classifying Market Regime...")
            self._classify_regime()
            results['stages']['regime'] = {
                'regime': self.regime,
                'label': self.regime_info.get('regime_label', 'Unknown'),
                'confidence': self.regime_info.get('confidence', 0.0),
            }
            logger.info(f"  Regime: {self.regime_info.get('regime_label')} (confidence: {self.regime_info.get('confidence', 0):.2%})")

            # Stage 3: Forecasting
            logger.info("\n[Stage 3/7] Generating Forecasts...")
            forecasts = self._generate_forecasts(universe)
            results['stages']['forecasts'] = {
                'n_forecasts': len(forecasts),
                'avg_return': np.mean([f.expected_return for f in forecasts.values()]),
            }
            logger.info(f"  Generated {len(forecasts)} forecasts")

            # Stage 4: Signal Generation & Meta-Labeling
            logger.info("\n[Stage 4/7] Generating Trade Signals...")
            self.signals = self._generate_signals(forecasts)
            filtered = self._filter_signals(self.signals)
            results['stages']['signals'] = {
                'raw_signals': len(self.signals),
                'filtered_signals': len(filtered),
            }
            logger.info(f"  Signals: {len(self.signals)} raw -> {len(filtered)} filtered")

            if not filtered:
                logger.warning("No signals passed filters")
                results['status'] = 'no_signals'
                return results

            # Stage 5: Instrument Selection
            logger.info("\n[Stage 5/7] Selecting Instruments...")
            selections = self._select_instruments(filtered)
            results['stages']['instruments'] = summarize_selections(selections)
            logger.info(f"  Selected {len(selections)} instruments")

            # Stage 6: Allocation
            logger.info("\n[Stage 6/7] Optimizing Allocation...")
            self.allocations = self._allocate(filtered)
            results['stages']['allocation'] = {
                'expected_return': self.allocations.expected_return,
                'expected_vol': self.allocations.expected_vol,
                'sharpe': self.allocations.sharpe,
                'n_positions': len(self.allocations.weights[self.allocations.weights > 0.01]),
            }
            logger.info(f"  Expected Return: {self.allocations.expected_return:.2%}")
            logger.info(f"  Expected Vol: {self.allocations.expected_vol:.2%}")
            logger.info(f"  Sharpe: {self.allocations.sharpe:.3f}")

            # Stage 7: Order Generation
            logger.info("\n[Stage 7/7] Generating Orders...")
            self.orders = self._generate_orders(selections)
            results['stages']['orders'] = {
                'n_orders': len(self.orders),
                'orders': self.orders,
            }
            logger.info(f"  Generated {len(self.orders)} orders")

            results['status'] = 'completed'
            results['success'] = True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['success'] = False

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(f"Pipeline Status: {results['status'].upper()}")
        logger.info("=" * 60)

        return results

    def _build_universe(self, tickers: Optional[List[str]] = None) -> List[str]:
        """Build or use provided universe."""
        if tickers:
            return tickers

        # Default universe (large caps)
        default_universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK-B', 'JPM', 'V',
            'JNJ', 'WMT', 'PG', 'MA', 'HD',
            'XOM', 'CVX', 'PFE', 'ABBV', 'MRK',
        ]

        if HAS_PIPELINE:
            try:
                # Try to build universe properly
                universe_config = self.config.get('universe', {})
                # For now, return default
                return default_universe
            except Exception as e:
                logger.warning(f"Universe building failed: {e}")

        return default_universe

    def _classify_regime(self):
        """Classify current market regime."""
        if HAS_PIPELINE and HAS_DATA:
            try:
                # Load macro data
                end_date = self.as_of_date
                start_date = end_date - pd.Timedelta(days=365)

                macro_data = load_macro_features(start_date, end_date)

                # Get SPY for regime detection
                spy_data = fetch_data('SPY', period='1y')

                regime_result = classify_regime(
                    as_of_date=self.as_of_date,
                    price_data=spy_data,
                    config=self.config,
                )

                self.regime = regime_result['regime']
                self.regime_info = regime_result
                return

            except Exception as e:
                logger.warning(f"Regime classification failed: {e}")

        # Fallback to neutral regime
        self.regime = 2  # Neutral
        self.regime_info = {
            'regime': 2,
            'regime_label': 'Neutral',
            'confidence': 0.5,
        }

    def _generate_forecasts(self, universe: List[str]) -> Dict[str, Forecast]:
        """Generate forecasts for universe."""
        forecasts = {}
        horizon = self.config.get('forecaster', {}).get('horizon_days', 10)

        for ticker in universe:
            try:
                # Fetch price history
                if HAS_DATA:
                    price_history = fetch_data(ticker, period='1y')
                else:
                    # Generate dummy data for testing
                    dates = pd.date_range(end=self.as_of_date, periods=252, freq='D')
                    price_history = pd.DataFrame({
                        'Close': 100 + np.cumsum(np.random.randn(252) * 2),
                    }, index=dates)

                if price_history is None or len(price_history) < 60:
                    continue

                # Generate forecast
                if HAS_ML:
                    result = self.forecaster.forecast(price_history, horizon_days=horizon)
                    forecast = self.forecaster.to_pipeline_forecast(
                        ticker=ticker,
                        result=result,
                        as_of_date=self.as_of_date,
                        horizon_days=horizon,
                    )
                else:
                    # Simple momentum-based forecast
                    returns = price_history['Close'].pct_change().dropna()
                    forecast = Forecast(
                        ticker=ticker,
                        as_of_date=self.as_of_date,
                        horizon_days=horizon,
                        expected_return=float(returns.mean() * horizon),
                        volatility=float(returns.std() * np.sqrt(horizon)),
                        confidence=0.5,
                    )

                forecasts[ticker] = forecast

            except Exception as e:
                logger.debug(f"Forecast failed for {ticker}: {e}")
                continue

        return forecasts

    def _generate_signals(self, forecasts: Dict[str, Forecast]) -> List[TradeSignal]:
        """Generate trade signals from forecasts."""
        signals = []

        for ticker, forecast in forecasts.items():
            # Determine direction
            if forecast.expected_return > 0.01:
                direction = 'long'
            elif forecast.expected_return < -0.01:
                direction = 'short'
            else:
                direction = 'flat'

            # Create signal
            signal = TradeSignal(
                ticker=ticker,
                direction=direction,
                expected_return=forecast.expected_return,
                expected_vol=forecast.volatility,
                regime=self.regime,
                regime_label=self.regime_info.get('regime_label', 'Neutral'),
                forecast=forecast,
            )

            # Add meta-probability
            if HAS_ML and self.meta_model and self.meta_model.is_fitted:
                try:
                    meta_features = build_meta_features(
                        forecast=forecast.__dict__,
                        regime=self.regime,
                        regime_label=self.regime_info.get('regime_label', 'Neutral'),
                    )
                    signal.meta_prob = self.meta_model.predict_proba(meta_features)
                except Exception:
                    signal.meta_prob = None
            else:
                # Estimate meta_prob from signal characteristics
                signal.meta_prob = min(0.9, max(0.3, 0.5 + forecast.expected_return * 2))

            signals.append(signal)

        return signals

    def _filter_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Filter and rank signals."""
        if not signals:
            return []

        # Apply filters
        filtered = filter_signals(signals, config=self.config)

        # Rank and select top
        max_positions = self.config.get('allocation', {}).get('max_positions', 15)
        ranked = rank_signals(filtered, ranking_metric='risk_reward')
        selected = select_top_signals(ranked, max_signals=max_positions)

        return selected

    def _select_instruments(self, signals: List[TradeSignal]) -> List[InstrumentSelection]:
        """Select instruments for signals."""
        portfolio_value = self.config.get('allocation', {}).get('portfolio_value', 100000)

        # Get current prices (mock for now)
        prices = {}
        for signal in signals:
            # Would fetch real prices here
            prices[signal.ticker] = 100.0  # Placeholder

        selections = select_instruments_batch(
            signals=signals,
            portfolio_value=portfolio_value,
            prices=prices,
            config=self.config,
        )

        # Apply regime adjustments
        adjusted = apply_regime_adjustments(
            selections=selections,
            regime=self.regime,
            config=self.config,
        )

        return adjusted

    def _allocate(self, signals: List[TradeSignal]):
        """Allocate capital to signals."""
        portfolio_value = self.config.get('allocation', {}).get('portfolio_value', 100000)

        # Get returns data
        if HAS_DATA:
            # Fetch returns for allocation
            tickers = [s.ticker for s in signals]
            returns_list = []
            for ticker in tickers:
                try:
                    hist = fetch_data(ticker, period='1y')
                    if hist is not None and len(hist) > 60:
                        returns_list.append(hist['Close'].pct_change().dropna().rename(ticker))
                except:
                    continue

            if returns_list:
                returns = pd.concat(returns_list, axis=1).dropna()
            else:
                # Generate dummy returns
                dates = pd.date_range(end=self.as_of_date, periods=252, freq='D')
                returns = pd.DataFrame(
                    np.random.randn(252, len(tickers)) * 0.02,
                    index=dates,
                    columns=tickers,
                )
        else:
            tickers = [s.ticker for s in signals]
            dates = pd.date_range(end=self.as_of_date, periods=252, freq='D')
            returns = pd.DataFrame(
                np.random.randn(252, len(tickers)) * 0.02,
                index=dates,
                columns=tickers,
            )

        # Allocate
        result = allocate_from_signals(
            signals=signals,
            returns=returns,
            portfolio_value=portfolio_value,
            regime=self.regime,
            config=self.config,
        )

        return result

    def _generate_orders(self, selections: List[InstrumentSelection]) -> List[Dict]:
        """Generate executable orders."""
        orders = []

        for sel in selections:
            if sel.quantity <= 0:
                continue

            order = {
                'ticker': sel.ticker,
                'action': 'buy' if sel.direction == 'long' else 'sell',
                'quantity': sel.quantity,
                'instrument_type': sel.instrument_type.value,
                'strategy': sel.strategy_name,
                'notional': sel.notional_value,
            }

            # Add option details if applicable
            if sel.option_contract:
                order['option_contract'] = sel.option_contract
                order['strike'] = sel.strike
                order['expiration'] = sel.expiration.isoformat() if sel.expiration else None
                order['n_contracts'] = sel.n_contracts

            orders.append(order)

        return orders


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    if not HAS_YAML:
        logger.warning("PyYAML not available")
        return {}

    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run daily trading pipeline')
    parser.add_argument('--config', type=str, help='Path to config YAML')
    parser.add_argument('--tickers', type=str, nargs='+', help='Specific tickers to analyze')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no execution)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_config(args.config)

    # Override with CLI args
    if args.dry_run:
        config.setdefault('execution', {})['dry_run'] = True

    # Run pipeline
    pipeline = DailyPipeline(config=config)
    results = pipeline.run(tickers=args.tickers)

    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results written to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Date: {results['as_of_date']}")

    if 'stages' in results:
        stages = results['stages']

        if 'universe' in stages:
            print(f"\nUniverse: {stages['universe']['n_tickers']} tickers")

        if 'regime' in stages:
            print(f"Regime: {stages['regime']['label']} (conf: {stages['regime']['confidence']:.1%})")

        if 'signals' in stages:
            print(f"Signals: {stages['signals']['raw_signals']} raw -> {stages['signals']['filtered_signals']} filtered")

        if 'allocation' in stages:
            alloc = stages['allocation']
            print(f"\nAllocation:")
            print(f"  Expected Return: {alloc['expected_return']:.2%}")
            print(f"  Expected Vol: {alloc['expected_vol']:.2%}")
            print(f"  Sharpe: {alloc['sharpe']:.3f}")
            print(f"  Positions: {alloc['n_positions']}")

        if 'orders' in stages:
            print(f"\nOrders: {stages['orders']['n_orders']}")
            for order in stages['orders']['orders'][:5]:
                print(f"  {order['action'].upper()} {order['quantity']} {order['ticker']}")

    return 0 if results.get('success', False) else 1


if __name__ == '__main__':
    sys.exit(main())
