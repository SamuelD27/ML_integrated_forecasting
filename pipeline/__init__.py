"""
Pipeline Module
================
6-phase trading pipeline for quantitative stock analysis and trading.

Phases:
1. Universe Builder - Fundamentals + Factor scoring
2. Regime Detection - Market regime classification
3. Ensemble Forecaster - N-BEATS, LSTM, TFT forecasts
4. Meta-Labeling - Triple barrier + XGBoost/RF
5. Instrument Selection - Options strategy selection
6. CVaR Allocation - Portfolio optimization

Usage:
    >>> from pipeline.core_types import StockSnapshot, TradeSignal
    >>> from pipeline.universe_builder import build_universe
    >>> from pipeline.regime_utils import get_regime_for_date
"""
from pipeline.core_types import StockSnapshot, TradeSignal

__all__ = ['StockSnapshot', 'TradeSignal']
