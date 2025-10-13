"""
Advanced Chart Generation and Visualization Module
===================================================

Generates comprehensive charts with detailed insights for portfolio analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple
import io
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


class ChartGenerator:
    """Generate comprehensive charts with insights for portfolio analysis."""

    def __init__(self, prices_df: pd.DataFrame, returns: pd.DataFrame,
                 holdings_df: pd.DataFrame, ticker: str):
        """
        Initialize chart generator.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Historical price data
        returns : pd.DataFrame
            Returns matrix
        holdings_df : pd.DataFrame
            Portfolio holdings
        ticker : str
            Primary ticker symbol
        """
        self.prices_df = prices_df
        self.returns = returns
        self.holdings_df = holdings_df
        self.ticker = ticker
        self.insights = {}

    def generate_price_performance_chart(self) -> Tuple[plt.Figure, Dict]:
        """
        Generate normalized price performance comparison.

        Returns
        -------
        fig : matplotlib.Figure
            Chart figure
        insights : dict
            Chart insights and statistics
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

        # Normalize prices to 100
        normalized = self.prices_df / self.prices_df.iloc[0] * 100

        # Plot each asset
        for col in normalized.columns:
            alpha = 1.0 if col == self.ticker else 0.6
            linewidth = 2.5 if col == self.ticker else 1.5
            ax1.plot(normalized.index, normalized[col],
                    label=col, alpha=alpha, linewidth=linewidth)

        ax1.set_title('Normalized Price Performance (Base 100)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Normalized Price')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax1.grid(True, alpha=0.3)

        # Calculate performance metrics
        final_values = normalized.iloc[-1]
        returns_total = (final_values - 100) / 100

        # Highlight primary ticker
        primary_return = returns_total[self.ticker]
        ax1.axhline(y=final_values[self.ticker], color='red',
                   linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(normalized.index[-1], final_values[self.ticker],
                f' {self.ticker}: {primary_return*100:.1f}%',
                verticalalignment='center', fontsize=10, fontweight='bold')

        # Volume subplot - show volatility instead
        rolling_vol = self.returns[self.ticker].rolling(20).std() * np.sqrt(252) * 100
        ax2.fill_between(rolling_vol.index, rolling_vol, alpha=0.3, color='orange')
        ax2.plot(rolling_vol.index, rolling_vol, color='orange', linewidth=1.5)
        ax2.set_title(f'{self.ticker} Rolling 20-Day Volatility (Annualized)', fontsize=12)
        ax2.set_ylabel('Volatility (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Generate insights
        insights = {
            'best_performer': final_values.idxmax(),
            'best_return': returns_total.max() * 100,
            'worst_performer': final_values.idxmin(),
            'worst_return': returns_total.min() * 100,
            'primary_return': primary_return * 100,
            'avg_return': returns_total.mean() * 100,
            'avg_volatility': rolling_vol.mean(),
            'current_volatility': rolling_vol.iloc[-1],
            'max_volatility': rolling_vol.max(),
        }

        return fig, insights

    def generate_correlation_heatmap(self) -> Tuple[plt.Figure, Dict]:
        """
        Generate correlation heatmap with clustering.

        Returns
        -------
        fig : matplotlib.Figure
            Chart figure
        insights : dict
            Correlation insights
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Calculate correlation matrix
        corr = self.returns.corr()

        # Heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                   cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                   square=True, ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')

        # Correlation with primary ticker
        primary_corr = corr[self.ticker].drop(self.ticker).sort_values(ascending=False)
        colors = ['green' if x > 0.5 else 'orange' if x > 0 else 'red' for x in primary_corr]
        primary_corr.plot(kind='barh', ax=ax2, color=colors, alpha=0.7)
        ax2.set_title(f'Correlation with {self.ticker}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.axvline(x=-0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # Generate insights
        insights = {
            'highest_correlation_asset': primary_corr.idxmax(),
            'highest_correlation': primary_corr.max(),
            'lowest_correlation_asset': primary_corr.idxmin(),
            'lowest_correlation': primary_corr.min(),
            'avg_correlation': primary_corr.mean(),
            'diversification_score': 1 - primary_corr.mean(),  # Higher is better
            'high_corr_count': (primary_corr > 0.7).sum(),
            'negative_corr_count': (primary_corr < 0).sum(),
        }

        return fig, insights

    def generate_risk_return_scatter(self, opt_result: Dict) -> Tuple[plt.Figure, Dict]:
        """
        Generate risk-return scatter plot with efficient frontier.

        Parameters
        ----------
        opt_result : dict
            Optimization results

        Returns
        -------
        fig : matplotlib.Figure
            Chart figure
        insights : dict
            Risk-return insights
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Calculate asset-level metrics
        mean_returns = self.returns.mean() * 252 * 100  # Annualized %
        volatility = self.returns.std() * np.sqrt(252) * 100  # Annualized %

        # Size bubbles by portfolio weight
        weights = opt_result['weights']
        sizes = np.array([weights.get(ticker, 0) * 5000 for ticker in mean_returns.index])

        # Color by Sharpe ratio
        sharpe_ratios = mean_returns / volatility

        scatter = ax.scatter(volatility, mean_returns, s=sizes, c=sharpe_ratios,
                           cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=1.5)

        # Add labels
        for ticker, vol, ret in zip(mean_returns.index, volatility, mean_returns):
            weight = weights.get(ticker, 0)
            if weight > 0.01:  # Only label assets with >1% weight
                ax.annotate(f'{ticker}\n({weight*100:.1f}%)',
                          (vol, ret), fontsize=9, ha='center',
                          xytext=(5, 5), textcoords='offset points')

        # Highlight primary ticker
        primary_vol = volatility[self.ticker]
        primary_ret = mean_returns[self.ticker]
        ax.scatter(primary_vol, primary_ret, s=300, c='red',
                  marker='*', edgecolors='darkred', linewidth=2, zorder=10,
                  label=f'{self.ticker} (Primary)')

        # Plot portfolio point
        port_vol = opt_result['port_vol'] * 100
        port_ret = opt_result['port_return'] * 100
        ax.scatter(port_vol, port_ret, s=400, c='blue',
                  marker='D', edgecolors='darkblue', linewidth=2, zorder=10,
                  label='Portfolio')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=12)

        ax.set_xlabel('Volatility (% annualized)', fontsize=12)
        ax.set_ylabel('Expected Return (% annualized)', fontsize=12)
        ax.set_title('Risk-Return Profile (bubble size = portfolio weight)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Generate insights
        insights = {
            'portfolio_return': port_ret,
            'portfolio_volatility': port_vol,
            'portfolio_sharpe': opt_result['sharpe'],
            'best_sharpe_asset': sharpe_ratios.idxmax(),
            'best_sharpe_value': sharpe_ratios.max(),
            'primary_sharpe': sharpe_ratios[self.ticker],
            'risk_reduction': ((primary_vol - port_vol) / primary_vol * 100),
            'return_improvement': ((port_ret - primary_ret) / abs(primary_ret) * 100) if primary_ret != 0 else 0,
        }

        return fig, insights

    def generate_allocation_charts(self) -> Tuple[plt.Figure, Dict]:
        """
        Generate portfolio allocation visualizations.

        Returns
        -------
        fig : matplotlib.Figure
            Chart figure
        insights : dict
            Allocation insights
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Filter holdings with weight > 0.5%
        holdings = self.holdings_df[self.holdings_df['weight'] > 0.005].sort_values('weight', ascending=False)

        # Pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        colors = plt.cm.Set3(np.linspace(0, 1, len(holdings)))
        wedges, texts, autotexts = ax1.pie(holdings['weight'], labels=holdings['ticker'],
                                           autopct='%1.1f%%', colors=colors,
                                           startangle=90, textprops={'fontsize': 10})
        ax1.set_title('Portfolio Allocation (% by weight)', fontsize=12, fontweight='bold')

        # Make primary ticker slice explode
        if self.ticker in holdings['ticker'].values:
            idx = holdings[holdings['ticker'] == self.ticker].index[0]
            wedges[list(holdings.index).index(idx)].set_edgecolor('red')
            wedges[list(holdings.index).index(idx)].set_linewidth(2)

        # Bar chart by dollar value
        ax2 = fig.add_subplot(gs[0, 1])
        holdings_sorted = self.holdings_df.sort_values('dollars', ascending=True)
        colors_bar = ['red' if t == self.ticker else 'steelblue' for t in holdings_sorted['ticker']]
        ax2.barh(holdings_sorted['ticker'], holdings_sorted['dollars'], color=colors_bar, alpha=0.7)
        ax2.set_xlabel('Investment ($)', fontsize=11)
        ax2.set_title('Portfolio Allocation ($ value)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Categorize assets
        ax3 = fig.add_subplot(gs[1, :])

        # Categorize by ticker type
        categories = {'Stocks': [], 'ETFs': []}
        for _, row in self.holdings_df.iterrows():
            ticker = row['ticker']
            # Simple heuristic: if ticker contains common ETF patterns
            if any(x in ticker.upper() for x in ['ETF', 'SPDR', 'QQQ', 'SPY', 'IWM', 'XL', 'VT', 'VO']):
                categories['ETFs'].append(row['weight'])
            else:
                categories['Stocks'].append(row['weight'])

        category_weights = {k: sum(v) for k, v in categories.items() if v}

        if category_weights:
            x_pos = np.arange(len(category_weights))
            ax3.bar(x_pos, list(category_weights.values()),
                   color=['steelblue', 'orange'], alpha=0.7, width=0.6)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(list(category_weights.keys()), fontsize=11)
            ax3.set_ylabel('Portfolio Weight', fontsize=11)
            ax3.set_title('Asset Class Distribution', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')

            # Add percentage labels
            for i, (cat, weight) in enumerate(category_weights.items()):
                ax3.text(i, weight + 0.01, f'{weight*100:.1f}%',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()

        # Generate insights
        insights = {
            'num_positions': len(self.holdings_df),
            'num_significant_positions': len(holdings),  # >0.5%
            'concentration': holdings['weight'].iloc[0] if len(holdings) > 0 else 0,  # Largest position
            'top3_concentration': holdings['weight'].head(3).sum() if len(holdings) >= 3 else holdings['weight'].sum(),
            'herfindahl_index': (self.holdings_df['weight'] ** 2).sum(),  # Concentration metric
            'stock_allocation': category_weights.get('Stocks', 0) * 100,
            'etf_allocation': category_weights.get('ETFs', 0) * 100,
        }

        return fig, insights

    def generate_all_insights_report(self) -> pd.DataFrame:
        """
        Compile all insights into a summary DataFrame.

        Returns
        -------
        pd.DataFrame
            Comprehensive insights table
        """
        all_insights = {}

        for key, insights in self.insights.items():
            for metric, value in insights.items():
                all_insights[f'{key}_{metric}'] = value

        df = pd.DataFrame([all_insights]).T
        df.columns = ['Value']
        df.index.name = 'Metric'

        return df
