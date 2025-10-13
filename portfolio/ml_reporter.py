"""
ML-Enhanced Excel Reporter
===========================

Comprehensive Excel reporting with:
- Holdings with ML scores
- Hedge overlay details
- Performance and risk-return charts
- Feature importance visualization
- Provider diagnostics
- Regime analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tempfile

sns.set_style('whitegrid')


class MLPortfolioReporter:
    """Enhanced Excel reporter with ML diagnostics."""

    def __init__(self, output_path: str):
        """
        Initialize reporter.

        Parameters
        ----------
        output_path : str
            Output Excel file path
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.charts = {}

    def generate_report(self, analysis: Dict):
        """
        Generate comprehensive Excel report.

        Parameters
        ----------
        analysis : dict
            Complete analysis results containing:
            - holdings_df
            - opt_result
            - ml_diagnostics (optional)
            - regime_info (optional)
            - options_overlay (optional)
            - provider_diagnostics (optional)
        """
        print(f"\n[Reporter] Generating comprehensive Excel report...")

        try:
            # Generate charts first
            self._generate_all_charts(analysis)

            # Create Excel file
            with pd.ExcelWriter(self.output_path, engine='xlsxwriter') as writer:
                workbook = writer.book

                # Define formats
                formats = self._create_formats(workbook)

                # Sheet 1: Portfolio Summary
                self._write_summary_sheet(workbook, analysis, formats)

                # Sheet 2: ML Diagnostics
                if 'ml_diagnostics' in analysis and analysis['ml_diagnostics']:
                    self._write_ml_diagnostics_sheet(workbook, analysis, formats)

                # Sheet 3: Charts
                self._write_charts_sheet(workbook, analysis, formats)

                # Sheet 4: Detailed Diagnostics
                self._write_diagnostics_sheet(workbook, analysis, formats)

            print(f"  ✓ Report saved: {self.output_path}")

        except Exception as e:
            print(f"  Error generating report: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            raise

    def _create_formats(self, workbook):
        """Create Excel cell formats."""
        return {
            'title': workbook.add_format({
                'bold': True, 'font_size': 16, 'font_color': '#1F4788',
                'align': 'center', 'valign': 'vcenter'
            }),
            'header': workbook.add_format({
                'bold': True, 'bg_color': '#4472C4', 'font_color': 'white',
                'border': 1, 'align': 'center'
            }),
            'section': workbook.add_format({
                'bold': True, 'font_size': 12, 'bg_color': '#D9E1F2', 'border': 1
            }),
            'number': workbook.add_format({'num_format': '#,##0.00'}),
            'percent': workbook.add_format({'num_format': '0.00%'}),
            'integer': workbook.add_format({'num_format': '#,##0'}),
            'highlight_positive': workbook.add_format({
                'bg_color': '#C6EFCE', 'font_color': '#006100'
            }),
            'highlight_negative': workbook.add_format({
                'bg_color': '#FFC7CE', 'font_color': '#9C0006'
            }),
        }

    def _write_summary_sheet(self, workbook, analysis: Dict, formats: Dict):
        """Write portfolio summary sheet."""
        ws = workbook.add_worksheet('Portfolio Summary')
        ws.set_column('A:A', 30)
        ws.set_column('B:F', 16)

        row = 0

        # Title
        ws.merge_range(row, 0, row, 5,
                      f"ML-ENHANCED PORTFOLIO: {analysis['ticker']}",
                      formats['title'])
        row += 2

        # Parameters
        ws.merge_range(row, 0, row, 5, 'PORTFOLIO PARAMETERS', formats['section'])
        row += 1

        params = [
            ['Primary Ticker', analysis.get('ticker', 'N/A')],
            ['Total Capital', f"${analysis.get('capital', 0):,.2f}"],
            ['RRR (Reward-Risk)', f"{analysis.get('rrr', 0):.2f}"],
            ['Universe Size', len(analysis.get('universe', []))],
            ['ML Enhanced', 'Yes' if analysis.get('ml_diagnostics') else 'No'],
        ]

        regime_info = analysis.get('regime_info')
        if regime_info:
            regime = regime_info.get('regime', 'unknown')
            confidence = regime_info.get('confidence', 0)
            params.append(['Market Regime', f"{regime.upper()} ({confidence:.0%} conf.)"])

        for label, value in params:
            ws.write(row, 0, label)
            ws.write(row, 1, value)
            row += 1

        row += 1

        # Holdings
        ws.merge_range(row, 0, row, 5, 'PORTFOLIO HOLDINGS', formats['section'])
        row += 1

        headers = ['Ticker', 'Weight', 'Dollars', 'Shares', 'Price']
        if 'ml_rankings' in analysis:
            headers.append('ML Score')

        for col, header in enumerate(headers):
            ws.write(row, col, header, formats['header'])
        row += 1

        holdings_df = analysis.get('holdings_df')
        if holdings_df is None or holdings_df.empty:
            ws.write(row, 0, 'No holdings data available')
            row += 1
        else:
            holdings_df = holdings_df.sort_values('weight', ascending=False)

            for _, holding in holdings_df.iterrows():
                ws.write(row, 0, holding['ticker'])
                ws.write(row, 1, f"{holding['weight']*100:.2f}%")
                ws.write(row, 2, f"${holding['dollars']:,.2f}")
                ws.write(row, 3, int(holding['shares']))
                ws.write(row, 4, f"${holding['price']:.2f}")

                # ML score if available
                if 'ml_rankings' in analysis:
                    ml_rankings = analysis['ml_rankings']
                    if ml_rankings is not None:
                        ticker_score = ml_rankings[ml_rankings['ticker'] == holding['ticker']]
                        if not ticker_score.empty:
                            score = ticker_score.iloc[0]['ml_score']
                            ws.write(row, 5, f"{score:.3f}")

                row += 1

        row += 1

        # Portfolio Metrics
        ws.merge_range(row, 0, row, 5, 'PORTFOLIO METRICS', formats['section'])
        row += 1

        opt = analysis['opt_result']
        metrics = [
            ['Metric', 'Value', 'Interpretation'],
            ['Expected Return', f"{opt['port_return']*100:.2f}%", 'Annual'],
            ['Volatility', f"{opt['port_vol']*100:.2f}%", 'Annual'],
            ['Sharpe Ratio', f"{opt['sharpe']:.3f}", 'Risk-adjusted return'],
        ]

        if 'cvar' in opt:
            metrics.append(['CVaR (95%)', f"{opt['cvar']*100:.2f}%", 'Tail risk'])

        if opt.get('realized_beta') is not None:
            metrics.append(['Portfolio Beta', f"{opt['realized_beta']:.3f}", 'Market sensitivity'])

        for data in metrics:
            for col, val in enumerate(data):
                if col == 0:
                    ws.write(row, col, val, formats['header'])
                else:
                    ws.write(row, col, val)
            row += 1

        row += 1

        # Options Overlay
        overlay = analysis.get('options_overlay')
        if overlay:
            ws.merge_range(row, 0, row, 5, 'OPTIONS HEDGE OVERLAY', formats['section'])
            row += 1

            hedge_info = [
                ['Strategy', overlay.get('strategy', 'N/A').upper()],
                ['Strike', f"${overlay.get('strike', 0):.2f}"],
                ['Expiration', overlay.get('expiration', 'N/A')],
                ['Contracts', overlay.get('contracts', 0)],
                ['Premium', f"${overlay.get('total_premium', 0):,.2f}"],
                ['Coverage', f"{overlay.get('hedge_coverage', 0)*100:.1f}%"],
            ]

            for label, value in hedge_info:
                ws.write(row, 0, label)
                ws.write(row, 1, value)
                row += 1

    def _write_ml_diagnostics_sheet(self, workbook, analysis: Dict, formats: Dict):
        """Write ML diagnostics sheet."""
        ws = workbook.add_worksheet('ML Diagnostics')
        ws.set_column('A:A', 30)
        ws.set_column('B:C', 20)

        row = 0

        ws.merge_range(row, 0, row, 2, 'MACHINE LEARNING DIAGNOSTICS', formats['title'])
        row += 2

        ml_diag = analysis.get('ml_diagnostics')

        if not ml_diag:
            ws.write(row, 0, 'ML features not enabled for this portfolio')
            return

        # Training Metrics
        ws.merge_range(row, 0, row, 2, 'MODEL TRAINING', formats['section'])
        row += 1

        if 'train_metrics' in ml_diag:
            train = ml_diag['train_metrics']
            if train.get('status') == 'success':
                metrics = [
                    ['Model Type', ml_diag.get('model_type', 'lightgbm')],
                    ['Training Samples', train.get('n_samples', 0)],
                    ['Features Used', train.get('n_features', 0)],
                    ['CV Score (NDCG)', f"{train.get('cv_score_mean', 0):.4f}"],
                    ['CV Std Dev', f"{train.get('cv_score_std', 0):.4f}"],
                ]

                for label, value in metrics:
                    ws.write(row, 0, label)
                    ws.write(row, 1, str(value))
                    row += 1
            else:
                ws.write(row, 0, 'Status')
                ws.write(row, 1, 'Insufficient training data')
                row += 1

        row += 1

        # Feature Importance
        if 'feature_importance' in ml_diag and not ml_diag['feature_importance'].empty:
            ws.merge_range(row, 0, row, 2, 'TOP FEATURE IMPORTANCE', formats['section'])
            row += 1

            ws.write(row, 0, 'Feature', formats['header'])
            ws.write(row, 1, 'Importance', formats['header'])
            row += 1

            feat_imp = ml_diag['feature_importance'].head(15)
            for _, feat_row in feat_imp.iterrows():
                ws.write(row, 0, feat_row['feature'])
                ws.write(row, 1, f"{feat_row['importance']:.2f}")
                row += 1

        row += 1

        # Stock Rankings
        if 'ml_rankings' in analysis:
            ws.merge_range(row, 0, row, 2, 'ML STOCK RANKINGS', formats['section'])
            row += 1

            ws.write(row, 0, 'Rank', formats['header'])
            ws.write(row, 1, 'Ticker', formats['header'])
            ws.write(row, 2, 'ML Score', formats['header'])
            row += 1

            rankings = analysis['ml_rankings'].head(20)
            for _, rank_row in rankings.iterrows():
                ws.write(row, 0, int(rank_row['ml_rank']))
                ws.write(row, 1, rank_row['ticker'])
                ws.write(row, 2, f"{rank_row['ml_score']:.4f}")
                row += 1

    def _write_charts_sheet(self, workbook, analysis: Dict, formats: Dict):
        """Write charts sheet."""
        ws = workbook.add_worksheet('Charts')

        row = 0
        ws.merge_range(row, 0, row, 6, 'PORTFOLIO VISUALIZATIONS', formats['title'])
        row += 2

        chart_list = [
            ('allocation', 'Portfolio Allocation'),
            ('performance', 'Performance Comparison'),
            ('risk_return', 'Risk-Return Profile'),
            ('feature_importance', 'ML Feature Importance'),
        ]

        for chart_key, chart_title in chart_list:
            if chart_key in self.charts:
                ws.write(row, 0, chart_title, formats['section'])
                row += 1
                try:
                    ws.insert_image(row, 0, str(self.charts[chart_key]),
                                  {'x_scale': 0.6, 'y_scale': 0.6})
                    row += 24
                except Exception as e:
                    print(f"  Warning: Could not insert chart {chart_key}: {e}")
                    row += 2

    def _write_diagnostics_sheet(self, workbook, analysis: Dict, formats: Dict):
        """Write detailed diagnostics sheet."""
        ws = workbook.add_worksheet('Diagnostics')
        ws.set_column('A:A', 30)
        ws.set_column('B:C', 25)

        row = 0

        ws.merge_range(row, 0, row, 2, 'SYSTEM DIAGNOSTICS', formats['title'])
        row += 2

        # Provider diagnostics
        provider_diag = analysis.get('provider_diagnostics')
        if provider_diag is not None and not provider_diag.empty:
            ws.merge_range(row, 0, row, 2, 'DATA PROVIDER STATISTICS', formats['section'])
            row += 1

            for col_idx, col_name in enumerate(provider_diag.columns):
                ws.write(row, col_idx, col_name, formats['header'])
            row += 1

            for _, prow in provider_diag.iterrows():
                for col_idx, val in enumerate(prow):
                    ws.write(row, col_idx, str(val))
                row += 1

            row += 1

        # Regime Information
        regime_info = analysis.get('regime_info')
        if regime_info:
            ws.merge_range(row, 0, row, 2, 'REGIME DETECTION', formats['section'])
            row += 1

            regime_data = [
                ['Current Regime', regime_info.get('regime', 'unknown').upper()],
                ['Confidence', f"{regime_info.get('confidence', 0):.0%}"],
            ]

            metrics = regime_info.get('metrics', {})
            if metrics:
                for key, val in metrics.items():
                    if isinstance(val, float):
                        regime_data.append([key, f"{val:.4f}"])
                    else:
                        regime_data.append([key, str(val)])

            for label, value in regime_data:
                ws.write(row, 0, label)
                ws.write(row, 1, value)
                row += 1

            row += 1

        # Optimization Details
        ws.merge_range(row, 0, row, 2, 'OPTIMIZATION DETAILS', formats['section'])
        row += 1

        opt = analysis.get('opt_result', {})
        opt_details = [
            ['Risk Aversion', f"{opt.get('risk_aversion', 0):.2f}"],
            ['Shrinkage Coefficient', f"{opt.get('shrinkage_coeff', 0):.3f}"],
            ['Assets in Portfolio', len(opt.get('weights', []))],
        ]

        for label, value in opt_details:
            ws.write(row, 0, label)
            ws.write(row, 1, value)
            row += 1

    def _generate_all_charts(self, analysis: Dict):
        """Generate all visualization charts."""
        print("  Generating charts...")

        # 1. Portfolio allocation pie chart
        self._create_allocation_chart(analysis)

        # 2. Performance comparison
        self._create_performance_chart(analysis)

        # 3. Risk-return scatter
        self._create_risk_return_chart(analysis)

        # 4. Feature importance (if ML used)
        if 'ml_diagnostics' in analysis and analysis['ml_diagnostics']:
            self._create_feature_importance_chart(analysis)

        print(f"    ✓ Generated {len(self.charts)} charts")

    def _create_allocation_chart(self, analysis: Dict):
        """Create portfolio allocation pie chart."""
        fig, ax = plt.subplots(figsize=(10, 7))

        weights = analysis['opt_result']['weights']
        weights = weights[weights > 0.01].sort_values(ascending=False)  # Filter small

        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))

        wedges, texts, autotexts = ax.pie(
            weights.values,
            labels=weights.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )

        for text in texts:
            text.set_fontsize(11)
            text.set_weight('bold')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')

        ax.set_title('Portfolio Allocation', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        path = self.temp_dir / 'allocation.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        self.charts['allocation'] = path
        plt.close(fig)

    def _create_performance_chart(self, analysis: Dict):
        """Create performance comparison chart."""
        if 'prices_df' not in analysis:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        prices_df = analysis['prices_df']
        weights = analysis['opt_result']['weights']

        # Normalize to 100
        norm_prices = prices_df / prices_df.iloc[0] * 100

        # Portfolio performance
        port_perf = (norm_prices * weights).sum(axis=1)

        # Plot portfolio
        ax.plot(port_perf.index, port_perf.values, linewidth=2.5,
               label='Portfolio', color='#2E86AB', alpha=0.9)

        # Plot primary ticker
        if analysis['ticker'] in norm_prices.columns:
            primary_perf = norm_prices[analysis['ticker']]
            ax.plot(primary_perf.index, primary_perf.values, linewidth=2,
                   label=f"{analysis['ticker']} Only", color='#F18F01',
                   alpha=0.7, linestyle='--')

        ax.set_title('Performance Comparison (Normalized to 100)',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.temp_dir / 'performance.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        self.charts['performance'] = path
        plt.close(fig)

    def _create_risk_return_chart(self, analysis: Dict):
        """Create risk-return scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 7))

        opt = analysis['opt_result']
        mu = opt['mu_annual']
        cov = opt['cov_annual']
        vols = np.sqrt(np.diag(cov))

        # Plot individual assets
        scatter = ax.scatter(vols * 100, mu * 100, s=200, alpha=0.6,
                           c=range(len(mu)), cmap='viridis', edgecolors='black',
                           linewidth=1.5)

        # Add labels
        for i, ticker in enumerate(mu.index):
            ax.annotate(ticker, (vols[i]*100, mu[i]*100),
                       fontsize=9, ha='center', va='bottom')

        # Plot portfolio
        port_vol = opt['port_vol']
        port_ret = opt['port_return']
        ax.scatter([port_vol*100], [port_ret*100], s=800, marker='*',
                  color='red', edgecolor='black', linewidth=2.5,
                  label='Portfolio', zorder=10)

        ax.set_xlabel('Volatility (% annual)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Expected Return (% annual)', fontsize=11, fontweight='bold')
        ax.set_title('Risk-Return Profile', fontsize=13, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.temp_dir / 'risk_return.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        self.charts['risk_return'] = path
        plt.close(fig)

    def _create_feature_importance_chart(self, analysis: Dict):
        """Create feature importance bar chart."""
        ml_diag = analysis.get('ml_diagnostics', {})
        if 'feature_importance' not in ml_diag or ml_diag['feature_importance'].empty:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        feat_imp = ml_diag['feature_importance'].head(20)

        y_pos = np.arange(len(feat_imp))
        ax.barh(y_pos, feat_imp['importance'], color='#4472C4', alpha=0.8,
               edgecolor='black', linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_imp['feature'], fontsize=9)
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax.set_title('Top 20 ML Feature Importance', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        path = self.temp_dir / 'feature_importance.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        self.charts['feature_importance'] = path
        plt.close(fig)
