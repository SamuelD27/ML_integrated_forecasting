"""
Monte Carlo Simulation Module
==============================

Portfolio performance simulations and risk analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple
import seaborn as sns


class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio analysis."""

    def __init__(self, returns: pd.DataFrame, weights: Dict[str, float],
                 initial_capital: float):
        """
        Initialize Monte Carlo simulator.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns matrix
        weights : dict
            Portfolio weights
        initial_capital : float
            Starting capital
        """
        self.returns = returns
        self.weights = weights
        self.initial_capital = initial_capital

        # Calculate portfolio statistics
        self.portfolio_returns = self._calculate_portfolio_returns()
        self.mean_return = self.portfolio_returns.mean()
        self.std_return = self.portfolio_returns.std()

    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate historical portfolio returns."""
        portfolio_ret = pd.Series(0, index=self.returns.index)

        for ticker, weight in self.weights.items():
            if ticker in self.returns.columns:
                portfolio_ret += self.returns[ticker] * weight

        return portfolio_ret

    def run_simulation(self, num_simulations: int = 10000,
                      time_horizon: int = 252) -> Dict:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        num_simulations : int
            Number of simulation paths (default: 10000)
        time_horizon : int
            Number of trading days to simulate (default: 252 = 1 year)

        Returns
        -------
        dict
            Simulation results and statistics
        """
        print(f"  Running {num_simulations:,} Monte Carlo simulations over {time_horizon} days...")

        # Initialize results array
        simulations = np.zeros((num_simulations, time_horizon))

        # Run simulations
        for i in range(num_simulations):
            # Generate random returns from normal distribution
            random_returns = np.random.normal(
                self.mean_return,
                self.std_return,
                time_horizon
            )

            # Calculate cumulative portfolio value
            price_series = self.initial_capital * np.cumprod(1 + random_returns)
            simulations[i, :] = price_series

        # Calculate statistics
        final_values = simulations[:, -1]

        results = {
            'simulations': simulations,
            'final_values': final_values,
            'mean_final': final_values.mean(),
            'median_final': np.median(final_values),
            'std_final': final_values.std(),
            'min_final': final_values.min(),
            'max_final': final_values.max(),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'probability_profit': (final_values > self.initial_capital).sum() / num_simulations,
            'probability_loss_10': (final_values < self.initial_capital * 0.9).sum() / num_simulations,
            'probability_loss_20': (final_values < self.initial_capital * 0.8).sum() / num_simulations,
            'expected_return': (final_values.mean() - self.initial_capital) / self.initial_capital,
            'var_95': np.percentile(final_values, 5) - self.initial_capital,  # Value at Risk
            'cvar_95': final_values[final_values <= np.percentile(final_values, 5)].mean() - self.initial_capital,  # Conditional VaR
        }

        return results

    def generate_simulation_chart(self, sim_results: Dict,
                                  num_paths_display: int = 100) -> Tuple[plt.Figure, Dict]:
        """
        Generate Monte Carlo simulation visualization.

        Parameters
        ----------
        sim_results : dict
            Simulation results from run_simulation()
        num_paths_display : int
            Number of simulation paths to display (default: 100)

        Returns
        -------
        fig : matplotlib.Figure
            Chart figure
        insights : dict
            Simulation insights
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        simulations = sim_results['simulations']
        final_values = sim_results['final_values']
        time_horizon = simulations.shape[1]

        # Plot 1: Simulation paths
        # Plot sample paths
        sample_indices = np.random.choice(len(simulations), num_paths_display, replace=False)
        for idx in sample_indices:
            ax1.plot(simulations[idx], alpha=0.1, color='steelblue', linewidth=0.5)

        # Plot percentiles
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'green', 'orange', 'red']
        for p, color in zip(percentiles, colors):
            percentile_path = np.percentile(simulations, p, axis=0)
            ax1.plot(percentile_path, color=color, linewidth=2,
                    label=f'{p}th percentile', linestyle='--' if p != 50 else '-')

        ax1.axhline(y=self.initial_capital, color='black',
                   linestyle='--', linewidth=1, label='Initial Capital')
        ax1.set_xlabel('Trading Days', fontsize=11)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax1.set_title(f'Monte Carlo Simulation - {len(simulations):,} Paths',
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Final value distribution
        ax2.hist(final_values, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(sim_results['mean_final'], color='green',
                   linestyle='--', linewidth=2, label=f"Mean: ${sim_results['mean_final']:,.0f}")
        ax2.axvline(sim_results['median_final'], color='blue',
                   linestyle='--', linewidth=2, label=f"Median: ${sim_results['median_final']:,.0f}")
        ax2.axvline(sim_results['percentile_5'], color='red',
                   linestyle='--', linewidth=2, label=f"5th %ile: ${sim_results['percentile_5']:,.0f}")
        ax2.axvline(sim_results['percentile_95'], color='orange',
                   linestyle='--', linewidth=2, label=f"95th %ile: ${sim_results['percentile_95']:,.0f}")
        ax2.axvline(self.initial_capital, color='black',
                   linestyle='-', linewidth=2, label=f"Initial: ${self.initial_capital:,.0f}")

        ax2.set_xlabel('Final Portfolio Value ($)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution of Final Portfolio Values', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Risk metrics over time
        # Calculate rolling VaR (5th percentile)
        var_path = np.percentile(simulations, 5, axis=0)
        median_path = np.percentile(simulations, 50, axis=0)
        drawdown = (var_path - self.initial_capital) / self.initial_capital * 100

        ax3.plot(drawdown, color='red', linewidth=2, label='VaR 95% Drawdown')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax3.set_xlabel('Trading Days', fontsize=11)
        ax3.set_ylabel('Drawdown (%)', fontsize=11)
        ax3.set_title('Value at Risk (95% confidence) - Maximum Expected Loss',
                     fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Probability cone
        percentile_bands = [(5, 95), (25, 75), (40, 60)]
        alphas = [0.2, 0.3, 0.4]
        colors_cone = ['red', 'orange', 'green']

        for (lower, upper), alpha, color in zip(percentile_bands, alphas, colors_cone):
            lower_band = np.percentile(simulations, lower, axis=0)
            upper_band = np.percentile(simulations, upper, axis=0)
            ax4.fill_between(range(time_horizon), lower_band, upper_band,
                           alpha=alpha, color=color,
                           label=f'{lower}th-{upper}th percentile')

        median_path = np.percentile(simulations, 50, axis=0)
        ax4.plot(median_path, color='darkgreen', linewidth=2.5, label='Median path')
        ax4.axhline(y=self.initial_capital, color='black',
                   linestyle='--', linewidth=1, label='Initial Capital')

        ax4.set_xlabel('Trading Days', fontsize=11)
        ax4.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax4.set_title('Probability Cone - Expected Range of Outcomes',
                     fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Generate insights
        insights = {
            'expected_return_pct': sim_results['expected_return'] * 100,
            'probability_profit_pct': sim_results['probability_profit'] * 100,
            'probability_loss_10_pct': sim_results['probability_loss_10'] * 100,
            'probability_loss_20_pct': sim_results['probability_loss_20'] * 100,
            'var_95_dollars': sim_results['var_95'],
            'cvar_95_dollars': sim_results['cvar_95'],
            'best_case': sim_results['max_final'],
            'worst_case': sim_results['min_final'],
            'median_outcome': sim_results['median_final'],
            'upside_potential': (sim_results['percentile_95'] - self.initial_capital) / self.initial_capital * 100,
            'downside_risk': (sim_results['percentile_5'] - self.initial_capital) / self.initial_capital * 100,
        }

        return fig, insights

    def scenario_analysis(self, scenarios: Dict[str, float]) -> pd.DataFrame:
        """
        Analyze portfolio under different market scenarios.

        Parameters
        ----------
        scenarios : dict
            Dictionary of scenario names and expected return adjustments
            Example: {'Bull Market': 0.15, 'Bear Market': -0.20, 'Crash': -0.40}

        Returns
        -------
        pd.DataFrame
            Scenario analysis results
        """
        results = []

        for scenario_name, return_adjustment in scenarios.items():
            # Adjust mean return for scenario
            adjusted_mean = self.mean_return + (return_adjustment / 252)  # Daily adjustment

            # Run quick simulation (1000 paths)
            num_sim = 1000
            time_horizon = 252
            final_values = []

            for _ in range(num_sim):
                random_returns = np.random.normal(adjusted_mean, self.std_return, time_horizon)
                final_value = self.initial_capital * np.prod(1 + random_returns)
                final_values.append(final_value)

            final_values = np.array(final_values)

            results.append({
                'Scenario': scenario_name,
                'Return Adjustment': f'{return_adjustment*100:+.0f}%',
                'Expected Final Value': final_values.mean(),
                'Expected Return': (final_values.mean() - self.initial_capital) / self.initial_capital * 100,
                'Median Final Value': np.median(final_values),
                '5th Percentile': np.percentile(final_values, 5),
                '95th Percentile': np.percentile(final_values, 95),
                'Prob. of Profit': (final_values > self.initial_capital).sum() / num_sim * 100,
            })

        return pd.DataFrame(results)
