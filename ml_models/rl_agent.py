"""
Deep Reinforcement Learning Trading Agent
==========================================

This module implements a PPO (Proximal Policy Optimization) agent for
algorithmic trading with risk-aware reward functions and position sizing.

Features:
- Custom trading environment (OpenAI Gym compatible)
- PPO with actor-critic architecture
- Risk constraints (max drawdown, position limits)
- Sharpe ratio optimization
- Transaction cost modeling
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from collections import deque
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Configuration for trading environment and agent."""

    # Environment settings
    initial_balance: float = 100000.0
    max_position_size: float = 1.0  # Max fraction of portfolio in single position
    transaction_cost: float = 0.001  # 0.1% transaction cost
    slippage: float = 0.0005  # 0.05% slippage

    # Risk constraints
    max_drawdown: float = 0.20  # 20% max drawdown
    stop_loss: float = 0.05  # 5% stop loss per position
    take_profit: float = 0.10  # 10% take profit

    # Reward settings
    risk_free_rate: float = 0.02  # Annual risk-free rate
    lookback_window: int = 20  # For Sharpe ratio calculation

    # Agent settings
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clip parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Training settings
    n_steps: int = 2048  # Steps per update
    n_epochs: int = 10  # PPO epochs
    batch_size: int = 64  # Mini-batch size


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning.

    State space includes:
    - Price features (OHLCV, returns, technical indicators)
    - Portfolio state (position, PnL, cash)
    - Market features (volatility, trend)

    Action space:
    - Discrete: {Sell, Hold, Buy} × position sizes
    - Or Continuous: [-1, 1] where sign is direction, magnitude is size
    """

    def __init__(self, data: pd.DataFrame, features: pd.DataFrame,
                 config: TradingConfig = None, discrete_actions: bool = True):
        """
        Initialize trading environment.

        Args:
            data: DataFrame with OHLCV data
            features: DataFrame with engineered features
            config: Trading configuration
            discrete_actions: Whether to use discrete or continuous actions
        """
        super().__init__()

        self.data = data
        self.features = features
        self.config = config or TradingConfig()
        self.discrete_actions = discrete_actions

        # Ensure alignment
        self.features = self.features.loc[self.data.index]

        # Define action and observation spaces
        if discrete_actions:
            # 9 actions: 3 directions × 3 position sizes
            # Sell (large, medium, small), Hold, Buy (small, medium, large)
            self.action_space = spaces.Discrete(9)
            self.action_mapping = {
                0: (-1.0, 1.0),    # Sell large
                1: (-1.0, 0.5),    # Sell medium
                2: (-1.0, 0.25),   # Sell small
                3: (0.0, 0.0),     # Hold
                4: (1.0, 0.25),    # Buy small
                5: (1.0, 0.5),     # Buy medium
                6: (1.0, 1.0),     # Buy large
                7: (-0.5, 0.5),    # Reduce position by half
                8: (0.5, 0.5),     # Increase position by half
            }
        else:
            # Continuous action: [-1, 1]
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space: features + portfolio state
        n_features = len(self.features.columns)
        n_portfolio_features = 10  # Position, PnL, cash, etc.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features + n_portfolio_features,),
            dtype=np.float32
        )

        # Initialize episode variables
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Random start point (leave some data for trading)
        max_start = len(self.data) - 100
        self.current_step = np.random.randint(50, max_start) if max_start > 50 else 50

        # Initialize portfolio
        self.balance = self.config.initial_balance
        self.position = 0.0  # Number of shares
        self.entry_price = 0.0
        self.peak_balance = self.balance

        # Track metrics
        self.trades = []
        self.portfolio_values = [self.balance]
        self.positions = [0.0]
        self.rewards = []

        return self._get_observation(), {}

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step.

        Args:
            action: Trading action (discrete or continuous)

        Returns:
            observation: Next state
            reward: Step reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Parse action
        if self.discrete_actions:
            direction, size = self.action_mapping[action]
        else:
            direction = float(action[0])
            size = abs(direction)
            direction = np.sign(direction)

        # Get current price
        current_price = self.data.iloc[self.current_step]['Close']

        # Execute trade
        old_position = self.position
        old_balance = self.balance

        if direction != 0:
            # Calculate position change
            max_position_value = self.balance * self.config.max_position_size
            max_shares = max_position_value / current_price

            if direction > 0:  # Buy
                shares_to_buy = min(max_shares * size, max_shares - abs(self.position))
                cost = shares_to_buy * current_price * (1 + self.config.transaction_cost + self.config.slippage)

                if cost <= self.balance:
                    self.balance -= cost
                    self.position += shares_to_buy
                    if self.position > 0 and old_position <= 0:
                        self.entry_price = current_price

                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost
                    })

            elif direction < 0:  # Sell
                if self.position > 0:
                    shares_to_sell = min(self.position * abs(size), self.position)
                    revenue = shares_to_sell * current_price * (1 - self.config.transaction_cost - self.config.slippage)

                    self.balance += revenue
                    self.position -= shares_to_sell

                    self.trades.append({
                        'step': self.current_step,
                        'action': 'sell',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'revenue': revenue
                    })

        # Calculate portfolio value
        portfolio_value = self.balance + self.position * current_price
        self.portfolio_values.append(portfolio_value)
        self.positions.append(self.position)

        # Update peak for drawdown calculation
        if portfolio_value > self.peak_balance:
            self.peak_balance = portfolio_value

        # Calculate reward
        reward = self._calculate_reward(old_balance, old_position, current_price)
        self.rewards.append(reward)

        # Check termination conditions
        self.current_step += 1

        terminated = False
        truncated = False

        # Check if episode should end
        if self.current_step >= len(self.data) - 1:
            truncated = True

        # Check risk constraints
        drawdown = (self.peak_balance - portfolio_value) / self.peak_balance
        if drawdown > self.config.max_drawdown:
            terminated = True
            reward -= 10  # Penalty for exceeding max drawdown

        # Check if bankrupt
        if self.balance < 0 or portfolio_value < self.config.initial_balance * 0.5:
            terminated = True
            reward -= 20  # Penalty for bankruptcy

        # Get next observation
        observation = self._get_observation()

        # Additional info
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'drawdown': drawdown,
            'n_trades': len(self.trades),
            'current_price': current_price
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Market features
        market_features = self.features.iloc[self.current_step].values

        # Get recent price history
        lookback = min(20, self.current_step)
        if lookback > 0:
            recent_prices = self.data.iloc[self.current_step-lookback:self.current_step]['Close'].values
            returns = np.diff(recent_prices) / recent_prices[:-1] if len(recent_prices) > 1 else np.array([0])
            recent_return = returns.mean() if len(returns) > 0 else 0
            recent_volatility = returns.std() if len(returns) > 0 else 0
        else:
            recent_return = 0
            recent_volatility = 0

        # Portfolio features
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_value = self.balance + self.position * current_price

        portfolio_features = np.array([
            self.position / 1000.0,  # Normalized position
            self.balance / self.config.initial_balance,  # Normalized balance
            portfolio_value / self.config.initial_balance,  # Normalized portfolio value
            (portfolio_value - self.config.initial_balance) / self.config.initial_balance,  # Return
            self.position * current_price / portfolio_value if portfolio_value > 0 else 0,  # Position ratio
            (self.peak_balance - portfolio_value) / self.peak_balance if self.peak_balance > 0 else 0,  # Drawdown
            recent_return,  # Recent return
            recent_volatility,  # Recent volatility
            len(self.trades) / 100.0,  # Trade frequency (normalized)
            float(self.position > 0)  # Position indicator
        ], dtype=np.float32)

        # Combine features
        observation = np.concatenate([market_features, portfolio_features])

        return observation

    def _calculate_reward(self, old_balance: float, old_position: float,
                         current_price: float) -> float:
        """
        Calculate step reward using Sharpe ratio and risk-adjusted returns.

        Args:
            old_balance: Previous balance
            old_position: Previous position
            current_price: Current price

        Returns:
            Reward value
        """
        # Current portfolio value
        old_portfolio = old_balance + old_position * self.data.iloc[self.current_step - 1]['Close']
        new_portfolio = self.balance + self.position * current_price

        # Simple return
        simple_return = (new_portfolio - old_portfolio) / old_portfolio if old_portfolio > 0 else 0

        # Risk-adjusted return (simplified Sharpe ratio)
        if len(self.portfolio_values) >= self.config.lookback_window:
            recent_values = self.portfolio_values[-self.config.lookback_window:]
            recent_returns = np.diff(recent_values) / recent_values[:-1]

            if len(recent_returns) > 0 and recent_returns.std() > 0:
                sharpe = np.sqrt(252) * recent_returns.mean() / recent_returns.std()
                risk_adjusted_reward = simple_return * (1 + sharpe)
            else:
                risk_adjusted_reward = simple_return
        else:
            risk_adjusted_reward = simple_return

        # Transaction penalty (to discourage overtrading)
        transaction_penalty = -0.001 if len(self.trades) > 0 and self.trades[-1]['step'] == self.current_step else 0

        # Position holding bonus (to encourage maintaining profitable positions)
        holding_bonus = 0.0001 * abs(self.position) if simple_return > 0 else 0

        # Combine rewards
        reward = risk_adjusted_reward * 100 + transaction_penalty + holding_bonus

        return reward

    def render(self, mode: str = 'human'):
        """Render current state (for debugging)."""
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_value = self.balance + self.position * current_price

        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Position: {self.position:.2f} shares")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Trades: {len(self.trades)}")
        print("-" * 40)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO.

    Shared backbone with separate heads for policy and value.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_actions: int = 9, continuous: bool = False):
        """
        Initialize actor-critic network.

        Args:
            input_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            n_actions: Number of discrete actions (ignored if continuous)
            continuous: Whether to use continuous actions
        """
        super().__init__()

        self.continuous = continuous

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Actor (policy) head
        if continuous:
            # Output mean and log_std for continuous actions
            self.actor_mean = nn.Linear(hidden_dim, 1)
            self.actor_log_std = nn.Parameter(torch.zeros(1))
        else:
            self.actor = nn.Linear(hidden_dim, n_actions)

        # Critic (value) head
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            state: State tensor

        Returns:
            action_probs/action_mean: Policy output
            value: Value estimate
        """
        shared_features = self.shared(state)

        if self.continuous:
            action_mean = torch.tanh(self.actor_mean(shared_features))
            value = self.critic(shared_features)
            return action_mean, value
        else:
            action_logits = self.actor(shared_features)
            value = self.critic(shared_features)
            return action_logits, value

    def get_action_and_value(self, state: torch.Tensor,
                           action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            state: State tensor
            action: Optional action for log_prob calculation

        Returns:
            action: Selected or given action
            log_prob: Log probability of action
            entropy: Entropy of distribution
            value: Value estimate
        """
        if self.continuous:
            action_mean, value = self(state)
            action_std = torch.exp(self.actor_log_std)

            dist = Normal(action_mean, action_std)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            action_logits, value = self(state)
            dist = Categorical(logits=action_logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) agent for trading.
    """

    def __init__(self, env: TradingEnvironment, config: TradingConfig = None,
                 device: str = None):
        """
        Initialize PPO agent.

        Args:
            env: Trading environment
            config: Training configuration
            device: Device for training
        """
        self.env = env
        self.config = config or TradingConfig()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Get dimensions
        self.state_dim = env.observation_space.shape[0]
        self.discrete = isinstance(env.action_space, spaces.Discrete)
        self.n_actions = env.action_space.n if self.discrete else 1

        # Initialize network
        self.network = ActorCriticNetwork(
            input_dim=self.state_dim,
            n_actions=self.n_actions,
            continuous=not self.discrete
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

        # Buffers for experience
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        logger.info(f"Initialized PPO agent on {self.device}")

    def collect_experience(self, n_steps: int) -> Dict[str, torch.Tensor]:
        """
        Collect experience by interacting with environment.

        Args:
            n_steps: Number of steps to collect

        Returns:
            Dictionary of collected experience
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for _ in range(n_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get action from policy
            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(state_tensor)

            # Take action in environment
            action_np = action.cpu().numpy().squeeze()
            next_state, reward, terminated, truncated, info = self.env.step(action_np)

            # Store experience
            self.states.append(state)
            self.actions.append(action.cpu().numpy().squeeze())
            self.rewards.append(reward)
            self.values.append(value.cpu().numpy().squeeze())
            self.log_probs.append(log_prob.cpu().numpy().squeeze())
            self.dones.append(terminated or truncated)

            episode_reward += reward
            episode_length += 1

            # Reset if episode ended
            if terminated or truncated:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state

        # Compute returns and advantages
        experience = self._process_experience()

        return experience

    def _process_experience(self) -> Dict[str, torch.Tensor]:
        """Process collected experience and compute advantages."""
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device) if self.discrete else torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)

        # Compute returns and advantages using GAE
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        last_return = 0
        last_value = 0
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            if dones[t]:
                last_return = rewards[t]
                last_value = rewards[t]
                last_advantage = 0
            else:
                last_return = rewards[t] + self.config.gamma * last_return
                td_error = rewards[t] + self.config.gamma * next_value - values[t]
                last_advantage = td_error + self.config.gamma * self.config.gae_lambda * last_advantage

            returns[t] = last_return
            advantages[t] = last_advantage

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        return {
            'states': states,
            'actions': actions,
            'returns': returns,
            'advantages': advantages,
            'log_probs': log_probs
        }

    def update(self, experience: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            experience: Dictionary of experience

        Returns:
            Dictionary of training metrics
        """
        states = experience['states']
        actions = experience['actions']
        returns = experience['returns']
        advantages = experience['advantages']
        old_log_probs = experience['log_probs']

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # PPO epochs
        for _ in range(self.config.n_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Get current policy outputs
                _, log_probs, entropy, values = self.network.get_action_and_value(
                    batch_states, batch_actions
                )

                # Policy loss (PPO objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                        1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                       self.config.value_coef * value_loss +
                       self.config.entropy_coef * entropy_loss)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
        }

    def train(self, n_iterations: int, save_interval: int = 10) -> List[Dict[str, float]]:
        """
        Train the agent.

        Args:
            n_iterations: Number of training iterations
            save_interval: How often to save checkpoints

        Returns:
            List of training metrics
        """
        training_history = []

        for iteration in range(1, n_iterations + 1):
            # Collect experience
            experience = self.collect_experience(self.config.n_steps)

            # Update policy
            metrics = self.update(experience)
            metrics['iteration'] = iteration
            training_history.append(metrics)

            # Log progress
            if iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}: "
                    f"Reward: {metrics['mean_episode_reward']:.2f}, "
                    f"Policy Loss: {metrics['policy_loss']:.4f}, "
                    f"Value Loss: {metrics['value_loss']:.4f}"
                )

            # Save checkpoint
            if iteration % save_interval == 0:
                self.save(f"checkpoints/ppo_agent_iter_{iteration}.pt")

        return training_history

    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths)
        }, path)
        logger.info(f"Saved agent to {path}")

    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint.get('config', self.config)
        logger.info(f"Loaded agent from {path}")

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        eval_rewards = []
        eval_profits = []
        eval_trades = []
        eval_sharpe_ratios = []

        for _ in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    if self.discrete:
                        action_logits, _ = self.network(state_tensor)
                        action = torch.argmax(action_logits, dim=1).cpu().numpy()[0]
                    else:
                        action_mean, _ = self.network(state_tensor)
                        action = action_mean.cpu().numpy()[0]

                state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            # Calculate metrics
            final_value = info['portfolio_value']
            profit = (final_value - self.config.initial_balance) / self.config.initial_balance
            n_trades = info['n_trades']

            # Calculate Sharpe ratio
            if len(self.env.portfolio_values) > 1:
                returns = np.diff(self.env.portfolio_values) / self.env.portfolio_values[:-1]
                if returns.std() > 0:
                    sharpe = np.sqrt(252) * returns.mean() / returns.std()
                else:
                    sharpe = 0
            else:
                sharpe = 0

            eval_rewards.append(episode_reward)
            eval_profits.append(profit)
            eval_trades.append(n_trades)
            eval_sharpe_ratios.append(sharpe)

        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_profit': np.mean(eval_profits),
            'mean_sharpe': np.mean(eval_sharpe_ratios),
            'mean_trades': np.mean(eval_trades)
        }


def create_trading_agent(data: pd.DataFrame, features: pd.DataFrame,
                        config: Optional[TradingConfig] = None) -> PPOAgent:
    """
    Create and initialize a trading agent.

    Args:
        data: OHLCV data
        features: Engineered features
        config: Trading configuration

    Returns:
        Initialized PPO agent
    """
    # Create environment
    env = TradingEnvironment(data, features, config)

    # Create agent
    agent = PPOAgent(env, config)

    return agent


if __name__ == "__main__":
    # Example usage
    logger.info("Creating example trading environment...")

    # Create synthetic data for testing
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    n_days = len(dates)

    # Generate price data
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.02, 0, n_days)),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 1e7, n_days)
    }, index=dates)

    # Create random features for testing
    features = pd.DataFrame(
        np.random.randn(n_days, 20),
        index=dates,
        columns=[f'feature_{i}' for i in range(20)]
    )

    # Create agent
    config = TradingConfig()
    agent = create_trading_agent(data, features, config)

    logger.info("Agent created successfully!")
    logger.info(f"State dimension: {agent.state_dim}")
    logger.info(f"Action space: {agent.n_actions} actions")

    # Quick test
    logger.info("\nTesting agent with random episode...")
    state, _ = agent.env.reset()

    for step in range(10):
        action = agent.env.action_space.sample()
        state, reward, terminated, truncated, info = agent.env.step(action)
        logger.info(f"Step {step}: Reward={reward:.2f}, Portfolio={info['portfolio_value']:.2f}")

        if terminated or truncated:
            break

    logger.info("Test completed!")