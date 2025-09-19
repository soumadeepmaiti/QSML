import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class HedgeType(Enum):
    """Types of hedging strategies."""
    DELTA = "delta"
    DELTA_GAMMA = "delta_gamma"
    DELTA_VEGA = "delta_vega"
    DELTA_GAMMA_VEGA = "delta_gamma_vega"
    STATIC = "static"  # Buy and hold


class RebalanceFrequency(Enum):
    """Rebalancing frequency options."""
    CONTINUOUS = "continuous"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class TransactionCosts:
    """Transaction cost model parameters."""
    
    # Fixed costs per trade
    fixed_cost_per_trade: float = 0.0  # $ per trade
    
    # Proportional costs
    stock_cost_bps: float = 5.0        # Basis points for stock trades
    option_cost_bps: float = 10.0      # Basis points for option trades
    
    # Bid-ask spread costs
    stock_spread_bps: float = 1.0      # Half-spread in basis points
    option_spread_bps: float = 5.0     # Half-spread in basis points
    
    # Market impact model (square-root impact)
    impact_coefficient: float = 0.0    # Impact coefficient
    impact_exponent: float = 0.5       # Impact exponent (0.5 = square root)
    
    # Borrowing costs
    stock_borrow_rate: float = 0.0     # Annual rate for short stock positions
    
    def compute_trading_cost(
        self,
        trade_value: float,
        trade_volume: float,
        is_stock: bool = True,
        daily_volume: float = 1e6
    ) -> float:
        """
        Compute total trading cost for a trade.
        
        Args:
            trade_value: Dollar value of trade
            trade_volume: Number of shares/contracts traded
            is_stock: True for stock, False for options
            daily_volume: Average daily volume for market impact
            
        Returns:
            Total trading cost in dollars
        """
        total_cost = 0.0
        
        # Fixed cost
        if trade_volume != 0:
            total_cost += self.fixed_cost_per_trade
        
        # Proportional cost
        cost_bps = self.stock_cost_bps if is_stock else self.option_cost_bps
        total_cost += abs(trade_value) * cost_bps / 10000
        
        # Bid-ask spread cost
        spread_bps = self.stock_spread_bps if is_stock else self.option_spread_bps
        total_cost += abs(trade_value) * spread_bps / 10000
        
        # Market impact cost
        if daily_volume > 0 and self.impact_coefficient > 0:
            participation_rate = abs(trade_volume) / daily_volume
            impact = self.impact_coefficient * (participation_rate ** self.impact_exponent)
            total_cost += abs(trade_value) * impact
        
        return total_cost


@dataclass
class HedgingConfig:
    """Configuration for hedging simulation."""
    
    # Strategy parameters
    hedge_type: HedgeType = HedgeType.DELTA
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    custom_rebalance_times: Optional[List[float]] = None  # Custom times in years
    
    # Rebalancing triggers
    delta_threshold: float = 0.1        # Rebalance if |delta| changes by this amount
    gamma_threshold: float = 0.05       # Rebalance if |gamma| changes by this amount
    vega_threshold: float = 0.02        # Rebalance if |vega| changes by this amount
    time_threshold: float = 1.0         # Rebalance at least every N days
    
    # Position limits
    max_stock_position: float = 1e6     # Maximum stock position in dollars
    max_option_position: int = 1000     # Maximum number of option contracts
    
    # Risk management
    stop_loss_level: Optional[float] = None      # Stop loss as fraction of premium
    profit_target_level: Optional[float] = None # Profit target as fraction of premium
    
    # Market data
    stock_volatility: float = 0.2       # Realized stock volatility for simulation
    correlation_with_model: float = 1.0 # Correlation between realized and model vol
    
    # Transaction costs
    transaction_costs: TransactionCosts = field(default_factory=TransactionCosts)
    
    # Simulation parameters
    n_paths: int = 1000                 # Number of Monte Carlo paths
    seed: Optional[int] = 42            # Random seed for reproducibility


@dataclass
class Position:
    """Represents a trading position."""
    
    # Option position
    option_contracts: float = 0.0      # Number of option contracts (positive = long)
    option_price: float = 0.0          # Current option price per contract
    
    # Stock hedge position  
    stock_shares: float = 0.0          # Number of stock shares (positive = long)
    stock_price: float = 0.0           # Current stock price per share
    
    # Cash position
    cash: float = 0.0                  # Cash balance
    
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    
    def net_delta(self) -> float:
        """Calculate net portfolio delta."""
        return self.option_contracts * self.delta + self.stock_shares
    
    def portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        option_value = self.option_contracts * self.option_price * 100  # Options are per 100 shares
        stock_value = self.stock_shares * self.stock_price
        return option_value + stock_value + self.cash
    
    def to_dict(self) -> Dict[str, float]:
        """Convert position to dictionary."""
        return {
            'option_contracts': self.option_contracts,
            'option_price': self.option_price,
            'stock_shares': self.stock_shares,
            'stock_price': self.stock_price,
            'cash': self.cash,
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'net_delta': self.net_delta(),
            'portfolio_value': self.portfolio_value()
        }


@dataclass
class Trade:
    """Represents a single trade."""
    
    timestamp: float                    # Time of trade (in years from start)
    instrument: str                     # "stock" or "option"
    quantity: float                     # Quantity traded (positive = buy, negative = sell)
    price: float                        # Price per unit
    trade_value: float                  # Total trade value
    transaction_cost: float             # Transaction cost
    reason: str                         # Reason for trade
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'timestamp': self.timestamp,
            'instrument': self.instrument,
            'quantity': self.quantity,
            'price': self.price,
            'trade_value': self.trade_value,
            'transaction_cost': self.transaction_cost,
            'reason': self.reason
        }


class PathSimulator:
    """Simulate stock price paths for hedging analysis."""
    
    def __init__(self, config: HedgingConfig):
        """Initialize path simulator."""
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    def simulate_stock_paths(
        self,
        S0: float,
        r: float,
        q: float,
        T: float,
        time_steps: np.ndarray
    ) -> np.ndarray:
        """
        Simulate stock price paths using geometric Brownian motion.
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            q: Dividend yield
            T: Time to expiration
            time_steps: Array of time points
            
        Returns:
            Array of shape (n_paths, n_time_steps) with stock prices
        """
        n_steps = len(time_steps)
        dt = np.diff(np.concatenate([[0], time_steps]))
        
        # Generate random shocks
        dW = self.rng.normal(0, 1, (self.config.n_paths, n_steps))
        
        # Initialize paths
        paths = np.zeros((self.config.n_paths, n_steps))
        paths[:, 0] = S0
        
        # Simulate paths
        sigma = self.config.stock_volatility
        
        for i in range(1, n_steps):
            # Geometric Brownian motion
            drift = (r - q - 0.5 * sigma**2) * dt[i]
            diffusion = sigma * np.sqrt(dt[i]) * dW[:, i]
            
            paths[:, i] = paths[:, i-1] * np.exp(drift + diffusion)
        
        return paths
    
    def simulate_volatility_paths(
        self,
        v0: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        time_steps: np.ndarray,
        correlation: float = 0.0
    ) -> np.ndarray:
        """
        Simulate stochastic volatility paths using Heston model.
        
        Args:
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma_v: Volatility of volatility
            time_steps: Array of time points
            correlation: Correlation with stock returns
            
        Returns:
            Array of shape (n_paths, n_time_steps) with variance paths
        """
        n_steps = len(time_steps)
        dt = np.diff(np.concatenate([[0], time_steps]))
        
        # Generate correlated random shocks
        Z1 = self.rng.normal(0, 1, (self.config.n_paths, n_steps))
        Z2 = self.rng.normal(0, 1, (self.config.n_paths, n_steps))
        
        # Apply correlation
        W_v = correlation * Z1 + np.sqrt(1 - correlation**2) * Z2
        
        # Initialize variance paths
        var_paths = np.zeros((self.config.n_paths, n_steps))
        var_paths[:, 0] = v0
        
        # Simulate variance using Euler scheme with absorption at zero
        for i in range(1, n_steps):
            dv = kappa * (theta - np.maximum(var_paths[:, i-1], 0)) * dt[i]
            dv += sigma_v * np.sqrt(np.maximum(var_paths[:, i-1], 0)) * np.sqrt(dt[i]) * W_v[:, i]
            
            var_paths[:, i] = np.maximum(var_paths[:, i-1] + dv, 0)
        
        return var_paths


class GreeksCalculator:
    """Calculate option Greeks for hedging."""
    
    def __init__(self, pricing_function: Callable):
        """
        Initialize Greeks calculator.
        
        Args:
            pricing_function: Function that prices options given (S, K, r, q, T, params)
        """
        self.pricing_function = pricing_function
    
    def calculate_greeks(
        self,
        S: float,
        K: float,
        r: float,
        q: float,
        T: float,
        params: Dict[str, float],
        option_type: str = "call"
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences.
        
        Args:
            S: Stock price
            K: Strike price
            r: Risk-free rate
            q: Dividend yield
            T: Time to expiration
            params: Model parameters
            option_type: "call" or "put"
            
        Returns:
            Dictionary with Greeks
        """
        # Base price
        base_price = self.pricing_function(S, K, r, q, T, params, option_type)
        
        # Delta (sensitivity to stock price)
        h_s = 0.01 * S
        price_up = self.pricing_function(S + h_s, K, r, q, T, params, option_type)
        price_down = self.pricing_function(S - h_s, K, r, q, T, params, option_type)
        delta = (price_up - price_down) / (2 * h_s)
        
        # Gamma (second derivative w.r.t. stock price)
        gamma = (price_up - 2 * base_price + price_down) / (h_s**2)
        
        # Vega (sensitivity to volatility)
        if 'sigma' in params:
            h_v = 0.01
            params_up = params.copy()
            params_up['sigma'] += h_v
            params_down = params.copy()
            params_down['sigma'] -= h_v
            
            price_v_up = self.pricing_function(S, K, r, q, T, params_up, option_type)
            price_v_down = self.pricing_function(S, K, r, q, T, params_down, option_type)
            vega = (price_v_up - price_v_down) / (2 * h_v)
        else:
            vega = 0.0
        
        # Theta (sensitivity to time)
        h_t = 1/365  # One day
        if T > h_t:
            price_t = self.pricing_function(S, K, r, q, T - h_t, params, option_type)
            theta = -(price_t - base_price) / h_t
        else:
            theta = 0.0
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }


class HedgingSimulator:
    """Main hedging simulation engine."""
    
    def __init__(
        self,
        config: HedgingConfig,
        pricing_function: Callable,
        greeks_calculator: Optional[GreeksCalculator] = None
    ):
        """
        Initialize hedging simulator.
        
        Args:
            config: Hedging configuration
            pricing_function: Function to price options
            greeks_calculator: Greeks calculator (optional, will create if None)
        """
        self.config = config
        self.pricing_function = pricing_function
        self.greeks_calculator = greeks_calculator or GreeksCalculator(pricing_function)
        self.path_simulator = PathSimulator(config)
        
        self.logger = logger
    
    def should_rebalance(
        self,
        current_time: float,
        last_rebalance_time: float,
        current_greeks: Dict[str, float],
        last_greeks: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            current_time: Current time
            last_rebalance_time: Time of last rebalance
            current_greeks: Current option Greeks
            last_greeks: Greeks at last rebalance
            
        Returns:
            Tuple of (should_rebalance, reason)
        """
        # Time-based rebalancing
        time_since_rebalance = current_time - last_rebalance_time
        if time_since_rebalance >= self.config.time_threshold / 365:
            return True, "time_threshold"
        
        # Delta threshold
        delta_change = abs(current_greeks['delta'] - last_greeks['delta'])
        if delta_change >= self.config.delta_threshold:
            return True, f"delta_change_{delta_change:.3f}"
        
        # Gamma threshold (if using delta-gamma hedging)
        if self.config.hedge_type in [HedgeType.DELTA_GAMMA, HedgeType.DELTA_GAMMA_VEGA]:
            gamma_change = abs(current_greeks['gamma'] - last_greeks['gamma'])
            if gamma_change >= self.config.gamma_threshold:
                return True, f"gamma_change_{gamma_change:.3f}"
        
        # Vega threshold (if using vega hedging)
        if self.config.hedge_type in [HedgeType.DELTA_VEGA, HedgeType.DELTA_GAMMA_VEGA]:
            vega_change = abs(current_greeks['vega'] - last_greeks['vega'])
            if vega_change >= self.config.vega_threshold:
                return True, f"vega_change_{vega_change:.3f}"
        
        return False, "no_trigger"
    
    def calculate_hedge_trades(
        self,
        current_position: Position,
        target_greeks: Dict[str, float],
        current_stock_price: float
    ) -> List[Trade]:
        """
        Calculate trades needed to achieve target Greeks.
        
        Args:
            current_position: Current portfolio position
            target_greeks: Target Greeks (usually zero)
            current_stock_price: Current stock price
            
        Returns:
            List of trades to execute
        """
        trades = []
        timestamp = 0.0  # Will be set by caller
        
        if self.config.hedge_type == HedgeType.DELTA:
            # Delta hedging: adjust stock position to neutralize delta
            current_net_delta = current_position.net_delta()
            target_net_delta = target_greeks.get('delta', 0.0)
            delta_to_hedge = current_net_delta - target_net_delta
            
            if abs(delta_to_hedge) > 1e-6:  # Avoid tiny trades
                stock_trade_quantity = -delta_to_hedge
                trade_value = stock_trade_quantity * current_stock_price
                transaction_cost = self.config.transaction_costs.compute_trading_cost(
                    trade_value, abs(stock_trade_quantity), is_stock=True
                )
                
                trade = Trade(
                    timestamp=timestamp,
                    instrument="stock",
                    quantity=stock_trade_quantity,
                    price=current_stock_price,
                    trade_value=trade_value,
                    transaction_cost=transaction_cost,
                    reason="delta_hedge"
                )
                trades.append(trade)
        
        elif self.config.hedge_type == HedgeType.STATIC:
            # Static hedge: no trades after initial setup
            pass
        
        # TODO: Implement delta-gamma, delta-vega, and delta-gamma-vega hedging
        # These would require additional instruments (other options) to hedge
        
        return trades
    
    def execute_trades(
        self,
        position: Position,
        trades: List[Trade],
        current_time: float
    ) -> float:
        """
        Execute trades and update position.
        
        Args:
            position: Current position (modified in place)
            trades: List of trades to execute
            current_time: Current time
            
        Returns:
            Total transaction costs
        """
        total_costs = 0.0
        
        for trade in trades:
            trade.timestamp = current_time
            
            if trade.instrument == "stock":
                position.stock_shares += trade.quantity
                position.cash -= trade.trade_value + trade.transaction_cost
                
            elif trade.instrument == "option":
                position.option_contracts += trade.quantity
                position.cash -= trade.trade_value + trade.transaction_cost
            
            total_costs += trade.transaction_cost
        
        return total_costs
    
    def simulate_single_path(
        self,
        path_idx: int,
        S0: float,
        K: float,
        r: float,
        q: float,
        T: float,
        params: Dict[str, float],
        option_type: str = "call",
        stock_path: Optional[np.ndarray] = None,
        time_steps: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Simulate hedging for a single price path.
        
        Args:
            path_idx: Path index for identification
            S0: Initial stock price
            K: Strike price
            r: Risk-free rate
            q: Dividend yield
            T: Time to expiration
            params: Model parameters
            option_type: "call" or "put"
            stock_path: Pre-generated stock price path (optional)
            time_steps: Time steps for simulation (optional)
            
        Returns:
            Dictionary with simulation results
        """
        # Setup time steps
        if time_steps is None:
            if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
                n_steps = int(T * 365) + 1
                time_steps = np.linspace(0, T, n_steps)
            else:
                # Default to weekly rebalancing
                n_steps = int(T * 52) + 1
                time_steps = np.linspace(0, T, n_steps)
        
        # Generate stock path if not provided
        if stock_path is None:
            stock_paths = self.path_simulator.simulate_stock_paths(S0, r, q, T, time_steps)
            stock_path = stock_paths[0]  # Use first path
        
        # Initialize tracking
        positions = []
        trades = []
        pnl_history = []
        
        # Initial position setup
        initial_greeks = self.greeks_calculator.calculate_greeks(
            S0, K, r, q, T, params, option_type
        )
        
        position = Position(
            option_contracts=1.0,  # Start with 1 option contract (short)
            option_price=initial_greeks['price'],
            stock_shares=0.0,
            stock_price=S0,
            cash=initial_greeks['price'] * 100,  # Receive premium
            delta=initial_greeks['delta'],
            gamma=initial_greeks['gamma'],
            vega=initial_greeks['vega'],
            theta=initial_greeks['theta']
        )
        
        # Initial hedge
        initial_trades = self.calculate_hedge_trades(
            position, {'delta': 0.0}, S0
        )
        
        total_costs = self.execute_trades(position, initial_trades, 0.0)
        
        trades.extend(initial_trades)
        positions.append(position.to_dict())
        pnl_history.append(position.portfolio_value())
        
        last_rebalance_time = 0.0
        last_greeks = initial_greeks.copy()
        
        # Simulate through time
        for i, (t, S_t) in enumerate(zip(time_steps[1:], stock_path[1:]), 1):
            time_to_expiry = T - t
            
            if time_to_expiry <= 0:
                # Option has expired
                if option_type == "call":
                    option_value = max(S_t - K, 0)
                else:
                    option_value = max(K - S_t, 0)
                
                position.option_price = option_value
                position.stock_price = S_t
                position.delta = 0.0
                position.gamma = 0.0
                position.vega = 0.0
                position.theta = 0.0
            else:
                # Calculate current Greeks
                current_greeks = self.greeks_calculator.calculate_greeks(
                    S_t, K, r, q, time_to_expiry, params, option_type
                )
                
                position.option_price = current_greeks['price']
                position.stock_price = S_t
                position.delta = current_greeks['delta']
                position.gamma = current_greeks['gamma']
                position.vega = current_greeks['vega']
                position.theta = current_greeks['theta']
                
                # Check if rebalancing is needed
                should_rebal, reason = self.should_rebalance(
                    t, last_rebalance_time, current_greeks, last_greeks
                )
                
                if should_rebal and time_to_expiry > 1/365:  # Don't rebalance in last day
                    # Calculate and execute hedge trades
                    hedge_trades = self.calculate_hedge_trades(
                        position, {'delta': 0.0}, S_t
                    )
                    
                    if hedge_trades:
                        costs = self.execute_trades(position, hedge_trades, t)
                        total_costs += costs
                        trades.extend(hedge_trades)
                        
                        last_rebalance_time = t
                        last_greeks = current_greeks.copy()
            
            positions.append(position.to_dict())
            pnl_history.append(position.portfolio_value())
        
        # Final P&L calculation
        final_pnl = position.portfolio_value()
        initial_value = initial_greeks['price'] * 100  # Initial premium received
        
        return {
            'path_idx': path_idx,
            'final_pnl': final_pnl,
            'total_costs': total_costs,
            'n_trades': len(trades),
            'initial_value': initial_value,
            'net_pnl': final_pnl - initial_value,
            'positions': positions,
            'trades': [trade.to_dict() for trade in trades],
            'pnl_history': pnl_history,
            'time_steps': time_steps.tolist(),
            'stock_path': stock_path.tolist()
        }
    
    def run_simulation(
        self,
        S0: float,
        K: float,
        r: float,
        q: float,
        T: float,
        params: Dict[str, float],
        option_type: str = "call"
    ) -> Dict[str, Any]:
        """
        Run complete hedging simulation across multiple paths.
        
        Args:
            S0: Initial stock price
            K: Strike price
            r: Risk-free rate
            q: Dividend yield
            T: Time to expiration
            params: Model parameters
            option_type: "call" or "put"
            
        Returns:
            Dictionary with complete simulation results
        """
        self.logger.info(f"Running hedging simulation with {self.config.n_paths} paths")
        
        # Setup time steps
        if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            n_steps = int(T * 365) + 1
        elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            n_steps = int(T * 52) + 1
        else:
            n_steps = 101  # Default
        
        time_steps = np.linspace(0, T, n_steps)
        
        # Generate all stock paths
        stock_paths = self.path_simulator.simulate_stock_paths(S0, r, q, T, time_steps)
        
        # Run simulation for each path
        path_results = []
        
        for path_idx in range(self.config.n_paths):
            if path_idx % 100 == 0:
                self.logger.info(f"Processing path {path_idx}/{self.config.n_paths}")
            
            try:
                result = self.simulate_single_path(
                    path_idx=path_idx,
                    S0=S0,
                    K=K,
                    r=r,
                    q=q,
                    T=T,
                    params=params,
                    option_type=option_type,
                    stock_path=stock_paths[path_idx],
                    time_steps=time_steps
                )
                path_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Path {path_idx} failed: {e}")
                continue
        
        # Aggregate results
        aggregated_results = self._aggregate_path_results(path_results)
        
        # Add simulation metadata
        aggregated_results.update({
            'config': self.config.__dict__,
            'simulation_params': {
                'S0': S0, 'K': K, 'r': r, 'q': q, 'T': T,
                'params': params, 'option_type': option_type
            },
            'n_successful_paths': len(path_results),
            'time_steps': time_steps.tolist()
        })
        
        self.logger.info("Hedging simulation completed")
        return aggregated_results
    
    def _aggregate_path_results(self, path_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all paths."""
        if not path_results:
            return {}
        
        # Extract key metrics
        final_pnls = [r['final_pnl'] for r in path_results]
        net_pnls = [r['net_pnl'] for r in path_results]
        total_costs = [r['total_costs'] for r in path_results]
        n_trades = [r['n_trades'] for r in path_results]
        
        # Statistical summary
        summary_stats = {
            'mean_final_pnl': np.mean(final_pnls),
            'std_final_pnl': np.std(final_pnls),
            'mean_net_pnl': np.mean(net_pnls),
            'std_net_pnl': np.std(net_pnls),
            'mean_total_costs': np.mean(total_costs),
            'mean_n_trades': np.mean(n_trades),
            'pnl_percentiles': {
                'p5': np.percentile(net_pnls, 5),
                'p25': np.percentile(net_pnls, 25),
                'p50': np.percentile(net_pnls, 50),
                'p75': np.percentile(net_pnls, 75),
                'p95': np.percentile(net_pnls, 95)
            }
        }
        
        return {
            'summary_stats': summary_stats,
            'path_results': path_results
        }