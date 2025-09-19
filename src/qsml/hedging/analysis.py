import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from .simulation import HedgingSimulator, HedgingConfig, HedgeType, RebalanceFrequency

logger = logging.getLogger(__name__)


class HedgingAnalyzer:
    """Analyze and visualize hedging simulation results."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.logger = logger
    
    def analyze_pnl_distribution(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze P&L distribution from hedging simulation.
        
        Args:
            results: Results from HedgingSimulator.run_simulation()
            
        Returns:
            Dictionary with P&L analysis
        """
        if 'path_results' not in results:
            return {}
        
        path_results = results['path_results']
        net_pnls = [r['net_pnl'] for r in path_results]
        
        # Basic statistics
        pnl_stats = {
            'mean': np.mean(net_pnls),
            'std': np.std(net_pnls),
            'min': np.min(net_pnls),
            'max': np.max(net_pnls),
            'median': np.median(net_pnls),
            'skewness': self._calculate_skewness(net_pnls),
            'kurtosis': self._calculate_kurtosis(net_pnls)
        }
        
        # Risk metrics
        risk_metrics = {
            'var_95': np.percentile(net_pnls, 5),   # 95% VaR
            'var_99': np.percentile(net_pnls, 1),   # 99% VaR
            'cvar_95': np.mean([pnl for pnl in net_pnls if pnl <= np.percentile(net_pnls, 5)]),
            'cvar_99': np.mean([pnl for pnl in net_pnls if pnl <= np.percentile(net_pnls, 1)]),
            'probability_of_loss': np.mean([1 for pnl in net_pnls if pnl < 0]),
            'max_drawdown': np.min(net_pnls)
        }
        
        # Performance metrics
        if pnl_stats['std'] > 0:
            sharpe_ratio = pnl_stats['mean'] / pnl_stats['std']
            sortino_ratio = pnl_stats['mean'] / np.std([pnl for pnl in net_pnls if pnl < 0]) if any(pnl < 0 for pnl in net_pnls) else np.inf
        else:
            sharpe_ratio = np.inf if pnl_stats['mean'] > 0 else 0
            sortino_ratio = sharpe_ratio
        
        performance_metrics = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': sum([pnl for pnl in net_pnls if pnl > 0]) / abs(sum([pnl for pnl in net_pnls if pnl < 0])) if any(pnl < 0 for pnl in net_pnls) else np.inf
        }
        
        return {
            'pnl_statistics': pnl_stats,
            'risk_metrics': risk_metrics,
            'performance_metrics': performance_metrics,
            'raw_pnls': net_pnls
        }
    
    def analyze_trading_activity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trading activity from hedging simulation.
        
        Args:
            results: Results from HedgingSimulator.run_simulation()
            
        Returns:
            Dictionary with trading analysis
        """
        if 'path_results' not in results:
            return {}
        
        path_results = results['path_results']
        
        # Aggregate trading statistics
        total_trades = [r['n_trades'] for r in path_results]
        total_costs = [r['total_costs'] for r in path_results]
        
        # Transaction cost analysis
        cost_stats = {
            'mean_total_cost': np.mean(total_costs),
            'std_total_cost': np.std(total_costs),
            'mean_trades_per_path': np.mean(total_trades),
            'std_trades_per_path': np.std(total_trades),
            'cost_per_trade': np.mean(total_costs) / np.mean(total_trades) if np.mean(total_trades) > 0 else 0
        }
        
        # Trading frequency analysis
        frequency_stats = {}
        if len(path_results) > 0 and 'trades' in path_results[0]:
            all_trades = []
            for result in path_results:
                all_trades.extend(result['trades'])
            
            if all_trades:
                trade_df = pd.DataFrame(all_trades)
                
                # Trade timing analysis
                trade_times = trade_df['timestamp'].values
                time_intervals = np.diff(sorted(trade_times))
                
                frequency_stats = {
                    'mean_trade_interval': np.mean(time_intervals) if len(time_intervals) > 0 else 0,
                    'std_trade_interval': np.std(time_intervals) if len(time_intervals) > 0 else 0,
                    'trades_by_reason': trade_df['reason'].value_counts().to_dict() if 'reason' in trade_df.columns else {}
                }
        
        return {
            'cost_statistics': cost_stats,
            'frequency_statistics': frequency_stats
        }
    
    def analyze_hedging_effectiveness(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze effectiveness of hedging strategy.
        
        Args:
            results: Results from HedgingSimulator.run_simulation()
            
        Returns:
            Dictionary with hedging effectiveness metrics
        """
        if 'path_results' not in results:
            return {}
        
        path_results = results['path_results']
        
        # Calculate hedge effectiveness metrics
        effectiveness_metrics = {}
        
        # For each path, analyze how well delta was hedged
        delta_tracking_errors = []
        
        for result in path_results:
            if 'positions' not in result:
                continue
            
            positions = result['positions']
            
            # Extract net delta over time
            net_deltas = [pos.get('net_delta', 0) for pos in positions]
            
            # Calculate tracking error (RMS of net delta)
            tracking_error = np.sqrt(np.mean(np.array(net_deltas)**2))
            delta_tracking_errors.append(tracking_error)
        
        if delta_tracking_errors:
            effectiveness_metrics = {
                'mean_delta_tracking_error': np.mean(delta_tracking_errors),
                'std_delta_tracking_error': np.std(delta_tracking_errors),
                'max_delta_tracking_error': np.max(delta_tracking_errors),
                'hedge_efficiency': 1 - np.mean(delta_tracking_errors)  # Simple efficiency metric
            }
        
        return effectiveness_metrics
    
    def compare_strategies(
        self,
        strategy_results: Dict[str, Dict[str, Any]],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple hedging strategies.
        
        Args:
            strategy_results: Dictionary of strategy_name -> results
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with strategy comparison
        """
        if metrics is None:
            metrics = [
                'mean_net_pnl', 'std_net_pnl', 'var_95', 'cvar_95',
                'sharpe_ratio', 'mean_total_cost', 'mean_trades_per_path'
            ]
        
        comparison_data = []
        
        for strategy_name, results in strategy_results.items():
            # Analyze each strategy
            pnl_analysis = self.analyze_pnl_distribution(results)
            trading_analysis = self.analyze_trading_activity(results)
            
            row = {'strategy': strategy_name}
            
            # Extract metrics
            if 'pnl_statistics' in pnl_analysis:
                row.update({
                    'mean_net_pnl': pnl_analysis['pnl_statistics']['mean'],
                    'std_net_pnl': pnl_analysis['pnl_statistics']['std'],
                    'median_net_pnl': pnl_analysis['pnl_statistics']['median']
                })
            
            if 'risk_metrics' in pnl_analysis:
                row.update({
                    'var_95': pnl_analysis['risk_metrics']['var_95'],
                    'cvar_95': pnl_analysis['risk_metrics']['cvar_95'],
                    'probability_of_loss': pnl_analysis['risk_metrics']['probability_of_loss']
                })
            
            if 'performance_metrics' in pnl_analysis:
                row.update({
                    'sharpe_ratio': pnl_analysis['performance_metrics']['sharpe_ratio'],
                    'sortino_ratio': pnl_analysis['performance_metrics']['sortino_ratio']
                })
            
            if 'cost_statistics' in trading_analysis:
                row.update({
                    'mean_total_cost': trading_analysis['cost_statistics']['mean_total_cost'],
                    'mean_trades_per_path': trading_analysis['cost_statistics']['mean_trades_per_path']
                })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Filter to requested metrics
        available_metrics = [m for m in metrics if m in df.columns]
        if available_metrics:
            df = df[['strategy'] + available_metrics]
        
        return df
    
    def plot_pnl_distribution(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        title: str = "P&L Distribution"
    ):
        """Plot P&L distribution from hedging simulation."""
        
        pnl_analysis = self.analyze_pnl_distribution(results)
        
        if 'raw_pnls' not in pnl_analysis:
            self.logger.warning("No P&L data available for plotting")
            return
        
        net_pnls = pnl_analysis['raw_pnls']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram
        axes[0, 0].hist(net_pnls, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Break-even')
        axes[0, 0].axvline(np.mean(net_pnls), color='green', linestyle='-', label=f'Mean: {np.mean(net_pnls):.2f}')
        axes[0, 0].set_xlabel('Net P&L')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('P&L Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_pnls = np.sort(net_pnls)
        cumulative = np.arange(1, len(sorted_pnls) + 1) / len(sorted_pnls)
        axes[0, 1].plot(sorted_pnls, cumulative)
        axes[0, 1].axvline(0, color='red', linestyle='--', label='Break-even')
        axes[0, 1].axvline(np.percentile(net_pnls, 5), color='orange', linestyle=':', label='5% VaR')
        axes[0, 1].set_xlabel('Net P&L')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Cumulative P&L Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot (normal)
        from scipy import stats
        stats.probplot(net_pnls, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot vs Normal Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, 1].boxplot(net_pnls, vert=True)
        axes[1, 1].set_ylabel('Net P&L')
        axes[1, 1].set_title('P&L Box Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hedging_paths(
        self,
        results: Dict[str, Any],
        n_paths: int = 10,
        save_path: Optional[str] = None,
        title: str = "Sample Hedging Paths"
    ):
        """Plot sample hedging paths."""
        
        if 'path_results' not in results:
            self.logger.warning("No path data available for plotting")
            return
        
        path_results = results['path_results']
        n_to_plot = min(n_paths, len(path_results))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i in range(n_to_plot):
            result = path_results[i]
            
            if 'time_steps' not in result or 'stock_path' not in result:
                continue
            
            time_steps = result['time_steps']
            stock_path = result['stock_path']
            pnl_history = result.get('pnl_history', [])
            
            # Stock price paths
            axes[0, 0].plot(time_steps, stock_path, alpha=0.6, linewidth=1)
            
            # P&L evolution
            if pnl_history:
                axes[0, 1].plot(time_steps[:len(pnl_history)], pnl_history, alpha=0.6, linewidth=1)
            
            # Net delta over time (if available)
            if 'positions' in result:
                positions = result['positions']
                net_deltas = [pos.get('net_delta', 0) for pos in positions]
                axes[1, 0].plot(time_steps[:len(net_deltas)], net_deltas, alpha=0.6, linewidth=1)
            
            # Portfolio value over time
            if 'positions' in result:
                portfolio_values = [pos.get('portfolio_value', 0) for pos in positions]
                axes[1, 1].plot(time_steps[:len(portfolio_values)], portfolio_values, alpha=0.6, linewidth=1)
        
        # Configure plots
        axes[0, 0].set_xlabel('Time (years)')
        axes[0, 0].set_ylabel('Stock Price')
        axes[0, 0].set_title('Stock Price Paths')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Time (years)')
        axes[0, 1].set_ylabel('P&L')
        axes[0, 1].set_title('P&L Evolution')
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Time (years)')
        axes[1, 0].set_ylabel('Net Delta')
        axes[1, 0].set_title('Delta Hedging Effectiveness')
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Time (years)')
        axes[1, 1].set_ylabel('Portfolio Value')
        axes[1, 1].set_title('Portfolio Value Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_strategy_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = None,
        save_path: Optional[str] = None,
        title: str = "Strategy Comparison"
    ):
        """Plot comparison of different hedging strategies."""
        
        if metrics is None:
            # Select numeric columns excluding strategy name
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns.tolist()
            metrics = numeric_cols[:6]  # Limit to 6 metrics for readability
        
        # Filter metrics that exist in DataFrame
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            self.logger.warning("No valid metrics found for plotting")
            return
        
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric in enumerate(available_metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Bar plot for each metric
            strategies = comparison_df['strategy']
            values = comparison_df[metric]
            
            bars = ax.bar(strategies, values, alpha=0.7)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Color bars based on whether higher is better
            if 'cost' in metric.lower() or 'var' in metric.lower() or 'loss' in metric.lower():
                # Lower is better - color worst red
                worst_idx = values.idxmax()
                bars[worst_idx].set_color('red')
                best_idx = values.idxmin()
                bars[best_idx].set_color('green')
            else:
                # Higher is better - color best green
                best_idx = values.idxmax()
                bars[best_idx].set_color('green')
                worst_idx = values.idxmin()
                bars[worst_idx].set_color('red')
        
        # Hide empty subplots
        for i in range(len(available_metrics), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data."""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis


def generate_hedging_report(
    results: Dict[str, Any],
    save_dir: str,
    report_name: str = "hedging_analysis"
) -> Dict[str, Any]:
    """
    Generate comprehensive hedging analysis report.
    
    Args:
        results: Results from HedgingSimulator.run_simulation()
        save_dir: Directory to save report files
        report_name: Base name for report files
        
    Returns:
        Dictionary with complete analysis
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    analyzer = HedgingAnalyzer()
    
    # Perform all analyses
    pnl_analysis = analyzer.analyze_pnl_distribution(results)
    trading_analysis = analyzer.analyze_trading_activity(results)
    effectiveness_analysis = analyzer.analyze_hedging_effectiveness(results)
    
    # Generate plots
    analyzer.plot_pnl_distribution(
        results,
        save_path=str(save_path / f"{report_name}_pnl_distribution.png"),
        title="P&L Distribution Analysis"
    )
    
    analyzer.plot_hedging_paths(
        results,
        save_path=str(save_path / f"{report_name}_sample_paths.png"),
        title="Sample Hedging Paths"
    )
    
    # Compile complete report
    complete_report = {
        'pnl_analysis': pnl_analysis,
        'trading_analysis': trading_analysis,
        'effectiveness_analysis': effectiveness_analysis,
        'simulation_metadata': {
            'config': results.get('config', {}),
            'simulation_params': results.get('simulation_params', {}),
            'n_successful_paths': results.get('n_successful_paths', 0)
        }
    }
    
    # Save report as JSON
    import json
    with open(save_path / f"{report_name}_report.json", 'w') as f:
        json.dump(complete_report, f, indent=2, default=str)
    
    logger.info(f"Hedging analysis report saved to: {save_path}")
    
    return complete_report