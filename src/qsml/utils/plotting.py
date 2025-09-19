import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Set default plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def setup_matplotlib_style():
    """Setup matplotlib with publication-quality defaults."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_iv_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv_surface: np.ndarray,
    title: str = "Implied Volatility Surface",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot 3D implied volatility surface using Plotly.
    
    Args:
        strikes: Array of strike prices
        maturities: Array of maturities in years
        iv_surface: 2D array of implied volatilities (maturities x strikes)
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=[go.Surface(
        x=strikes,
        y=maturities,
        z=iv_surface,
        colorscale='Viridis',
        name='IV Surface'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='Maturity (Years)',
            zaxis_title='Implied Volatility',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=800,
        height=600
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"IV surface plot saved to {save_path}")
    
    return fig


def plot_iv_smile(
    moneyness: np.ndarray,
    iv: np.ndarray,
    market_iv: Optional[np.ndarray] = None,
    title: str = "Implied Volatility Smile",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot implied volatility smile.
    
    Args:
        moneyness: Log-moneyness values
        iv: Model implied volatilities
        market_iv: Optional market implied volatilities for comparison
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(moneyness, iv, 'b-', linewidth=2, label='Model IV')
    
    if market_iv is not None:
        ax.scatter(moneyness, market_iv, c='red', alpha=0.7, s=50, label='Market IV')
    
    ax.set_xlabel('Log-Moneyness')
    ax.set_ylabel('Implied Volatility')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"IV smile plot saved to {save_path}")
    
    return fig


def plot_heston_params_timeseries(
    params_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series of Heston parameters.
    
    Args:
        params_df: DataFrame with columns ['date', 'kappa', 'theta', 'sigma', 'rho', 'v0']
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0']
    param_labels = ['κ (mean reversion)', 'θ (long-term variance)', 
                   'σ (vol of vol)', 'ρ (correlation)', 'v₀ (initial variance)']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (param, label) in enumerate(zip(param_names, param_labels)):
        if param in params_df.columns:
            axes[i].plot(params_df['date'], params_df[param], linewidth=1.5)
            axes[i].set_title(label)
            axes[i].set_xlabel('Date')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heston parameters plot saved to {save_path}")
    
    return fig


def plot_hedging_pnl(
    pnl_df: pd.DataFrame,
    methods: list[str] = ['bs', 'heston', 'surrogate'],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot hedging P&L comparison across methods.
    
    Args:
        pnl_df: DataFrame with P&L data for different methods
        methods: List of methods to compare
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cumulative P&L
    for method in methods:
        if f'pnl_{method}' in pnl_df.columns:
            cumulative_pnl = pnl_df[f'pnl_{method}'].cumsum()
            ax1.plot(pnl_df.index, cumulative_pnl, label=method.upper(), linewidth=2)
    
    ax1.set_title('Cumulative Hedging P&L')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative P&L')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # P&L distributions
    pnl_data = []
    labels = []
    for method in methods:
        if f'pnl_{method}' in pnl_df.columns:
            pnl_data.append(pnl_df[f'pnl_{method}'])
            labels.append(method.upper())
    
    ax2.boxplot(pnl_data, labels=labels)
    ax2.set_title('P&L Distribution')
    ax2.set_ylabel('P&L')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Hedging P&L plot saved to {save_path}")
    
    return fig


def plot_surrogate_error_heatmap(
    moneyness_grid: np.ndarray,
    maturity_grid: np.ndarray,
    error_grid: np.ndarray,
    title: str = "Surrogate Pricing Error",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot surrogate model error as a heatmap.
    
    Args:
        moneyness_grid: 2D grid of log-moneyness values
        maturity_grid: 2D grid of maturity values
        error_grid: 2D grid of pricing errors
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.contourf(moneyness_grid, maturity_grid, error_grid, 
                     levels=20, cmap='RdYlBu_r')
    
    ax.set_xlabel('Log-Moneyness')
    ax.set_ylabel('Maturity (Years)')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Error (%)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Error heatmap saved to {save_path}")
    
    return fig


def plot_option_surface_comparison(
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_surface: np.ndarray,
    model_surface: np.ndarray,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot comparison of market vs model option surfaces.
    
    Args:
        strikes: Array of strike prices
        maturities: Array of maturities
        market_surface: Market option prices/IVs
        model_surface: Model option prices/IVs
        save_path: Optional path to save the plot
        
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Market Surface', 'Model Surface'],
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )
    
    # Market surface
    fig.add_trace(
        go.Surface(x=strikes, y=maturities, z=market_surface, 
                   colorscale='Viridis', name='Market'),
        row=1, col=1
    )
    
    # Model surface
    fig.add_trace(
        go.Surface(x=strikes, y=maturities, z=model_surface, 
                   colorscale='Plasma', name='Model'),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Market vs Model Surface Comparison",
        width=1200,
        height=600
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Surface comparison plot saved to {save_path}")
    
    return fig


def plot_calibration_diagnostics(
    residuals: np.ndarray,
    fitted_values: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration diagnostic plots.
    
    Args:
        residuals: Calibration residuals
        fitted_values: Fitted values from calibration
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs fitted
    ax1.scatter(fitted_values, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot of residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot of Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax3.hist(residuals, bins=30, alpha=0.7, density=True)
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)
    
    # Residuals over time (if applicable)
    ax4.plot(residuals)
    ax4.set_xlabel('Observation')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration diagnostics saved to {save_path}")
    
    return fig


# Initialize matplotlib style
setup_matplotlib_style()