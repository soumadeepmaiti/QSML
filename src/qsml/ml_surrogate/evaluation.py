import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from .architecture import HestonSurrogateNet, EnsembleSurrogate
from .training import HestonDataset
from ..pricers.heston_fft import price_strikes_fft
from ..pricers.heston_integral import heston_option_price_integral

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Test data
    batch_size: int = 1024
    device: str = "auto"
    
    # Metrics
    compute_percentile_errors: bool = True
    percentiles: List[float] = None
    
    # Arbitrage validation
    check_arbitrage: bool = True
    moneyness_grid: np.ndarray = None
    time_grid: np.ndarray = None
    
    # Benchmark comparison
    compare_to_heston: bool = True
    heston_pricing_method: str = "fft"  # "fft" or "integral"
    
    # Plotting
    create_plots: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300
    
    def __post_init__(self):
        if self.percentiles is None:
            self.percentiles = [50, 75, 90, 95, 99]
        
        if self.moneyness_grid is None:
            self.moneyness_grid = np.linspace(0.8, 1.2, 21)
        
        if self.time_grid is None:
            self.time_grid = np.array([1/12, 3/12, 6/12, 12/12])  # 1M, 3M, 6M, 1Y


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Basic metrics
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error
    r2: float = 0.0
    
    # Correlation metrics
    pearson_corr: float = 0.0
    spearman_corr: float = 0.0
    
    # Percentile errors
    percentile_errors: Optional[Dict[str, float]] = None
    
    # Relative error metrics
    mean_relative_error: float = 0.0
    median_relative_error: float = 0.0
    
    # Max errors
    max_absolute_error: float = 0.0
    max_relative_error: float = 0.0
    
    # Arbitrage violations
    arbitrage_violations: Optional[Dict[str, int]] = None
    
    # Timing
    inference_time_per_sample: float = 0.0  # milliseconds
    
    def to_dict(self) -> Dict[str, any]:
        """Convert metrics to dictionary."""
        result = {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'r2': self.r2,
            'pearson_corr': self.pearson_corr,
            'spearman_corr': self.spearman_corr,
            'mean_relative_error': self.mean_relative_error,
            'median_relative_error': self.median_relative_error,
            'max_absolute_error': self.max_absolute_error,
            'max_relative_error': self.max_relative_error,
            'inference_time_per_sample': self.inference_time_per_sample
        }
        
        if self.percentile_errors:
            result['percentile_errors'] = self.percentile_errors
        
        if self.arbitrage_violations:
            result['arbitrage_violations'] = self.arbitrage_violations
        
        return result


class HestonSurrogateEvaluator:
    """Comprehensive evaluation framework for Heston surrogate models."""
    
    def __init__(self, config: EvaluationConfig = None):
        """Initialize evaluator with configuration."""
        self.config = config or EvaluationConfig()
        self.device = self._get_device(self.config.device)
        self.logger = logger
    
    def _get_device(self, device_str: str) -> torch.device:
        """Get appropriate device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_str)
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_dataset: HestonDataset,
        save_dir: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained surrogate model
            test_dataset: Test dataset
            save_dir: Directory to save evaluation results
            
        Returns:
            EvaluationMetrics with all computed metrics
        """
        model.eval()
        model.to(self.device)
        
        # Get predictions and targets
        predictions, targets, inputs = self._get_model_predictions(model, test_dataset)
        
        # Compute basic metrics
        metrics = self._compute_basic_metrics(predictions, targets)
        
        # Compute percentile errors
        if self.config.compute_percentile_errors:
            metrics.percentile_errors = self._compute_percentile_errors(
                predictions, targets, self.config.percentiles
            )
        
        # Check arbitrage violations
        if self.config.check_arbitrage:
            metrics.arbitrage_violations = self._check_arbitrage_violations(
                model, inputs, test_dataset
            )
        
        # Benchmark comparison
        if self.config.compare_to_heston:
            heston_metrics = self._compare_to_heston_benchmark(
                inputs, targets, test_dataset
            )
            self.logger.info(f"Heston benchmark RMSE: {heston_metrics['rmse']:.6f}")
            self.logger.info(f"Surrogate RMSE: {metrics.rmse:.6f}")
            speedup = heston_metrics['inference_time'] / metrics.inference_time_per_sample
            self.logger.info(f"Speedup: {speedup:.1f}x")
        
        # Create evaluation plots
        if self.config.create_plots:
            self._create_evaluation_plots(
                predictions, targets, inputs, test_dataset, metrics, save_dir
            )
        
        # Log summary
        self._log_evaluation_summary(metrics)
        
        return metrics
    
    def _get_model_predictions(
        self,
        model: torch.nn.Module,
        dataset: HestonDataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions on dataset."""
        
        # Create data loader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        predictions = []
        targets = []
        inputs = []
        
        # Measure inference time
        import time
        total_time = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                
                start_time = time.time()
                
                # Forward pass
                if isinstance(model, EnsembleSurrogate):
                    batch_predictions, _ = model(batch_inputs)
                else:
                    batch_predictions = model(batch_inputs)
                
                end_time = time.time()
                
                # Accumulate timing
                total_time += (end_time - start_time)
                total_samples += len(batch_inputs)
                
                # Collect results
                predictions.append(batch_predictions.cpu().numpy())
                targets.append(batch_targets.cpu().numpy())
                inputs.append(batch_inputs.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0).flatten()
        targets = np.concatenate(targets, axis=0).flatten()
        inputs = np.concatenate(inputs, axis=0)
        
        # Store inference time
        self.inference_time_per_sample = (total_time / total_samples) * 1000  # ms
        
        return predictions, targets, inputs
    
    def _compute_basic_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> EvaluationMetrics:
        """Compute basic evaluation metrics."""
        
        # Basic regression metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Percentage error
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # Correlation metrics
        pearson_corr, _ = pearsonr(targets, predictions)
        spearman_corr, _ = spearmanr(targets, predictions)
        
        # Relative errors
        relative_errors = (predictions - targets) / (targets + 1e-8)
        mean_relative_error = np.mean(relative_errors)
        median_relative_error = np.median(relative_errors)
        
        # Max errors
        absolute_errors = np.abs(predictions - targets)
        max_absolute_error = np.max(absolute_errors)
        max_relative_error = np.max(np.abs(relative_errors))
        
        return EvaluationMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            mape=mape,
            r2=r2,
            pearson_corr=pearson_corr,
            spearman_corr=spearman_corr,
            mean_relative_error=mean_relative_error,
            median_relative_error=median_relative_error,
            max_absolute_error=max_absolute_error,
            max_relative_error=max_relative_error,
            inference_time_per_sample=self.inference_time_per_sample
        )
    
    def _compute_percentile_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        percentiles: List[float]
    ) -> Dict[str, float]:
        """Compute percentile errors."""
        
        absolute_errors = np.abs(predictions - targets)
        relative_errors = np.abs((predictions - targets) / (targets + 1e-8))
        
        percentile_errors = {}
        
        for p in percentiles:
            percentile_errors[f'abs_error_p{p}'] = np.percentile(absolute_errors, p)
            percentile_errors[f'rel_error_p{p}'] = np.percentile(relative_errors, p)
        
        return percentile_errors
    
    def _check_arbitrage_violations(
        self,
        model: torch.nn.Module,
        inputs: np.ndarray,
        dataset: HestonDataset
    ) -> Dict[str, int]:
        """Check for arbitrage violations in model predictions."""
        
        violations = {
            'monotonicity': 0,
            'convexity': 0,
            'calendar_spread': 0
        }
        
        # This would need actual implementation based on specific arbitrage constraints
        # For now, return empty violations
        self.logger.info("Arbitrage violation checking not yet implemented")
        
        return violations
    
    def _compare_to_heston_benchmark(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        dataset: HestonDataset
    ) -> Dict[str, float]:
        """Compare surrogate performance to analytical Heston pricing."""
        
        # Extract parameters from denormalized inputs
        if dataset.input_stats is not None:
            denormalized_inputs = dataset.denormalize_inputs(inputs)
        else:
            denormalized_inputs = inputs
        
        heston_predictions = []
        
        import time
        start_time = time.time()
        
        # Compute Heston prices for sample (limit to avoid long computation)
        n_samples = min(1000, len(denormalized_inputs))
        sample_indices = np.random.choice(len(denormalized_inputs), n_samples, replace=False)
        
        for i in sample_indices:
            row = denormalized_inputs[i]
            
            # Extract parameters: [S, K, r, q, T, kappa, theta, sigma, rho, v0]
            S, K, r, q, T = row[:5]
            heston_params = {
                'kappa': row[5],
                'theta': row[6], 
                'sigma': row[7],
                'rho': row[8],
                'v0': row[9]
            }
            
            try:
                if self.config.heston_pricing_method == "fft":
                    price = price_strikes_fft(S, r, q, T, heston_params, np.array([K]))[0]
                else:
                    price = heston_option_price_integral(S, K, r, q, T, heston_params, "call")
                
                heston_predictions.append(price)
            except:
                # Skip problematic parameters
                continue
        
        end_time = time.time()
        heston_time_per_sample = ((end_time - start_time) / len(heston_predictions)) * 1000
        
        # Compute metrics for the sample
        if len(heston_predictions) > 0:
            heston_predictions = np.array(heston_predictions)
            sample_targets = targets[sample_indices[:len(heston_predictions)]]
            
            heston_rmse = np.sqrt(mean_squared_error(sample_targets, heston_predictions))
            heston_mae = mean_absolute_error(sample_targets, heston_predictions)
        else:
            heston_rmse = np.nan
            heston_mae = np.nan
        
        return {
            'rmse': heston_rmse,
            'mae': heston_mae,
            'inference_time': heston_time_per_sample,
            'n_samples': len(heston_predictions)
        }
    
    def _create_evaluation_plots(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        inputs: np.ndarray,
        dataset: HestonDataset,
        metrics: EvaluationMetrics,
        save_dir: Optional[str]
    ):
        """Create comprehensive evaluation plots."""
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Prediction vs Target scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.5, s=1)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('True Option Price')
        plt.ylabel('Predicted Option Price')
        plt.title(f'Predictions vs Targets (R² = {metrics.r2:.4f})')
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            plt.savefig(save_path / f"predictions_vs_targets.{self.config.plot_format}", 
                       dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
        
        # 2. Error distribution
        errors = predictions - targets
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.axvline(0, color='red', linestyle='--')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        relative_errors = errors / (targets + 1e-8)
        plt.hist(relative_errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.title('Relative Error Distribution')
        plt.axvline(0, color='red', linestyle='--')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_path / f"error_distribution.{self.config.plot_format}",
                       dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
        
        # 3. Error by moneyness (if data is available)
        if inputs.shape[1] >= 2:  # Assume S, K are first two columns
            if dataset.input_stats is not None:
                denorm_inputs = dataset.denormalize_inputs(inputs)
            else:
                denorm_inputs = inputs
            
            S = denorm_inputs[:, 0]
            K = denorm_inputs[:, 1]
            moneyness = K / S
            
            plt.figure(figsize=(10, 6))
            plt.scatter(moneyness, np.abs(errors), alpha=0.5, s=1)
            plt.xlabel('Moneyness (K/S)')
            plt.ylabel('Absolute Error')
            plt.title('Absolute Error by Moneyness')
            plt.grid(True, alpha=0.3)
            
            if save_dir:
                plt.savefig(save_path / f"error_by_moneyness.{self.config.plot_format}",
                           dpi=self.config.dpi, bbox_inches='tight')
            plt.show()
        
        # 4. Q-Q plot for error normality
        from scipy import stats
        
        plt.figure(figsize=(8, 6))
        stats.probplot(errors, dist="norm", plot=plt)
        plt.title('Q-Q Plot: Error Distribution vs Normal')
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            plt.savefig(save_path / f"qq_plot.{self.config.plot_format}",
                       dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
    
    def _log_evaluation_summary(self, metrics: EvaluationMetrics):
        """Log evaluation summary."""
        
        self.logger.info("=== Evaluation Summary ===")
        self.logger.info(f"RMSE: {metrics.rmse:.6f}")
        self.logger.info(f"MAE: {metrics.mae:.6f}")
        self.logger.info(f"MAPE: {metrics.mape:.2f}%")
        self.logger.info(f"R²: {metrics.r2:.4f}")
        self.logger.info(f"Pearson Correlation: {metrics.pearson_corr:.4f}")
        self.logger.info(f"Mean Relative Error: {metrics.mean_relative_error:.4f}")
        self.logger.info(f"Max Absolute Error: {metrics.max_absolute_error:.6f}")
        self.logger.info(f"Inference Time: {metrics.inference_time_per_sample:.2f} ms/sample")
        
        if metrics.percentile_errors:
            self.logger.info("Percentile Errors:")
            for key, value in metrics.percentile_errors.items():
                self.logger.info(f"  {key}: {value:.6f}")


def create_evaluation_report(
    metrics: EvaluationMetrics,
    model_info: Dict[str, any],
    dataset_info: Dict[str, any],
    save_path: Optional[str] = None
) -> Dict[str, any]:
    """
    Create comprehensive evaluation report.
    
    Args:
        metrics: Evaluation metrics
        model_info: Information about the model
        dataset_info: Information about the dataset
        save_path: Path to save report JSON
        
    Returns:
        Complete evaluation report dictionary
    """
    
    report = {
        'evaluation_summary': metrics.to_dict(),
        'model_info': model_info,
        'dataset_info': dataset_info,
        'evaluation_timestamp': pd.Timestamp.now().isoformat()
    }
    
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to: {save_path}")
    
    return report


def compare_models(
    models: Dict[str, torch.nn.Module],
    test_dataset: HestonDataset,
    config: EvaluationConfig = None
) -> pd.DataFrame:
    """
    Compare multiple models on the same test dataset.
    
    Args:
        models: Dictionary of model_name -> model
        test_dataset: Test dataset
        config: Evaluation configuration
        
    Returns:
        DataFrame with comparison metrics
    """
    
    evaluator = HestonSurrogateEvaluator(config)
    results = []
    
    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        
        metrics = evaluator.evaluate_model(model, test_dataset)
        
        result = {
            'model': model_name,
            **metrics.to_dict()
        }
        results.append(result)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('rmse')
    
    return comparison_df