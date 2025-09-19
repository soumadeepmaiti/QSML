import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from .architecture import HestonSurrogateNet, EnsembleSurrogate, NetworkConfig
from .training import HestonDataset, HestonSurrogateTrainer, TrainingConfig, TrainingMetrics
from .evaluation import HestonSurrogateEvaluator, EvaluationConfig, EvaluationMetrics
from ..calibration import HestonCalibrationEngine, CalibrationConstraints, HestonParameters
from ..pricers.heston_fft import price_strikes_fft
from ..pricers.heston_integral import heston_option_price_integral

logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """Configuration for generating training data."""
    
    # Parameter ranges for data generation
    S_range: Tuple[float, float] = (50.0, 150.0)
    K_range: Tuple[float, float] = (40.0, 160.0)
    r_range: Tuple[float, float] = (0.0, 0.05)
    q_range: Tuple[float, float] = (0.0, 0.03)
    T_range: Tuple[float, float] = (1/365, 2.0)
    
    # Heston parameter ranges
    kappa_range: Tuple[float, float] = (0.5, 10.0)
    theta_range: Tuple[float, float] = (0.01, 0.5)
    sigma_range: Tuple[float, float] = (0.1, 1.0)
    rho_range: Tuple[float, float] = (-0.9, 0.9)
    v0_range: Tuple[float, float] = (0.01, 0.5)
    
    # Sampling
    n_samples: int = 100000
    sampling_method: str = "sobol"  # "random", "sobol", "lhs"
    enforce_feller: bool = True
    
    # Pricing method
    pricing_method: str = "fft"  # "fft" or "integral"
    
    # Data quality
    min_option_price: float = 0.001
    max_option_price: float = 1000.0
    filter_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations
    
    # Parallelization
    n_workers: int = None  # None = auto-detect
    chunk_size: int = 1000


def generate_sobol_samples(n_samples: int, n_dimensions: int, seed: int = 42) -> np.ndarray:
    """Generate Sobol sequence samples."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=n_dimensions, seed=seed)
        samples = sampler.random(n_samples)
        return samples
    except ImportError:
        logger.warning("scipy.stats.qmc not available, falling back to random sampling")
        np.random.seed(seed)
        return np.random.rand(n_samples, n_dimensions)


def generate_lhs_samples(n_samples: int, n_dimensions: int, seed: int = 42) -> np.ndarray:
    """Generate Latin Hypercube samples."""
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
        samples = sampler.random(n_samples)
        return samples
    except ImportError:
        logger.warning("scipy.stats.qmc not available, falling back to random sampling")
        np.random.seed(seed)
        return np.random.rand(n_samples, n_dimensions)


def sample_parameters(config: DataGenerationConfig) -> pd.DataFrame:
    """Generate parameter samples according to configuration."""
    
    n_dim = 10  # [S, K, r, q, T, kappa, theta, sigma, rho, v0]
    
    # Generate samples
    if config.sampling_method == "sobol":
        samples = generate_sobol_samples(config.n_samples, n_dim)
    elif config.sampling_method == "lhs":
        samples = generate_lhs_samples(config.n_samples, n_dim)
    else:  # random
        np.random.seed(42)
        samples = np.random.rand(config.n_samples, n_dim)
    
    # Scale to parameter ranges
    params = pd.DataFrame({
        'S': samples[:, 0] * (config.S_range[1] - config.S_range[0]) + config.S_range[0],
        'K': samples[:, 1] * (config.K_range[1] - config.K_range[0]) + config.K_range[0],
        'r': samples[:, 2] * (config.r_range[1] - config.r_range[0]) + config.r_range[0],
        'q': samples[:, 3] * (config.q_range[1] - config.q_range[0]) + config.q_range[0],
        'T': samples[:, 4] * (config.T_range[1] - config.T_range[0]) + config.T_range[0],
        'kappa': samples[:, 5] * (config.kappa_range[1] - config.kappa_range[0]) + config.kappa_range[0],
        'theta': samples[:, 6] * (config.theta_range[1] - config.theta_range[0]) + config.theta_range[0],
        'sigma': samples[:, 7] * (config.sigma_range[1] - config.sigma_range[0]) + config.sigma_range[0],
        'rho': samples[:, 8] * (config.rho_range[1] - config.rho_range[0]) + config.rho_range[0],
        'v0': samples[:, 9] * (config.v0_range[1] - config.v0_range[0]) + config.v0_range[0]
    })
    
    # Enforce Feller condition if required
    if config.enforce_feller:
        feller_mask = 2 * params['kappa'] * params['theta'] >= params['sigma']**2
        params = params[feller_mask].reset_index(drop=True)
        logger.info(f"Filtered {config.n_samples - len(params)} samples violating Feller condition")
    
    return params


def price_option_chunk(chunk_data: Tuple[pd.DataFrame, str]) -> List[Tuple[int, float]]:
    """Price a chunk of options (for parallel processing)."""
    chunk_df, pricing_method = chunk_data
    results = []
    
    for idx, row in chunk_df.iterrows():
        try:
            # Extract parameters
            S, K, r, q, T = row['S'], row['K'], row['r'], row['q'], row['T']
            heston_params = {
                'kappa': row['kappa'],
                'theta': row['theta'],
                'sigma': row['sigma'],
                'rho': row['rho'],
                'v0': row['v0']
            }
            
            # Price option
            if pricing_method == "fft":
                price = price_strikes_fft(S, r, q, T, heston_params, np.array([K]))[0]
            else:
                price = heston_option_price_integral(S, K, r, q, T, heston_params, "call")
            
            results.append((idx, price))
            
        except Exception as e:
            # Skip failed pricing
            logger.debug(f"Pricing failed for row {idx}: {e}")
            continue
    
    return results


def generate_training_data(config: DataGenerationConfig) -> pd.DataFrame:
    """
    Generate training data for surrogate model.
    
    Args:
        config: Data generation configuration
        
    Returns:
        DataFrame with input parameters and option prices
    """
    logger.info(f"Generating {config.n_samples} training samples...")
    
    # Sample parameters
    params_df = sample_parameters(config)
    logger.info(f"Generated {len(params_df)} parameter combinations")
    
    # Setup parallel processing
    n_workers = config.n_workers or mp.cpu_count()
    chunk_size = config.chunk_size
    
    # Split data into chunks
    chunks = []
    for i in range(0, len(params_df), chunk_size):
        chunk = params_df.iloc[i:i+chunk_size].copy()
        chunks.append((chunk, config.pricing_method))
    
    logger.info(f"Processing {len(chunks)} chunks with {n_workers} workers")
    
    # Process chunks in parallel
    all_results = []
    
    if n_workers == 1:
        # Sequential processing
        for chunk_data in chunks:
            results = price_option_chunk(chunk_data)
            all_results.extend(results)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_chunk = {executor.submit(price_option_chunk, chunk): i 
                             for i, chunk in enumerate(chunks)}
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    
                    if chunk_idx % 10 == 0:
                        logger.info(f"Processed chunk {chunk_idx}/{len(chunks)}")
                        
                except Exception as e:
                    logger.warning(f"Chunk {chunk_idx} failed: {e}")
    
    # Combine results
    valid_indices = [idx for idx, _ in all_results]
    prices = [price for _, price in all_results]
    
    # Filter to valid samples
    valid_df = params_df.iloc[valid_indices].copy()
    valid_df['option_price'] = prices
    
    logger.info(f"Successfully priced {len(valid_df)} options")
    
    # Quality filtering
    if config.filter_outliers:
        initial_count = len(valid_df)
        
        # Filter by price range
        price_mask = ((valid_df['option_price'] >= config.min_option_price) & 
                     (valid_df['option_price'] <= config.max_option_price))
        valid_df = valid_df[price_mask]
        
        # Filter outliers based on z-score
        prices = valid_df['option_price']
        z_scores = np.abs((prices - prices.mean()) / prices.std())
        outlier_mask = z_scores <= config.outlier_threshold
        valid_df = valid_df[outlier_mask]
        
        logger.info(f"Filtered {initial_count - len(valid_df)} outliers")
    
    valid_df = valid_df.reset_index(drop=True)
    logger.info(f"Final dataset size: {len(valid_df)} samples")
    
    return valid_df


class SurrogateModelPipeline:
    """End-to-end pipeline for training and evaluating surrogate models."""
    
    def __init__(
        self,
        data_config: DataGenerationConfig,
        network_config: NetworkConfig,
        training_config: TrainingConfig,
        evaluation_config: EvaluationConfig
    ):
        """Initialize pipeline with configurations."""
        self.data_config = data_config
        self.network_config = network_config
        self.training_config = training_config
        self.evaluation_config = evaluation_config
        
        self.logger = logger
        
        # Pipeline state
        self.training_data = None
        self.dataset = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.metrics = None
    
    def generate_data(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Generate training data."""
        self.logger.info("Starting data generation...")
        
        self.training_data = generate_training_data(self.data_config)
        
        if save_path:
            self.training_data.to_csv(save_path, index=False)
            self.logger.info(f"Training data saved to: {save_path}")
        
        return self.training_data
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load training data from file."""
        self.training_data = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(self.training_data)} training samples from {data_path}")
        return self.training_data
    
    def create_dataset(self) -> HestonDataset:
        """Create dataset from training data."""
        if self.training_data is None:
            raise ValueError("No training data available. Generate or load data first.")
        
        input_columns = ['S', 'K', 'r', 'q', 'T', 'kappa', 'theta', 'sigma', 'rho', 'v0']
        target_column = 'option_price'
        
        self.dataset = HestonDataset(
            data=self.training_data,
            input_columns=input_columns,
            target_column=target_column,
            normalize=True
        )
        
        self.logger.info(f"Created dataset with {len(self.dataset)} samples")
        return self.dataset
    
    def create_model(self, ensemble: bool = False, n_models: int = 5) -> torch.nn.Module:
        """Create surrogate model."""
        from .architecture import create_model
        
        self.model = create_model(
            config=self.network_config,
            ensemble=ensemble,
            n_models=n_models
        )
        
        from .architecture import get_model_summary
        summary = get_model_summary(self.model, (self.network_config.input_dim,))
        self.logger.info(summary)
        
        return self.model
    
    def train_model(self, save_dir: str, resume_from: Optional[str] = None) -> TrainingMetrics:
        """Train surrogate model."""
        if self.model is None:
            raise ValueError("No model available. Create model first.")
        
        if self.dataset is None:
            raise ValueError("No dataset available. Create dataset first.")
        
        self.trainer = HestonSurrogateTrainer(
            model=self.model,
            config=self.training_config,
            save_dir=save_dir
        )
        
        self.logger.info("Starting model training...")
        training_metrics = self.trainer.train(self.dataset, resume_from)
        
        self.logger.info("Training completed!")
        return training_metrics
    
    def evaluate_model(self, save_dir: str) -> EvaluationMetrics:
        """Evaluate trained model."""
        if self.model is None:
            raise ValueError("No model available. Train model first.")
        
        if self.dataset is None:
            raise ValueError("No dataset available. Create dataset first.")
        
        self.evaluator = HestonSurrogateEvaluator(self.evaluation_config)
        
        self.logger.info("Starting model evaluation...")
        self.metrics = self.evaluator.evaluate_model(
            model=self.model,
            test_dataset=self.dataset,  # In practice, would use separate test set
            save_dir=save_dir
        )
        
        self.logger.info("Evaluation completed!")
        return self.metrics
    
    def run_full_pipeline(
        self,
        output_dir: str,
        generate_new_data: bool = True,
        data_path: Optional[str] = None
    ) -> Dict[str, any]:
        """Run complete pipeline from data generation to evaluation."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        try:
            # 1. Data generation/loading
            if generate_new_data:
                data_save_path = output_path / "training_data.csv"
                self.generate_data(str(data_save_path))
            else:
                if data_path is None:
                    raise ValueError("Must provide data_path if not generating new data")
                self.load_data(data_path)
            
            results['data_samples'] = len(self.training_data)
            
            # 2. Dataset creation
            self.create_dataset()
            
            # 3. Model creation
            self.create_model()
            
            # 4. Training
            training_dir = output_path / "training"
            training_metrics = self.train_model(str(training_dir))
            results['training_metrics'] = training_metrics
            
            # 5. Evaluation
            evaluation_dir = output_path / "evaluation"
            evaluation_metrics = self.evaluate_model(str(evaluation_dir))
            results['evaluation_metrics'] = evaluation_metrics
            
            # 6. Save final results
            results_path = output_path / "pipeline_results.json"
            import json
            with open(results_path, 'w') as f:
                json.dump({
                    'data_config': self.data_config.__dict__,
                    'network_config': self.network_config.__dict__,
                    'training_config': self.training_config.__dict__,
                    'evaluation_config': self.evaluation_config.__dict__,
                    'results': results
                }, f, indent=2, default=str)
            
            self.logger.info(f"Pipeline completed! Results saved to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        return results


def create_default_configs() -> Tuple[DataGenerationConfig, NetworkConfig, TrainingConfig, EvaluationConfig]:
    """Create default configurations for the pipeline."""
    
    data_config = DataGenerationConfig(
        n_samples=50000,
        sampling_method="sobol",
        enforce_feller=True
    )
    
    network_config = NetworkConfig(
        input_dim=10,
        hidden_dims=[256, 512, 512, 256, 128],
        output_dim=1,
        dropout_rate=0.1,
        use_constraints=True,
        constraint_weight=0.1
    )
    
    training_config = TrainingConfig(
        epochs=100,
        batch_size=512,
        learning_rate=1e-3,
        patience=15,
        loss_type="mse",
        constraint_weight=0.1
    )
    
    evaluation_config = EvaluationConfig(
        compare_to_heston=True,
        create_plots=True,
        save_plots=True
    )
    
    return data_config, network_config, training_config, evaluation_config


def run_experiment(
    output_dir: str,
    experiment_name: str = "heston_surrogate",
    **config_overrides
) -> Dict[str, any]:
    """
    Run a complete surrogate model experiment.
    
    Args:
        output_dir: Directory to save all outputs
        experiment_name: Name for this experiment
        **config_overrides: Override default configuration values
        
    Returns:
        Dictionary with experiment results
    """
    
    # Create experiment directory
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configurations
    data_config, network_config, training_config, evaluation_config = create_default_configs()
    
    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(data_config, key):
            setattr(data_config, key, value)
        elif hasattr(network_config, key):
            setattr(network_config, key, value)
        elif hasattr(training_config, key):
            setattr(training_config, key, value)
        elif hasattr(evaluation_config, key):
            setattr(evaluation_config, key, value)
    
    # Create and run pipeline
    pipeline = SurrogateModelPipeline(
        data_config=data_config,
        network_config=network_config,
        training_config=training_config,
        evaluation_config=evaluation_config
    )
    
    results = pipeline.run_full_pipeline(str(exp_dir))
    
    logger.info(f"Experiment '{experiment_name}' completed!")
    logger.info(f"Results available in: {exp_dir}")
    
    return results