"""
Machine Learning Surrogate Models for Heston Option Pricing.

This package provides a complete framework for training neural network
surrogate models to approximate Heston option pricing:

- Sophisticated neural network architectures with market-aware features
- Arbitrage constraint enforcement during training
- Comprehensive training pipeline with advanced optimizers
- Thorough evaluation framework with benchmarking
- End-to-end pipeline for data generation, training, and evaluation
"""

from .architecture import (
    HestonSurrogateNet,
    EnsembleSurrogate,
    NetworkConfig,
    ArbitrageConstraint,
    MonotonicityConstraint,
    ConvexityConstraint,
    PutCallParityConstraint,
    create_model,
    count_parameters,
    get_model_summary
)

from .training import (
    HestonDataset,
    TrainingConfig,
    TrainingMetrics,
    HestonSurrogateTrainer
)

from .evaluation import (
    EvaluationConfig,
    EvaluationMetrics,
    HestonSurrogateEvaluator,
    create_evaluation_report,
    compare_models
)

from .pipeline import (
    DataGenerationConfig,
    SurrogateModelPipeline,
    generate_training_data,
    create_default_configs,
    run_experiment
)

__all__ = [
    # Architecture
    'HestonSurrogateNet',
    'EnsembleSurrogate', 
    'NetworkConfig',
    'ArbitrageConstraint',
    'MonotonicityConstraint',
    'ConvexityConstraint',
    'PutCallParityConstraint',
    'create_model',
    'count_parameters',
    'get_model_summary',
    
    # Training
    'HestonDataset',
    'TrainingConfig',
    'TrainingMetrics',
    'HestonSurrogateTrainer',
    
    # Evaluation
    'EvaluationConfig',
    'EvaluationMetrics',
    'HestonSurrogateEvaluator',
    'create_evaluation_report',
    'compare_models',
    
    # Pipeline
    'DataGenerationConfig',
    'SurrogateModelPipeline',
    'generate_training_data',
    'create_default_configs',
    'run_experiment'
]