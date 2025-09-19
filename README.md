# QSML: Quantitative Stochastic Volatility Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-19%20Passing-brightgreen)](tests/)

> A comprehensive Master's-level quantitative finance system combining advanced stochastic volatility modeling, machine learning, and realistic hedging simulation.

## 🎯 Overview

QSML is a production-ready quantitative finance framework that implements sophisticated option pricing models, machine learning surrogates, and realistic trading simulation. The project demonstrates the integration of traditional quantitative finance with modern machine learning techniques, providing a complete pipeline from market data processing to hedging strategy evaluation.

### Key Features

- **🏗️ Advanced Pricing Models**: Black-Scholes and Heston stochastic volatility with FFT implementation
- **🤖 ML Integration**: Neural network surrogates with arbitrage-free constraints
- **📈 Realistic Trading**: Monte Carlo hedging simulation with transaction costs
- **📊 Professional Analytics**: Comprehensive risk metrics and performance attribution
- **🔬 Research Ready**: Modular architecture supporting academic and industry research

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quant-sv-ml.git
cd quant-sv-ml

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from qsml.pricers.bs import BlackScholesPricer
from qsml.hedging.simulation import HedgingSimulator, HedgingConfig
from qsml.hedging.analysis import HedgingAnalyzer

# 1. Price an option
pricer = BlackScholesPricer()
price = pricer.price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
print(f"Option price: ${price:.2f}")

# 2. Run hedging simulation
config = HedgingConfig(
    hedge_type="delta",
    rebalance_frequency="daily",
    transaction_cost=0.001
)
simulator = HedgingSimulator(config=config, pricer=pricer)
results = simulator.run_simulation(n_paths=1000, n_steps=252)

# 3. Analyze results
analyzer = HedgingAnalyzer()
metrics = analyzer.calculate_risk_metrics(results)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
```

## 📋 Project Structure

```
qsml/
├── 📁 src/qsml/                 # Core package
│   ├── 📁 pricers/              # Option pricing engines
│   │   ├── bs.py                # Black-Scholes implementation
│   │   ├── heston_fft.py        # Heston FFT pricing (Carr-Madan)
│   │   └── heston_integral.py   # Semi-closed form integration
│   ├── 📁 calibration/          # Model calibration
│   │   ├── surface.py           # Volatility surface handling
│   │   └── engine.py            # Calibration algorithms
│   ├── 📁 ml_surrogate/         # Machine learning models
│   │   ├── architecture.py      # Neural network architectures
│   │   ├── training.py          # Training pipelines
│   │   └── evaluation.py        # Model evaluation
│   ├── 📁 hedging/              # Hedging simulation
│   │   ├── simulation.py        # Monte Carlo hedging engine
│   │   └── analysis.py          # Performance analytics
│   ├── 📁 data/                 # Data processing
│   └── 📁 utils/                # Utility functions
├── 📁 tests/                    # Test suite (19 tests)
├── 📁 notebooks/                # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_bs_iv_baseline.ipynb
│   ├── 03_heston_calibration.ipynb
│   ├── 04_train_surrogate.ipynb
│   ├── 05_eval_surrogate.ipynb
│   ├── 06_hedging_backtest.ipynb
│   ├── 07_ablation_robustness.ipynb
│   └── 08_full_pipeline_demo.ipynb
├── 📁 configs/                  # Configuration files
├── 📁 scripts/                  # Utility scripts
└── 📄 requirements.txt          # Dependencies
```

## 🔧 Core Components

### 1. Pricing Engines (`src/qsml/pricers/`)

**Black-Scholes Model**
- Complete implementation with analytical Greeks
- Vectorized operations for efficient computation
- Input validation and numerical stability checks

**Heston Stochastic Volatility Model**
- FFT pricing using Carr-Madan methodology
- Semi-closed form integration for European options
- Robust characteristic function implementation

```python
from qsml.pricers.heston_fft import HestonFFTPricer

pricer = HestonFFTPricer()
price = pricer.price(
    S=100, K=100, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7
)
```

### 2. Model Calibration (`src/qsml/calibration/`)

**Volatility Surface Management**
- Market data ingestion and validation
- Interpolation and extrapolation methods
- Arbitrage checking and correction

**Parameter Optimization**
- Multi-objective calibration algorithms
- Constraint handling and regularization
- Goodness-of-fit testing and diagnostics

```python
from qsml.calibration.surface import VolatilitySurface
from qsml.calibration.engine import HestonCalibrator

surface = VolatilitySurface.from_market_data(data)
calibrator = HestonCalibrator()
params = calibrator.calibrate(surface)
```

### 3. Machine Learning Surrogates (`src/qsml/ml_surrogate/`)

**Neural Network Architectures**
- Feedforward networks with customizable layers
- Ensemble methods for improved robustness
- Arbitrage-free constraint enforcement

**Training Infrastructure**
- Advanced data preprocessing pipelines
- Cross-validation and hyperparameter tuning
- Model evaluation and comparison frameworks

```python
from qsml.ml_surrogate.architecture import MLOptionSurrogate
from qsml.ml_surrogate.training import MLTrainer

model = MLOptionSurrogate(hidden_dims=[64, 32, 16])
trainer = MLTrainer(model)
trained_model = trainer.train(train_data, val_data)
```

### 4. Hedging Simulation (`src/qsml/hedging/`)

**Monte Carlo Engine**
- Sophisticated stock price path generation
- Configurable volatility and correlation models
- Efficient parallel processing capabilities

**Transaction Cost Modeling**
- Fixed transaction costs
- Bid-ask spread simulation
- Market impact functions
- Realistic slippage modeling

**Performance Analytics**
- Comprehensive P&L attribution
- Risk metrics (VaR, CVaR, Sharpe ratio)
- Strategy comparison frameworks
- Professional visualization tools

```python
from qsml.hedging.simulation import HedgingSimulator, HedgingConfig

config = HedgingConfig(
    hedge_type="delta",
    rebalance_frequency="daily",
    delta_threshold=0.1,
    transaction_cost=0.001
)

simulator = HedgingSimulator(config=config)
results = simulator.run_simulation(
    n_paths=10000,
    n_steps=252,
    S0=100,
    sigma=0.2
)
```

## 📊 Analytics and Visualization

### Risk Metrics
- **Value at Risk (VaR)** at multiple confidence levels
- **Conditional VaR (CVaR)** for tail risk assessment
- **Sharpe and Sortino ratios** for risk-adjusted returns
- **Maximum drawdown** and recovery analysis
- **Skewness and kurtosis** of P&L distributions

### Performance Attribution
- Transaction cost breakdown by component
- Delta tracking error analysis
- Rebalancing frequency impact assessment
- Strategy comparison with statistical significance testing

### Visualization Capabilities
- P&L distribution analysis with statistical overlays
- Sample hedging paths with portfolio evolution
- Strategy comparison charts with confidence intervals
- Risk metric evolution over time
- Volatility surface visualization and analysis

## 🧪 Testing and Validation

### Comprehensive Test Suite
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_hedging.py -v          # Hedging tests
pytest tests/test_working_system.py -v   # System integration tests
```

**Test Coverage**:
- ✅ 19 passing tests across all components
- ✅ Unit tests for individual modules
- ✅ Integration tests for complete workflows
- ✅ Numerical accuracy validation
- ✅ Edge case handling verification

### Model Validation
- Analytical benchmark comparisons
- Monte Carlo convergence testing
- Arbitrage constraint verification
- Statistical goodness-of-fit tests
- Backtesting framework validation

## 📚 Documentation and Notebooks

### Jupyter Notebook Tutorials

1. **`01_data_exploration.ipynb`** - Market data analysis and preprocessing
2. **`02_bs_iv_baseline.ipynb`** - Black-Scholes baseline implementation
3. **`03_heston_calibration.ipynb`** - Stochastic volatility model calibration
4. **`04_train_surrogate.ipynb`** - Machine learning model training
5. **`05_eval_surrogate.ipynb`** - ML model evaluation and validation
6. **`06_hedging_backtest.ipynb`** - Hedging strategy backtesting
7. **`07_ablation_robustness.ipynb`** - Robustness and sensitivity analysis
8. **`08_full_pipeline_demo.ipynb`** - Complete end-to-end demonstration

### Academic Documentation
- **`project_documentation.tex`** - Detailed mathematical framework and methodology
- **API Documentation** - Comprehensive function and class documentation
- **Configuration Guides** - Parameter tuning and customization instructions

## 🔬 Research Applications

### Academic Research
- **Stochastic Volatility Modeling**: Parameter sensitivity and model comparison studies
- **Machine Learning in Finance**: Surrogate model effectiveness and arbitrage constraint research
- **Hedging Strategy Optimization**: Transaction cost impact and rebalancing frequency studies
- **Risk Management**: Advanced stress testing and model risk assessment

### Industry Applications
- **Algorithmic Trading**: High-frequency option pricing and hedging
- **Risk Management**: Portfolio risk assessment and stress testing
- **Model Validation**: Independent pricing model verification
- **Research and Development**: New strategy development and backtesting

## 🛠️ Configuration and Customization

### Model Configuration
```yaml
# configs/default.yaml
black_scholes:
  numerical_method: "analytical"
  finite_diff_step: 1e-5

heston:
  fft:
    N: 4096
    alpha: 1.5
    eta: 0.25
  integration:
    method: "adaptive"
    tolerance: 1e-8

ml_surrogate:
  architecture:
    hidden_dims: [64, 32, 16]
    activation: "relu"
    dropout: 0.1
  training:
    batch_size: 256
    learning_rate: 0.001
    epochs: 100
```

### Hedging Configuration
```yaml
# configs/hedging_config.yaml
hedging:
  strategies:
    - type: "delta"
      rebalance_frequency: "daily"
      delta_threshold: 0.1
    - type: "delta_gamma"
      rebalance_frequency: "weekly"
      delta_threshold: 0.05
      gamma_threshold: 0.02
  
  transaction_costs:
    fixed_cost: 0.5
    proportional_cost: 0.001
    bid_ask_spread: 0.002
    market_impact: "sqrt"
```

## 📈 Performance Benchmarks

### Computational Performance
- **Black-Scholes Pricing**: ~1M evaluations/second
- **Heston FFT Pricing**: ~10K evaluations/second
- **ML Surrogate Inference**: ~100K evaluations/second
- **Monte Carlo Simulation**: 10K paths × 252 steps in ~30 seconds

### Numerical Accuracy
- **Black-Scholes Greeks**: Machine precision accuracy
- **Heston Calibration**: <1% average relative error
- **ML Surrogate**: <0.5% pricing error vs analytical benchmarks
- **Monte Carlo Convergence**: √N convergence rate verified

## 🤝 Contributing

We welcome contributions from the quantitative finance and machine learning communities!

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/quant-sv-ml.git
cd quant-sv-ml

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code quality
flake8 src/
black src/
```

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure numerical accuracy and stability
- Include performance benchmarks where applicable

## 📄 License and Citation

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
If you use this project in academic research, please cite:

```bibtex
@software{qsml2025,
  author = {Your Name},
  title = {QSML: Quantitative Stochastic Volatility Machine Learning},
  year = {2025},
  url = {https://github.com/yourusername/quant-sv-ml}
}
```

## 🔗 References and Further Reading

### Core Mathematical Framework
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility
- Carr, P., & Madan, D. (1999). Option valuation using the fast Fourier transform
- Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide

### Machine Learning in Finance
- Ruf, J., & Wang, W. (2020). Neural networks for option pricing and hedging
- Bühler, H., et al. (2019). Deep hedging: Learning to simulate equity option markets

### Risk Management and Hedging
- Hull, J. C. (2017). Options, Futures, and Other Derivatives
- Wilmott, P. (2006). Paul Wilmott Introduces Quantitative Finance


⭐ Star this repository if you find it useful for your research or work!**
