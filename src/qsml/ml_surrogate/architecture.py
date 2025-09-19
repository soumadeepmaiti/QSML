import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ArbitrageConstraint(ABC):
    """Abstract base class for arbitrage constraints."""
    
    @abstractmethod
    def apply(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply constraint to network outputs."""
        pass
    
    @abstractmethod
    def loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute constraint violation loss."""
        pass


class MonotonicityConstraint(ArbitrageConstraint):
    """Enforce monotonicity constraints on option prices."""
    
    def __init__(self, strike_dim: int = 1, penalty_weight: float = 1.0):
        """
        Args:
            strike_dim: Dimension index for strike price in input tensor
            penalty_weight: Weight for constraint violation penalty
        """
        self.strike_dim = strike_dim
        self.penalty_weight = penalty_weight
    
    def apply(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply monotonicity constraint via projection."""
        # For call options: price should decrease with strike
        # For put options: price should increase with strike
        
        # Sort by strike price within each batch
        batch_size = inputs.shape[0]
        constrained_outputs = outputs.clone()
        
        for i in range(batch_size):
            strikes = inputs[i, self.strike_dim]
            prices = outputs[i]
            
            # Sort by strike
            sorted_indices = torch.argsort(strikes)
            sorted_prices = prices[sorted_indices]
            
            # Apply isotonic regression (simplified)
            # For calls: enforce decreasing prices
            for j in range(1, len(sorted_prices)):
                if sorted_prices[j] > sorted_prices[j-1]:
                    sorted_prices[j] = sorted_prices[j-1]
            
            # Map back to original order
            constrained_outputs[i][sorted_indices] = sorted_prices
        
        return constrained_outputs
    
    def loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute monotonicity violation loss."""
        batch_size = inputs.shape[0]
        violations = torch.zeros(batch_size, device=inputs.device)
        
        for i in range(batch_size):
            strikes = inputs[i, self.strike_dim]
            prices = outputs[i]
            
            # Sort by strike
            sorted_indices = torch.argsort(strikes)
            sorted_strikes = strikes[sorted_indices]
            sorted_prices = prices[sorted_indices]
            
            # Compute violations
            price_diffs = sorted_prices[1:] - sorted_prices[:-1]
            strike_diffs = sorted_strikes[1:] - sorted_strikes[:-1]
            
            # For calls: price differences should be non-positive
            positive_violations = torch.relu(price_diffs)
            violations[i] = torch.sum(positive_violations)
        
        return self.penalty_weight * torch.mean(violations)


class ConvexityConstraint(ArbitrageConstraint):
    """Enforce convexity constraints on option prices."""
    
    def __init__(self, strike_dim: int = 1, penalty_weight: float = 1.0):
        self.strike_dim = strike_dim
        self.penalty_weight = penalty_weight
    
    def apply(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply convexity constraint via projection."""
        # Call option prices should be convex in strike
        # This is a simplified implementation
        return outputs  # TODO: Implement convexity projection
    
    def loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute convexity violation loss."""
        batch_size = inputs.shape[0]
        violations = torch.zeros(batch_size, device=inputs.device)
        
        for i in range(batch_size):
            strikes = inputs[i, self.strike_dim]
            prices = outputs[i]
            
            # Need at least 3 points for convexity
            if len(strikes) < 3:
                continue
            
            # Sort by strike
            sorted_indices = torch.argsort(strikes)
            sorted_strikes = strikes[sorted_indices]
            sorted_prices = prices[sorted_indices]
            
            # Check convexity using second differences
            for j in range(1, len(sorted_strikes) - 1):
                h1 = sorted_strikes[j] - sorted_strikes[j-1]
                h2 = sorted_strikes[j+1] - sorted_strikes[j]
                
                if h1 > 0 and h2 > 0:
                    # Second difference
                    second_diff = (sorted_prices[j+1] - sorted_prices[j]) / h2 - (sorted_prices[j] - sorted_prices[j-1]) / h1
                    
                    # Convexity violation (negative second difference)
                    violation = torch.relu(-second_diff)
                    violations[i] += violation
        
        return self.penalty_weight * torch.mean(violations)


class PutCallParityConstraint(ArbitrageConstraint):
    """Enforce put-call parity constraint."""
    
    def __init__(
        self,
        S_dim: int = 0,
        K_dim: int = 1,
        r_dim: int = 2,
        T_dim: int = 3,
        penalty_weight: float = 1.0
    ):
        self.S_dim = S_dim
        self.K_dim = K_dim
        self.r_dim = r_dim
        self.T_dim = T_dim
        self.penalty_weight = penalty_weight
    
    def apply(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply put-call parity constraint."""
        # Assume outputs contains both call and put prices
        # This would need to be adapted based on network architecture
        return outputs
    
    def loss(self, inputs: torch.Tensor, call_prices: torch.Tensor, put_prices: torch.Tensor) -> torch.Tensor:
        """Compute put-call parity violation loss."""
        S = inputs[:, self.S_dim]
        K = inputs[:, self.K_dim]
        r = inputs[:, self.r_dim]
        T = inputs[:, self.T_dim]
        
        # Put-call parity: C - P = S - K * exp(-r * T)
        parity_value = S - K * torch.exp(-r * T)
        parity_violation = torch.abs(call_prices - put_prices - parity_value)
        
        return self.penalty_weight * torch.mean(parity_violation)


@dataclass
class NetworkConfig:
    """Configuration for surrogate neural network."""
    
    # Architecture
    input_dim: int = 8  # [S, K, r, q, T, kappa, theta, sigma, rho, v0]
    hidden_dims: List[int] = None
    output_dim: int = 1  # Option price
    dropout_rate: float = 0.1
    
    # Activation functions
    activation: str = "relu"  # relu, tanh, gelu, swish
    output_activation: str = "softplus"  # For positive prices
    
    # Normalization
    batch_norm: bool = True
    layer_norm: bool = False
    
    # Initialization
    init_method: str = "xavier_uniform"
    
    # Constraints
    use_constraints: bool = True
    constraint_weight: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 512, 256, 128]


class PositionalEncoding(nn.Module):
    """Positional encoding for time-to-expiry and moneyness."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, dim_idx: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            dim_idx: Dimension index to encode
        """
        # Extract the dimension to encode
        values = x[:, dim_idx]
        
        # Scale to reasonable range for positional encoding
        scaled_values = (values * 1000).long().clamp(0, self.pe.size(0) - 1)
        
        # Get positional encodings
        encodings = self.pe[scaled_values]
        
        return encodings


class MonetnessEncoder(nn.Module):
    """Special encoder for moneyness (K/S) with market-aware features."""
    
    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(4, output_dim)  # [moneyness, log_moneyness, moneyness^2, 1/moneyness]
    
    def forward(self, S: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: Spot prices
            K: Strike prices
            
        Returns:
            Encoded moneyness features
        """
        moneyness = K / S
        log_moneyness = torch.log(moneyness)
        
        features = torch.stack([
            moneyness,
            log_moneyness,
            moneyness ** 2,
            1.0 / (moneyness + 1e-8)
        ], dim=-1)
        
        return self.linear(features)


class HestonSurrogateNet(nn.Module):
    """
    Neural network surrogate for Heston option pricing.
    
    Features:
    - Market-aware input encoding
    - Arbitrage constraints
    - Specialized architecture for financial data
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Input encoding layers
        self.moneyness_encoder = MonetnessEncoder(output_dim=16)
        self.time_encoder = PositionalEncoding(d_model=16)
        
        # Calculate encoded input dimension
        # Original inputs: [S, K, r, q, T, kappa, theta, sigma, rho, v0] = 10
        # Moneyness encoding: 16 features
        # Time encoding: 16 features  
        # Remaining features: [r, q, kappa, theta, sigma, rho, v0] = 7
        encoded_dim = 16 + 16 + 7
        
        # Main network layers
        layers = []
        input_dim = encoded_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(config.activation))
            
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, config.output_dim))
        
        if config.output_activation:
            layers.append(self._get_activation(config.output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Arbitrage constraints
        if config.use_constraints:
            self.constraints = [
                MonotonicityConstraint(strike_dim=1, penalty_weight=config.constraint_weight),
                ConvexityConstraint(strike_dim=1, penalty_weight=config.constraint_weight)
            ]
        else:
            self.constraints = []
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'softplus': nn.Softplus(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU()
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        
        return activations[name]
    
    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            if self.config.init_method == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif self.config.init_method == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif self.config.init_method == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif self.config.init_method == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def encode_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode raw inputs with market-aware features.
        
        Args:
            inputs: [batch_size, 10] tensor with [S, K, r, q, T, kappa, theta, sigma, rho, v0]
            
        Returns:
            Encoded features tensor
        """
        # Extract components
        S = inputs[:, 0]
        K = inputs[:, 1]
        r = inputs[:, 2]
        q = inputs[:, 3]
        T = inputs[:, 4]
        heston_params = inputs[:, 5:]  # [kappa, theta, sigma, rho, v0]
        
        # Encode moneyness
        moneyness_features = self.moneyness_encoder(S, K)
        
        # Encode time to expiry
        time_features = self.time_encoder(inputs, dim_idx=4)
        
        # Combine all features
        encoded = torch.cat([
            moneyness_features,
            time_features,
            r.unsqueeze(1),
            q.unsqueeze(1),
            heston_params
        ], dim=1)
        
        return encoded
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            inputs: Input tensor with market data and Heston parameters
            
        Returns:
            Predicted option prices
        """
        # Encode inputs
        encoded = self.encode_inputs(inputs)
        
        # Forward pass through main network
        outputs = self.network(encoded)
        
        return outputs
    
    def forward_with_constraints(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with constraint loss computation.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Tuple of (predicted_prices, constraint_loss)
        """
        outputs = self.forward(inputs)
        
        # Compute constraint losses
        constraint_loss = torch.tensor(0.0, device=inputs.device)
        
        for constraint in self.constraints:
            constraint_loss += constraint.loss(inputs, outputs)
        
        return outputs, constraint_loss
    
    def apply_constraints(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply constraints to outputs."""
        constrained_outputs = outputs
        
        for constraint in self.constraints:
            constrained_outputs = constraint.apply(inputs, constrained_outputs)
        
        return constrained_outputs


class EnsembleSurrogate(nn.Module):
    """Ensemble of surrogate networks for uncertainty quantification."""
    
    def __init__(self, config: NetworkConfig, n_models: int = 5):
        super().__init__()
        self.n_models = n_models
        
        # Create ensemble of networks
        self.models = nn.ModuleList([
            HestonSurrogateNet(config) for _ in range(n_models)
        ])
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        predictions = []
        
        for model in self.models:
            pred = model(inputs)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [n_models, batch_size, output_dim]
        
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return mean_pred, std_pred
    
    def forward_with_constraints(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with constraint losses."""
        predictions = []
        constraint_losses = []
        
        for model in self.models:
            pred, const_loss = model.forward_with_constraints(inputs)
            predictions.append(pred)
            constraint_losses.append(const_loss)
        
        predictions = torch.stack(predictions, dim=0)
        constraint_losses = torch.stack(constraint_losses, dim=0)
        
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        mean_constraint_loss = torch.mean(constraint_losses)
        
        return mean_pred, std_pred, mean_constraint_loss


def create_model(config: NetworkConfig, ensemble: bool = False, n_models: int = 5) -> nn.Module:
    """
    Factory function to create surrogate model.
    
    Args:
        config: Network configuration
        ensemble: Whether to create ensemble model
        n_models: Number of models in ensemble
        
    Returns:
        Surrogate model (single or ensemble)
    """
    if ensemble:
        return EnsembleSurrogate(config, n_models)
    else:
        return HestonSurrogateNet(config)


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> str:
    """Get summary of model architecture."""
    n_params = count_parameters(model)
    
    # Try to get output shape
    try:
        dummy_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            if isinstance(model, EnsembleSurrogate):
                output, _ = model(dummy_input)
            else:
                output = model(dummy_input)
        output_shape = output.shape[1:]
    except:
        output_shape = "Unknown"
    
    summary = f"""
Model Summary:
- Type: {type(model).__name__}
- Input shape: {input_shape}
- Output shape: {output_shape}
- Total parameters: {n_params:,}
- Memory usage: ~{n_params * 4 / 1024 / 1024:.1f} MB (float32)
"""
    
    return summary