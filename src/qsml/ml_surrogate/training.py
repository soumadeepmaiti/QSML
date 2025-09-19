
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import time
from .architecture import HestonSurrogateNet, EnsembleSurrogate, NetworkConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training surrogate models."""
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 512
    shuffle: bool = True
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adam"  # adam, adamw, sgd, rmsprop
    scheduler: str = "cosine"  # cosine, plateau, step, exponential
    
    # Training
    epochs: int = 100
    patience: int = 10  # Early stopping
    min_delta: float = 1e-6  # Minimum improvement for early stopping
    
    # Loss function
    loss_type: str = "mse"  # mse, mae, huber, relative_mse
    constraint_weight: float = 0.1
    regularization_weight: float = 0.0
    
    # Validation
    val_frequency: int = 1  # Validate every N epochs
    save_best: bool = True
    save_frequency: int = 10  # Save checkpoint every N epochs
    
    # Monitoring
    log_frequency: int = 100  # Log every N batches
    plot_training: bool = True
    
    # Reproducibility
    seed: Optional[int] = 42
    
    # Device
    device: str = "auto"  # auto, cpu, cuda, mps


class HestonDataset(Dataset):
    """Dataset for Heston option pricing data."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        input_columns: List[str],
        target_column: str,
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        """
        Args:
            data: DataFrame with market data and prices
            input_columns: Column names for model inputs
            target_column: Column name for target prices
            transform: Optional data transformation
            normalize: Whether to normalize inputs
        """
        self.data = data.copy()
        self.input_columns = input_columns
        self.target_column = target_column
        self.transform = transform
        
        # Extract inputs and targets
        self.inputs = data[input_columns].values.astype(np.float32)
        self.targets = data[target_column].values.astype(np.float32)
        
        # Normalization
        if normalize:
            self._compute_normalization()
            self.inputs = self._normalize_inputs(self.inputs)
        else:
            self.input_stats = None
    
    def _compute_normalization(self):
        """Compute normalization statistics."""
        self.input_stats = {
            'mean': np.mean(self.inputs, axis=0),
            'std': np.std(self.inputs, axis=0) + 1e-8  # Avoid division by zero
        }
    
    def _normalize_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize inputs using computed statistics."""
        if self.input_stats is None:
            return inputs
        
        return (inputs - self.input_stats['mean']) / self.input_stats['std']
    
    def denormalize_inputs(self, normalized_inputs: np.ndarray) -> np.ndarray:
        """Denormalize inputs."""
        if self.input_stats is None:
            return normalized_inputs
        
        return normalized_inputs * self.input_stats['std'] + self.input_stats['mean']
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.from_numpy(self.inputs[idx])
        targets = torch.from_numpy(np.array([self.targets[idx]]))
        
        if self.transform:
            inputs, targets = self.transform(inputs, targets)
        
        return inputs, targets


def get_loss_function(loss_type: str) -> Callable:
    """Get loss function by name."""
    
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.HuberLoss()
    elif loss_type == "relative_mse":
        def relative_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            relative_error = (pred - target) / (target + 1e-8)
            return torch.mean(relative_error ** 2)
        return relative_mse
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Get optimizer by name."""
    
    if config.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9
        )
    elif config.optimizer == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_scheduler(optimizer: optim.Optimizer, config: TrainingConfig) -> Optional[Any]:
    """Get learning rate scheduler."""
    
    if config.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs
        )
    elif config.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config.patience // 2,
            verbose=True
        )
    elif config.scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.epochs // 3,
            gamma=0.1
        )
    elif config.scheduler == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    else:
        return None


class TrainingMetrics:
    """Track training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.train_losses = []
        self.val_losses = []
        self.constraint_losses = []
        self.learning_rates = []
        self.epochs = []
        self.batch_times = []
        
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        constraint_loss: Optional[float] = None,
        lr: Optional[float] = None,
        batch_time: Optional[float] = None
    ):
        """Update metrics."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        if constraint_loss is not None:
            self.constraint_losses.append(constraint_loss)
        
        if lr is not None:
            self.learning_rates.append(lr)
        
        if batch_time is not None:
            self.batch_times.append(batch_time)
    
    def get_best_epoch(self) -> int:
        """Get epoch with best validation loss."""
        if not self.val_losses:
            return len(self.train_losses) - 1
        
        return np.argmin(self.val_losses)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(self.epochs, self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0, 0].plot(self.epochs, self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Constraint losses
        if self.constraint_losses:
            axes[0, 1].plot(self.epochs, self.constraint_losses, label='Constraint Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Constraint Loss')
            axes[0, 1].set_title('Constraint Violations')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate
        if self.learning_rates:
            axes[1, 0].plot(self.epochs, self.learning_rates)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True)
        
        # Batch times
        if self.batch_times:
            axes[1, 1].plot(self.epochs, self.batch_times)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time per Batch (s)')
            axes[1, 1].set_title('Training Speed')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class HestonSurrogateTrainer:
    """Trainer for Heston surrogate models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        save_dir: Optional[str] = None
    ):
        """
        Args:
            model: Neural network model to train
            config: Training configuration
            save_dir: Directory to save checkpoints and logs
        """
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Set device
        self.device = self._get_device(config.device)
        self.model.to(self.device)
        
        # Training components
        self.optimizer = get_optimizer(model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        self.loss_fn = get_loss_function(config.loss_type)
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logging
        self.logger = logger
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def create_data_loaders(
        self,
        dataset: HestonDataset
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test data loaders."""
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Random split
        generator = torch.Generator()
        if self.config.seed is not None:
            generator.manual_seed(self.config.seed)
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_constraint_loss = 0.0
        n_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, EnsembleSurrogate):
                predictions, _, constraint_loss = self.model.forward_with_constraints(inputs)
            elif hasattr(self.model, 'forward_with_constraints'):
                predictions, constraint_loss = self.model.forward_with_constraints(inputs)
            else:
                predictions = self.model(inputs)
                constraint_loss = torch.tensor(0.0, device=self.device)
            
            # Compute primary loss
            primary_loss = self.loss_fn(predictions, targets)
            
            # Total loss
            total_batch_loss = (
                primary_loss + 
                self.config.constraint_weight * constraint_loss
            )
            
            # Add regularization if specified
            if self.config.regularization_weight > 0:
                l2_reg = torch.tensor(0.0, device=self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2)
                total_batch_loss += self.config.regularization_weight * l2_reg
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += primary_loss.item()
            total_constraint_loss += constraint_loss.item()
            n_batches += 1
            
            # Logging
            if batch_idx % self.config.log_frequency == 0:
                batch_time = time.time() - batch_start_time
                self.logger.debug(
                    f"Batch {batch_idx}/{len(train_loader)}: "
                    f"Loss={primary_loss.item():.6f}, "
                    f"Constraint={constraint_loss.item():.6f}, "
                    f"Time={batch_time:.3f}s"
                )
        
        avg_loss = total_loss / n_batches
        avg_constraint_loss = total_constraint_loss / n_batches
        
        return avg_loss, avg_constraint_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_constraint_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if isinstance(self.model, EnsembleSurrogate):
                    predictions, _, constraint_loss = self.model.forward_with_constraints(inputs)
                elif hasattr(self.model, 'forward_with_constraints'):
                    predictions, constraint_loss = self.model.forward_with_constraints(inputs)
                else:
                    predictions = self.model(inputs)
                    constraint_loss = torch.tensor(0.0, device=self.device)
                
                # Compute loss
                loss = self.loss_fn(predictions, targets)
                
                total_loss += loss.item()
                total_constraint_loss += constraint_loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_constraint_loss = total_constraint_loss / n_batches
        
        return avg_loss, avg_constraint_loss
    
    def train(
        self,
        dataset: HestonDataset,
        resume_from: Optional[str] = None
    ) -> TrainingMetrics:
        """
        Main training loop.
        
        Args:
            dataset: Training dataset
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training metrics
        """
        # Set random seeds
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(dataset)
        
        self.logger.info(f"Training on {len(train_loader.dataset)} samples")
        self.logger.info(f"Validation on {len(val_loader.dataset)} samples") 
        self.logger.info(f"Test on {len(test_loader.dataset)} samples")
        self.logger.info(f"Device: {self.device}")
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        # Training loop
        for epoch in range(start_epoch, self.config.epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_constraint_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_constraint_loss = None, None
            if epoch % self.config.val_frequency == 0:
                val_loss, val_constraint_loss = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics
            epoch_time = time.time() - epoch_start_time
            self.metrics.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                constraint_loss=train_constraint_loss,
                lr=current_lr,
                batch_time=epoch_time / len(train_loader)
            )
            
            # Logging
            log_msg = f"Epoch {epoch}/{self.config.epochs}: Train Loss={train_loss:.6f}"
            if val_loss is not None:
                log_msg += f", Val Loss={val_loss:.6f}"
            log_msg += f", LR={current_lr:.2e}, Time={epoch_time:.1f}s"
            
            self.logger.info(log_msg)
            
            # Save best model
            if val_loss is not None and val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if self.config.save_best and self.save_dir:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Regular checkpoint saving
            if epoch % self.config.save_frequency == 0 and self.save_dir:
                self.save_checkpoint(epoch)
        
        # Final test evaluation
        test_loss, test_constraint_loss = self.validate(test_loader)
        self.logger.info(f"Final test loss: {test_loss:.6f}")
        
        # Plot training curves
        if self.config.plot_training and self.save_dir:
            plot_path = self.save_dir / "training_curves.png"
            self.metrics.plot_training_curves(str(plot_path))
        
        return self.metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.save_dir:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'metrics': self.metrics,
            'config': self.config
        }
        
        if is_best:
            save_path = self.save_dir / "best_model.pt"
        else:
            save_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.metrics = checkpoint['metrics']
        
        epoch = checkpoint['epoch']
        self.logger.info(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch + 1