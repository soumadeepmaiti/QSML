import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML config file and return a nested dict.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML is malformed
    """
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from {path}")
        return config or {}
    except FileNotFoundError:
        logger.error(f"Config file not found: {path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config {path}: {e}")
        raise


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save a configuration dictionary to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        path: Output path for the YAML file
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Successfully saved config to {path}")
    except Exception as e:
        logger.error(f"Error saving config to {path}: {e}")
        raise


def validate_config(config: Dict[str, Any], required_keys: list[str]) -> None:
    """
    Validate that a configuration contains required keys.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys (supports dot notation for nested keys)
        
    Raises:
        ValueError: If any required key is missing
    """
    missing_keys = []
    
    for key in required_keys:
        current = config
        key_parts = key.split('.')
        
        try:
            for part in key_parts:
                current = current[part]
        except (KeyError, TypeError):
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")


def get_nested_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get a nested value from a configuration dictionary using dot notation.
    
    Args:
        config: Configuration dictionary
        key: Key in dot notation (e.g., 'heston.fft.alpha')
        default: Default value if key is not found
        
    Returns:
        Value at the specified key or default
    """
    current = config
    key_parts = key.split('.')
    
    try:
        for part in key_parts:
            current = current[part]
        return current
    except (KeyError, TypeError):
        return default


def set_nested_value(config: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set a nested value in a configuration dictionary using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        key: Key in dot notation (e.g., 'heston.fft.alpha')
        value: Value to set
    """
    current = config
    key_parts = key.split('.')
    
    # Navigate to the parent of the final key
    for part in key_parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    
    # Set the final value
    current[key_parts[-1]] = value