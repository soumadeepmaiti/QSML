import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any
import logging

logger = logging.getLogger(__name__)


def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a parquet file into a DataFrame.
    
    Args:
        path: Path to the parquet file
        
    Returns:
        DataFrame containing the data
    """
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.error(f"Error reading parquet file {path}: {e}")
        raise


def write_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Write a DataFrame to parquet (snappy compression).
    
    Args:
        df: DataFrame to write
        path: Output path for the parquet file
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression='snappy', index=False)
        logger.info(f"Successfully wrote {len(df)} rows to {path}")
    except Exception as e:
        logger.error(f"Error writing parquet file {path}: {e}")
        raise


def read_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read a CSV file with error handling.
    
    Args:
        path: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame containing the data
    """
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        logger.error(f"Error reading CSV file {path}: {e}")
        raise


def write_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """
    Write a DataFrame to CSV with error handling.
    
    Args:
        df: DataFrame to write
        path: Output path for the CSV file
        **kwargs: Additional arguments for df.to_csv
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, **kwargs)
        logger.info(f"Successfully wrote {len(df)} rows to {path}")
    except Exception as e:
        logger.error(f"Error writing CSV file {path}: {e}")
        raise


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def list_files(directory: Union[str, Path], pattern: str = "*") -> list[Path]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        
    Returns:
        List of matching file paths
    """
    return list(Path(directory).glob(pattern))