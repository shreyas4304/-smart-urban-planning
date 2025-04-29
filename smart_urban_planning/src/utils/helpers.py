import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directory(directory: str):
    """Ensure a directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def save_json(data: Dict[str, Any], filepath: str):
    """Save dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON data to: {filepath}")

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file into dictionary."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON data from: {filepath}")
    return data

def save_model_parameters(model_params: Dict[str, Any], 
                         model_name: str,
                         output_dir: str):
    """Save model parameters to JSON file."""
    ensure_directory(output_dir)
    filepath = os.path.join(output_dir, f"{model_name}_params.json")
    save_json(model_params, filepath)

def calculate_statistics(data: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """Calculate basic statistics for numerical data."""
    if isinstance(data, pd.Series):
        data = data.values
        
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data)
    }

def format_metrics(metrics: Dict[str, float], 
                  decimal_places: int = 4) -> Dict[str, str]:
    """Format metric values for display."""
    return {k: f"{v:.{decimal_places}f}" for k, v in metrics.items()}

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: List[str]) -> bool:
    """Validate DataFrame has required columns."""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'mean',
                         columns: List[str] = None) -> pd.DataFrame:
    """Handle missing values in DataFrame."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
        
    df_copy = df.copy()
    
    for col in columns:
        if strategy == 'mean':
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == 'median':
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
    return df_copy 