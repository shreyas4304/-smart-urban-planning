import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import geopandas as gpd
import numpy as np
from typing import List, Dict, Any

class UrbanVisualizer:
    def __init__(self):
        plt.style.use('default')
        
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: np.ndarray,
                              title: str = "Feature Importance"):
        """Plot feature importance scores."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance_scores, y=feature_names)
        plt.title(title)
        plt.xlabel("Importance Score")
        plt.tight_layout()
        return plt
        
    def plot_prediction_vs_actual(self, y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                title: str = "Predicted vs Actual Values"):
        """Plot predicted vs actual values."""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2)
        plt.title(title)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.tight_layout()
        return plt
        
    def create_heatmap(self, gdf: gpd.GeoDataFrame, 
                      value_column: str,
                      title: str = "Urban Development Heatmap"):
        """Create an interactive heatmap using Folium."""
        # Calculate center of the map
        center_lat = gdf.geometry.centroid.y.mean()
        center_lon = gdf.geometry.centroid.x.mean()
        
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], 
                      zoom_start=12)
        
        # Create heatmap data
        heat_data = [[row.geometry.centroid.y, 
                     row.geometry.centroid.x, 
                     row[value_column]] 
                    for _, row in gdf.iterrows()]
        
        # Add heatmap layer
        HeatMap(heat_data).add_to(m)
        
        # Add title
        folium.TileLayer('cartodbpositron').add_to(m)
        folium.LayerControl().add_to(m)
        
        return m
        
    def plot_training_history(self, history: Dict[str, List[float]],
                            title: str = "Model Training History"):
        """Plot training history for neural network models."""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot metrics
        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        return plt
        
    def plot_spatial_distribution(self, gdf: gpd.GeoDataFrame,
                                value_column: str,
                                title: str = "Spatial Distribution"):
        """Plot spatial distribution of values."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        gdf.plot(column=value_column, 
                ax=ax, 
                legend=True,
                cmap='viridis')
        plt.title(title)
        return plt