import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict

class UrbanDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_census_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess census data."""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error loading census data: {e}")
            return None
            
    def load_spatial_data(self, file_path: str) -> gpd.GeoDataFrame:
        """Load and preprocess spatial data (shapefiles, GeoJSON)."""
        try:
            gdf = gpd.read_file(file_path)
            return gdf
        except Exception as e:
            print(f"Error loading spatial data: {e}")
            return None
            
    def preprocess_features(self, df: pd.DataFrame, 
                          categorical_cols: List[str] = None,
                          numerical_cols: List[str] = None) -> Tuple[np.ndarray, Dict]:
        """Preprocess features for machine learning models."""
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols)
            
        if numerical_cols:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            
        feature_names = df.columns.tolist()
        return df.values, {'feature_names': feature_names}
        
    def create_spatial_features(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Create spatial features from GeoDataFrame."""
        # Calculate area
        gdf['area'] = gdf.geometry.area
        
        # Calculate centroid coordinates
        gdf['centroid_x'] = gdf.geometry.centroid.x
        gdf['centroid_y'] = gdf.geometry.centroid.y
        
        # Calculate distance to city center (assuming city center at 0,0)
        gdf['distance_to_center'] = np.sqrt(gdf['centroid_x']**2 + gdf['centroid_y']**2)
        
        return gdf
        
    def merge_datasets(self, datasets: List[pd.DataFrame], 
                      merge_keys: List[str]) -> pd.DataFrame:
        """Merge multiple datasets based on common keys."""
        merged_df = datasets[0]
        for i in range(1, len(datasets)):
            merged_df = pd.merge(merged_df, datasets[i], 
                               on=merge_keys[i-1], how='inner')
        return merged_df 