import sys; sys.path.append('C:\\Users\\hp\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages')
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import random
from typing import Tuple

class UrbanDatasetGenerator:
    def __init__(self, n_areas: int = 100):
        self.n_areas = n_areas
        # Punjab, India bounding box (approx): lat 29.5 to 32.5, lon 73.8 to 76.9
        self.lat_min, self.lat_max = 29.5, 32.5
        self.lon_min, self.lon_max = 73.8, 76.9
        self.city_center = Point((self.lon_min + self.lon_max) / 2, (self.lat_min + self.lat_max) / 2)
        
    def generate_census_data(self) -> pd.DataFrame:
        """Generate sample census data."""
        data = {
            'area_id': range(1, self.n_areas + 1),
            'population': np.random.randint(1000, 100000, self.n_areas),
            'median_income': np.random.randint(50000, 300000, self.n_areas),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], self.n_areas),
            'employment_rate': np.random.uniform(0.5, 0.98, self.n_areas),
            'housing_density': np.random.randint(50, 2000, self.n_areas),
            'public_transport_score': np.random.uniform(0, 1, self.n_areas),
            'green_space_percentage': np.random.uniform(0.05, 0.4, self.n_areas)
        }
        return pd.DataFrame(data)
        
    def generate_spatial_data(self) -> gpd.GeoDataFrame:
        """Generate sample spatial data with polygons in Punjab, India."""
        points = []
        for _ in range(self.n_areas):
            lon = np.random.uniform(self.lon_min, self.lon_max)
            lat = np.random.uniform(self.lat_min, self.lat_max)
            points.append(Point(lon, lat))
        
        polygons = []
        for point in points:
            x, y = point.x, point.y
            polygon = Polygon([
                (x-0.05, y-0.05),
                (x+0.05, y-0.05),
                (x+0.05, y+0.05),
                (x-0.05, y+0.05)
            ])
            polygons.append(polygon)
        
        gdf = gpd.GeoDataFrame({
            'area_id': range(1, self.n_areas + 1),
            'geometry': polygons
        })
        gdf['distance_to_center'] = gdf.geometry.centroid.distance(self.city_center)
        gdf['area'] = gdf.geometry.area
        return gdf
        
    def generate_development_index(self, census_data: pd.DataFrame, 
                                 spatial_data: gpd.GeoDataFrame) -> pd.Series:
        """Generate development index with balanced feature importances and some noise."""
        # Normalize features for balanced contribution
        def norm(x):
            return (x - x.min()) / (x.max() - x.min())
        
        pop = norm(census_data['population'])
        income = norm(census_data['median_income'])
        edu = census_data['education_level'].map({'High School': 0.2, 'Bachelor': 0.5, 'Master': 0.8, 'PhD': 1.0})
        emp = norm(census_data['employment_rate'])
        housing = norm(census_data['housing_density'])
        transport = norm(census_data['public_transport_score'])
        green = norm(census_data['green_space_percentage'])
        dist = norm(spatial_data['distance_to_center'])
        
        # Balanced formula
        development_index = (
            0.18 * pop +
            0.18 * income +
            0.14 * edu +
            0.14 * emp +
            0.12 * housing +
            0.10 * transport +
            0.08 * green -
            0.06 * dist +
            np.random.normal(0, 0.03, self.n_areas)  # add some noise
        )
        # Normalize to 0-1
        development_index = (development_index - development_index.min()) / (development_index.max() - development_index.min())
        return development_index
        
    def generate_dataset(self) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
        census_data = self.generate_census_data()
        spatial_data = self.generate_spatial_data()
        development_index = self.generate_development_index(census_data, spatial_data)
        census_data['development_index'] = development_index
        return census_data, spatial_data 