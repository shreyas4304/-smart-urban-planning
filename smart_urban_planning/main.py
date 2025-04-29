import os
import sys
from src.data.dataset_generator import UrbanDatasetGenerator
from src.data.data_processor import UrbanDataProcessor
from src.models.model_trainer import UrbanModelTrainer
from src.visualization.visualizer import UrbanVisualizer
from src.utils.helpers import ensure_directory, save_json

def main():
    # Create output directory
    output_dir = "output"
    ensure_directory(output_dir)
    
    # Generate sample dataset
    print("Generating sample dataset...")
    dataset_generator = UrbanDatasetGenerator(n_areas=100)
    census_data, spatial_data = dataset_generator.generate_dataset()
    
    # Save generated data
    census_data.to_csv(os.path.join(output_dir, "census_data.csv"), index=False)
    spatial_data.to_file(os.path.join(output_dir, "spatial_data.geojson"), driver='GeoJSON')
    
    # Initialize data processor
    data_processor = UrbanDataProcessor()
    
    # Preprocess features
    print("Preprocessing data...")
    categorical_cols = ['education_level']
    numerical_cols = ['population', 'median_income', 'employment_rate', 
                     'housing_density', 'public_transport_score', 
                     'green_space_percentage']
    
    X, feature_info = data_processor.preprocess_features(
        census_data.drop('development_index', axis=1),
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols
    )
    y = census_data['development_index'].values
    
    # Initialize model trainer
    model_trainer = UrbanModelTrainer()
    
    # Split data
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(X, y)
    
    # Train models
    print("Training models...")
    models = {
        'random_forest': model_trainer.train_random_forest,
        'gradient_boosting': model_trainer.train_gradient_boosting
    }
    
    results = {}
    for model_name, train_func in models.items():
        print(f"\nTraining {model_name}...")
        model = train_func(X_train, y_train)
            
        # Evaluate model
        metrics = model_trainer.evaluate_model(model_name, X_test, y_test)
        results[model_name] = metrics
        
        # Log metrics
        model_trainer.log_model_metrics(model_name, metrics)
        
        print(f"{model_name} metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Save results
    save_json(results, os.path.join(output_dir, "model_results.json"))
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = UrbanVisualizer()
    
    # Plot feature importance for Random Forest
    rf_model = model_trainer.models['random_forest']
    feature_importance = rf_model.feature_importances_
    plt = visualizer.plot_feature_importance(
        feature_info['feature_names'],
        feature_importance
    )
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    
    # Add development index to spatial data
    spatial_data = spatial_data.merge(
        census_data[['area_id', 'development_index']], 
        on='area_id'
    )
    
    # Create heatmap
    heatmap = visualizer.create_heatmap(
        spatial_data,
        'development_index'
    )
    heatmap.save(os.path.join(output_dir, "development_heatmap.html"))
    
    print("\nAnalysis complete! Results saved in the 'output' directory.")

if __name__ == "__main__":
    main() 