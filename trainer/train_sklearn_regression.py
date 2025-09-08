import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import joblib
import argparse
from dataset import DamageOutcomeDataset, MAP_NAMES, WEAPON_NAMES
import random
from sklearn.mixture import GaussianMixture


def extract_features_from_dataset(dataset):
    """
    Extract features and target values from the dataset.
    
    Args:
        dataset: DamageOutcomeDataset instance
    
    Returns:
        X: Feature matrix as numpy array
        y: Target values (damage amount) as numpy array
    """
    # Initialize empty lists for features and targets
    features = []
    targets = []
    
    # Process each item in the dataset
    for i in range(len(dataset)):
        item = dataset[i]
        
        # Extract map features (one-hot encoded)
        map_features = item['map'].numpy()
        
        # Extract coordinates and angles
        coords_and_angles = item['coords_and_angles'].numpy()
        
        # Extract weapon features (one-hot encoded)
        weapon_features = item['weapon'].numpy()
        
        # Extract boolean features
        armor_features = item['armor_features'].numpy()
        
        # Combine all features into a single vector
        combined_features = np.concatenate([
            map_features,
            coords_and_angles,
            weapon_features,
            armor_features
        ])
        
        features.append(combined_features)
        
        # Extract damage amount (targets[1])
        damage = item['targets'].numpy()[1]
        targets.append(damage)
    
    # Convert lists to numpy arrays
    X = np.array(features)
    y = np.array(targets)
    
    print(f"Extracted {len(X)} samples with {X.shape[1]} features")
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y



def train_and_evaluate_models(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate multiple regression models.
    
    Args:
        X: Feature matrix
        y: Target values
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        results_df: DataFrame with model performance metrics
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to evaluate with verbose settings for those that support it
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        # 'SVR (RBF)': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            verbose=1  # Enable progress reporting
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            verbose=1  # Enable progress reporting
        )
    }
    
    # Results storage
    results = []
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Calculate RMSE
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        # Training time
        train_time = time.time() - start_time
        
        # Store results
        results.append({
            'Model': name,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Training Time (s)': train_time
        })
        
        # Print results
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Train MAE: {train_mae:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Save the model
        model_filename = f"models/{name.replace(' ', '_').lower()}_regressor.joblib"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")
        
        # Save the scaler with the model
        scaler_filename = f"models/{name.replace(' ', '_').lower()}_scaler.joblib"
        joblib.dump(scaler, scaler_filename)
        print(f"Scaler saved to {scaler_filename}")
        
        # For tree-based models, plot feature importance
        if name in ['Random Forest', 'Gradient Boosting']:
            plot_feature_importance(model, name)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, models

def plot_feature_importance(model, model_name):
    """Plot feature importance for tree-based models."""
    # Get feature importances
    importances = model.feature_importances_
    
    # Create feature names
    feature_names = []
    
    # Map names
    for map_name in MAP_NAMES:
        feature_names.append(f"Map_{map_name}")
    
    # Coordinates and angles
    coord_names = [
        "Attacker_X", "Attacker_Y", "Attacker_Z", 
        "Attacker_View_X", 
        "Victim_X", "Victim_Y", "Victim_Z", 
        "Victim_View_X", 
        "Attacker_HP", 
        "Distance", 
        "Relative_Angle"
    ]
    feature_names.extend(coord_names)
    
    # Weapons
    for weapon in WEAPON_NAMES:
        feature_names.append(f"Weapon_{weapon}")
    
    # Boolean features
    feature_names.extend(["Victim_Has_Helmet", "Victim_Has_Armor"])
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.title(f"Feature Importance - {model_name}")
    plt.bar(range(20), importances[indices[:20]])
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig(f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    # Print top 10 features
    print("\nTop 10 most important features:")
    for i in range(10):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def plot_results(results_df):
    """Plot model comparison results."""
    # Plot RMSE comparison
    plt.figure(figsize=(12, 6))
    models = results_df['Model']
    train_rmse = results_df['Train RMSE']
    test_rmse = results_df['Test RMSE']
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, train_rmse, width, label='Train RMSE')
    plt.bar(x + width/2, test_rmse, width, label='Test RMSE')
    
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('Model Comparison - RMSE')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison_rmse.png')
    plt.close()
    
    # Plot R² comparison
    plt.figure(figsize=(12, 6))
    train_r2 = results_df['Train R²']
    test_r2 = results_df['Test R²']
    
    plt.bar(x - width/2, train_r2, width, label='Train R²')
    plt.bar(x + width/2, test_r2, width, label='Test R²')
    
    plt.xlabel('Model')
    plt.ylabel('R²')
    plt.title('Model Comparison - R²')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison_r2.png')
    plt.close()

def analyze_best_model(X, y, best_model_name, models, test_size=0.2, random_state=42):
    """Analyze the best performing model in more detail."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get the best model
    best_model = models[best_model_name]
    
    # Make predictions
    y_pred = best_model.predict(X_test_scaled)
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.xlabel('Actual Damage')
    plt.ylabel('Predicted Damage')
    plt.title(f'{best_model_name} - Actual vs Predicted Damage')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    # Plot error distribution
    errors = y_pred - y_test
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{best_model_name} - Error Distribution')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()
    
    # Print detailed metrics
    print(f"\nDetailed Analysis of {best_model_name}:")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    # Calculate percentage of predictions within different error margins
    error_margins = [1, 5, 10, 20, 50]
    for margin in error_margins:
        within_margin = np.sum(np.abs(errors) <= margin) / len(errors) * 100
        print(f"Predictions within ±{margin} damage: {within_margin:.2f}%")

def train_with_latent_variables(X, y, test_size=0.2, random_state=42):
    """
    Train a model that attempts to account for latent hit location variables.
    
    This approach uses a mixture model to implicitly capture different damage distributions
    that might correspond to different hit locations.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # First, try to identify clusters in the data that might correspond to hit locations
    # Combine features and target for clustering
    X_y_train = np.column_stack((X_train_scaled, y_train.reshape(-1, 1)))
    
    # Fit a Gaussian Mixture Model to identify potential hit locations
    n_components = 5  # You might need to tune this (head, chest, arms, legs, etc.)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(X_y_train)
    
    # Get cluster assignments
    train_clusters = gmm.predict(X_y_train)
    
    # Add cluster assignments as a feature
    X_train_with_clusters = np.column_stack((X_train_scaled, np.eye(n_components)[train_clusters]))
    
    # For test data, predict the cluster
    X_test_with_clusters = np.column_stack((
        X_test_scaled, 
        gmm.predict_proba(np.column_stack((X_test_scaled, np.zeros((X_test.shape[0], 1)))))
    ))
    
    # Train a model with these augmented features
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state,
        verbose=1
    )
    
    model.fit(X_train_with_clusters, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_with_clusters)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Latent Variable Model - Test RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    return model, scaler, gmm

def train_multimodal_model(X, y, test_size=0.2, random_state=42):
    """
    Train a model that explicitly handles the multi-modal nature of damage distribution.
    This approach uses quantile regression to capture different parts of the distribution.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple quantile regression models to capture different parts of the distribution
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_models = {}
    
    for q in quantiles:
        print(f"\nTraining quantile regression model for {q} quantile...")
        model = GradientBoostingRegressor(
            loss='quantile', 
            alpha=q,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            verbose=1
        )
        
        model.fit(X_train_scaled, y_train)
        quantile_models[q] = model
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        print(f"Quantile {q} - Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    
    # Also train a model for the mean prediction
    print("\nTraining model for mean prediction...")
    mean_model = GradientBoostingRegressor(
        loss='ls',  # least squares for mean prediction
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state,
        verbose=1
    )
    mean_model.fit(X_train_scaled, y_train)
    
    # Evaluate mean model
    y_pred_mean = mean_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    r2 = r2_score(y_test, y_pred_mean)
    print(f"Mean Model - Test RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    # Return all models
    return mean_model, quantile_models, scaler

def plot_quantile_predictions(X, y, mean_model, quantile_models, scaler, n_samples=100):
    """Plot predictions with quantile ranges to visualize uncertainty."""
    # Select a random subset of samples
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    
    # Scale features
    X_scaled = scaler.transform(X_subset)
    
    # Get predictions
    mean_preds = mean_model.predict(X_scaled)
    quantile_preds = {q: model.predict(X_scaled) for q, model in quantile_models.items()}
    
    # Sort by actual damage for better visualization
    sort_idx = np.argsort(y_subset)
    y_sorted = y_subset[sort_idx]
    mean_sorted = mean_preds[sort_idx]
    quantile_sorted = {q: preds[sort_idx] for q, preds in quantile_preds.items()}
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot actual values
    plt.scatter(range(n_samples), y_sorted, color='black', alpha=0.6, label='Actual Damage')
    
    # Plot mean prediction
    plt.plot(range(n_samples), mean_sorted, 'r-', label='Mean Prediction')
    
    # Plot quantile ranges
    plt.fill_between(
        range(n_samples), 
        quantile_sorted[0.1], 
        quantile_sorted[0.9],
        alpha=0.2, color='blue', label='10-90% Quantile Range'
    )
    
    plt.fill_between(
        range(n_samples), 
        quantile_sorted[0.25], 
        quantile_sorted[0.75],
        alpha=0.3, color='blue', label='25-75% Quantile Range'
    )
    
    plt.xlabel('Sample Index (sorted by actual damage)')
    plt.ylabel('Damage')
    plt.title('Damage Prediction with Uncertainty Ranges')
    plt.legend()
    plt.tight_layout()
    plt.savefig('quantile_predictions.png')
    plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train sklearn regression models for damage prediction')
    parser.add_argument('--positive_samples_file', type=str, default='../data/positive_samples.npy',
                        help='Path to saved positive samples file')
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='Number of samples to extract if positive_samples_file does not exist')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_state)
    
    # Load dataset with block loading and preloaded positive samples
    data_dir = os.path.join("..", "data", "player_seq_allmap_full_npz_damage")
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    random.shuffle(file_paths)
    positive_samples = np.load(args.positive_samples_file)
    # Shuffle positive samples before splitting using numpy
    shuffle_indices = np.random.permutation(len(positive_samples))
    
    dataset = DamageOutcomeDataset(
        positive_samples=positive_samples,
        positive_only=True,
        balance_strategy='none'
    )
    dataset.reload_data()
        
    # Extract features and targets
    X, y = extract_features_from_dataset(
        dataset=dataset,
    )

    # # Train and evaluate models
    # results_df, models = train_and_evaluate_models(
    #     X, y, 
    #     test_size=args.test_size, 
    #     random_state=args.random_state
    # )
    
    # # Save results to CSV
    # results_df.to_csv('model_comparison_results.csv', index=False)
    # print("\nResults saved to model_comparison_results.csv")
    
    # # Plot results
    # plot_results(results_df)
    
    # # Find the best model based on Test RMSE
    # best_model_idx = results_df['Test RMSE'].idxmin()
    # best_model_name = results_df.loc[best_model_idx, 'Model']
    # print(f"\nBest model based on Test RMSE: {best_model_name}")
    
    # # Analyze the best model in more detail
    # analyze_best_model(X, y, best_model_name, models, 
    #                   test_size=args.test_size, 
    #                   random_state=args.random_state)
    
    # Add this after your regular model training
    print("\nTraining model with latent variable approach...")
    latent_model, latent_scaler, gmm = train_with_latent_variables(
        X, y, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    # Save these models
    os.makedirs("models", exist_ok=True)
    joblib.dump(latent_model, "models/latent_variable_model.joblib")
    joblib.dump(latent_scaler, "models/latent_variable_scaler.joblib")
    joblib.dump(gmm, "models/latent_variable_gmm.joblib")
    
    # Add this after your regular model training
    print("\nTraining multi-modal distribution model...")
    mean_model, quantile_models, mm_scaler = train_multimodal_model(
        X, y, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    # Plot quantile predictions
    plot_quantile_predictions(X, y, mean_model, quantile_models, mm_scaler)
    
    # Save these models
    os.makedirs("models", exist_ok=True)
    joblib.dump(mean_model, "models/multimodal_mean_model.joblib")
    joblib.dump(quantile_models, "models/multimodal_quantile_models.joblib")
    joblib.dump(mm_scaler, "models/multimodal_scaler.joblib")
    
    # Print total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")