# Import BE-DataHive package
import be_datahive

# Import required packages
from sklearn.model_selection import train_test_split, cross_val_score # pip install -U scikit-learn
from sklearn.ensemble import GradientBoostingRegressor # pip install -U scikit-learn
from scipy.stats import spearmanr # pip install -U scipy
import os
import json

################################################################
# Helper functions
################################################################
# Function to select features based on groups
def select_features(features, feature_names, selected_groups):
    selected_features = []
    for group in selected_groups:
        selected_features.extend(feature_groups[group])
    
    feature_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
    return features[:, feature_indices]

# Convert tuple keys to strings
def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(elem) for elem in obj]
    else:
        return obj

# Spearman correlation scorer
def spearman_scorer(y_true, y_pred):
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return spearman_corr

################################################################
# Machine Learning Training
################################################################
# Initialize the API
api = be_datahive()

# Get efficiency data
efficiency_data = api.get_efficiency()

# Get unique base editors
base_editors = efficiency_data['base_editor'].unique()

# Create a directory to store results
dir_path = f'results'
os.makedirs(dir_path, exist_ok=True)

# Define feature combinations to test
feature_combinations = [
    ['baseline'],
    ['baseline', 'energy_terms'],
    ['baseline', 'melting_temperature'],
    ['baseline', 'energy_terms', 'melting_temperature'],
]

# Set target column
target_col = "efficiency_full_grna_reported"

# Run analysis for each base editor
for base_editor in base_editors:
    print(f"\nAnalyzing base editor: {base_editor}")
    
    # Subset the data for the current base editor
    subset_data = efficiency_data[efficiency_data['base_editor'] == base_editor]

    # Update feature_groups based on the actual columns in variable_info["features"]
    feature_groups = {
        'baseline': [f for f in subset_data.columns if f.startswith('one_hot_grna') or f.startswith('one_hot_full_context_sequence_padded')],
        'energy_terms': [f for f in subset_data.columns if f.startswith('energy_') or f == 'free_energy'],
        'melting_temperature': [f for f in subset_data.columns if f.startswith('melt_temperature_')]
    }
    
    # Define all columns we want to keep
    all_feature_columns = set()
    for group in feature_groups.values():
        all_feature_columns.update(group)
    
    # Add the target column
    all_feature_columns.add(target_col)
    
    # Keep only the required columns
    subset_data = subset_data[list(all_feature_columns)]

    # Get features and target for the subset
    features, target, variable_info = api.get_efficiency_ml_arrays(subset_data, target_col=target_col, encoding='one-hot', clean=True, flatten=True)

    # Update feature_groups based on the actual columns in variable_info["features"]
    feature_groups = {
        'baseline': [f for f in variable_info["features"] if f.startswith('one_hot_grna') or f.startswith('one_hot_sequence')],
        'energy_terms': [f for f in variable_info["features"] if f.startswith('energy_') or f == 'free_energy'],
        'melting_temperature': [f for f in variable_info["features"] if f.startswith('melt_temperature_')]
    }

    # Get all feature names
    feature_names = variable_info["features"]
    
    # Initialize results dictionary
    results = {}

    # Loop through combinations
    for combination in feature_combinations:
        print(f"\nTraining with feature groups: {combination}")
        
        # Select features for this combination
        X = select_features(features, feature_names, combination)
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
        
        # Initialize the gradient boost regressor
        gbr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(gbr, X_train, y_train, cv=5, scoring=spearman_scorer)
        cv_spearman_corr = cv_scores.mean()
        
        # Fit the model on the entire training set
        gbr.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = gbr.predict(X_test)
        
        # Calculate Spearman correlation
        spearman_corr, _ = spearmanr(y_test, y_pred)
        
        # Store results
        results[tuple(combination)] = {
            'cv_spearman_corr': float(cv_spearman_corr),
            'test_spearman_corr': float(spearman_corr)
        }
        
        print(f"Cross-validation Spearman correlation: {cv_spearman_corr}")
        print(f"Test Spearman correlation: {spearman_corr}")

        if len(results) > 0:
            # Find best performing combination
            best_combination = max(results, key=lambda x: results[x]['test_spearman_corr'])
            
            # Add best combination to results
            results['best_combination'] = {
                'feature_groups': list(best_combination),
                'test_spearman_corr': float(results[best_combination]['test_spearman_corr'])
            }
            
            # Convert the results dictionary
            results_str_keys = convert_keys_to_str(results)

            # Save results to a JSON file
            with open(f'{dir_path}/{base_editor}_{target_col}_results.json', 'w') as f:
                json.dump(results_str_keys, f, indent=4)
            
            print(f"\nResults for {base_editor} saved to {dir_path}/{base_editor}_{target_col}_results.json")
