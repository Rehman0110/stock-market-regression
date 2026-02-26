from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def build_model():

    model = RandomForestRegressor(random_state=42)

    param_dist = {
        "n_estimators": [100, 150, 200],          
        "max_depth": [8, 12, 16],                 
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=8,            # slightly reduced
        cv=3,
        n_jobs=-1,
        random_state=42
    )

    return random_search