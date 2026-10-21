import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

# Sample FoodieApp dataset
data = pd.DataFrame({
    "price": [100, 150, 200, 250, 120, 180, 300, 220, 130, 170],
    "delivery_time": [20, 25, 30, 35, 18, 28, 40, 32, 22, 26],
    "orders": [100, 150, 200, 250, 120, 180, 300, 220, 130, 170],
    "rating": [4.2, 4.3, 4.5, 4.6, 4.1, 4.4, 4.8, 4.6, 4.2, 4.3]
})

X = data[["price", "delivery_time", "orders"]]
y = data["rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

mlflow.set_experiment("FoodieApp_Food_Rating")

with mlflow.start_run():

    n_estimators = 100

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("algorithm", "Random Forest")
    mlflow.log_param("n_estimators", n_estimators)

    # Log metrics
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2_Score", r2)

    # Log model
    mlflow.sklearn.log_model(
        model,
        "food_rating_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("Model trained successfully")
    print("MSE:", mse)
    print("R2 Score:", r2)