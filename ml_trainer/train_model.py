import pandas as pd
from influxdb_client import InfluxDBClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Fetch data from InfluxDB
def fetch_data():
    query = """
    from(bucket: "your_bucket")
    |> range(start: -30d)  # Adjust the range as needed
    |> filter(fn: (r) => r._measurement == "render_jobs")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    with InfluxDBClient(url="http://influxdb:8086", token="your_token", org="your_org") as client:
        tables = client.query_api().query(query=query)
        data = [record.values for table in tables for record in table.records]
        return pd.DataFrame(data)

# Train model
def train_model(data):
    X = data[["sampling", "resolution", "hardware"]]
    y = data["rendertime"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, "model.joblib")
    print("Model trained and saved")

# Main function
if __name__ == "__main__":
    data = fetch_data()
    train_model(data)
