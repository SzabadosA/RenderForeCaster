import requests
import json
from influxdb_client import InfluxDBClient, Point

# Example function to get data from Thinkbox Deadline (dummy data here)
def get_render_data():
    # This should be replaced with actual data collection logic
    return [
        {"time": "2024-06-19T00:00:00Z", "rendertime": 120, "sampling": 50, "resolution": "1920x1080", "hardware": "GPU1"},
        # Add more data points as needed
    ]

# Function to write data to InfluxDB
def write_to_influxdb(data):
    with InfluxDBClient(url="http://influxdb:8086", token="your_token", org="your_org") as client:
        write_api = client.write_api()
        for entry in data:
            point = Point("render_jobs").time(entry["time"]).field("rendertime", entry["rendertime"]).field("sampling", entry["sampling"]).field("resolution", entry["resolution"]).field("hardware", entry["hardware"])
            write_api.write(bucket="your_bucket", record=point)

# Main function
if __name__ == "__main__":
    data = get_render_data()
    write_to_influxdb(data)
