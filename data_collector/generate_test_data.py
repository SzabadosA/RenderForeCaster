import random
from datetime import datetime, timedelta
import uuid
import influxdb_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
# InfluxDB connection details
from utils import get_client

client, org, bucket = get_client()
write_api = client.write_api(write_options=SYNCHRONOUS)

# Configuration for generating data
project_names = ["lidl_0123", "nike_0456", "apple_0789"]
production_labels = ["lighting_complex", "lighting_easy", "lighting_mid", "fx_complex", "fx_easy"]
tasks = ["lighting", "animation_playblast", "compositing"]
resolutions = ["1920x1080", "1280x720", "2560x1440", "3840x2160"]
file_types = ["exr", "png", "jpg"]
job_statuses = ["active", "idle", "error", "finished"]
render_nodes = [{"name": f"node_{i}", "RAM": random.choice([32, 64, 128]), "CPU": random.choice([8, 16, 32]),
                 "GPU_VRAM": random.choice([8, 12, 24]), "timescale": random.uniform(0.5, 1.0)} for i in range(15)]

# Quality settings for different submissions of the same shot
quality_settings = [
    {"aa_samples": 2, "resolution": "1280x720", "label": "low"},
    {"aa_samples": 4, "resolution": "1920x1080", "label": "medium"},
    {"aa_samples": 8, "resolution": "2560x1440", "label": "high"},
    {"aa_samples": 16, "resolution": "3840x2160", "label": "final"}
]


# Function to generate a random job name
def generate_job_name(project, task, frame_number):
    return f"{project}_{task}.{frame_number:04d}"


# Function to generate a single render job data point
def generate_render_job_data(active_jobs_count, project, task, frame_number, quality_setting):
    job_id = str(uuid.uuid4())
    production_label = random.choice(production_labels)
    resolution = quality_setting["resolution"]
    aa_samples = quality_setting["aa_samples"]
    polygon_count = random.randint(10000, 1000000)
    light_count = random.randint(1, 20)
    aov_count = random.randint(1, 10)
    file_size = random.uniform(1.0, 10.0)
    file_type = random.choice(file_types)
    render_node = random.choice(render_nodes)

    # Calculate render time based on attributes
    base_time = 10  # base render time in minutes
    render_time = base_time * (1 + aa_samples * 0.1 + polygon_count * 0.00001 + light_count * 0.2 + aov_count * 0.1) / \
                  render_node["timescale"]
    total_render_time = render_time * 50  # assuming 50 frames per shot

    # Ensure not more active jobs than render nodes
    if active_jobs_count < len(render_nodes):
        job_status = random.choices(job_statuses, weights=[0.2, 0.4, 0.1, 0.3], k=1)[0]
        if job_status == "active":
            active_jobs_count += 1
    else:
        job_status = random.choices(job_statuses, weights=[0, 0.4, 0.1, 0.5], k=1)[
            0]  # No active jobs if active_jobs_count exceeds render nodes

    job_name = generate_job_name(project, task, frame_number)
    return {
        "measurement": "render_jobs",
        "tags": {
            "job_id": job_id,
            "project_name": project,
            "job_name": job_name,
            "file_type": file_type,
            "production_label": production_label,
            "task": task,
            "resolution": resolution,
            "render_node": render_node["name"],
            "job_status": job_status,
            "quality": quality_setting["label"]
        },
        "fields": {
            "file_size": file_size,
            "frame_number": frame_number,
            "aa_samples": aa_samples,
            "polygon_count": polygon_count,
            "light_count": light_count,
            "aov_count": aov_count,
            "render_time": render_time,
            "total_render_time": total_render_time
        },
        "time": datetime.utcnow().isoformat()
    }, active_jobs_count


# Function to write render job data to InfluxDB
def write_data_to_influxdb(data):
    for point in data:
        influx_point = (
            Point(point["measurement"])
            .tag("job_id", point["tags"]["job_id"])
            .tag("project_name", point["tags"]["project_name"])
            .tag("job_name", point["tags"]["job_name"])
            .tag("file_type", point["tags"]["file_type"])
            .tag("production_label", point["tags"]["production_label"])
            .tag("task", point["tags"]["task"])
            .tag("resolution", point["tags"]["resolution"])
            .tag("render_node", point["tags"]["render_node"])
            .tag("job_status", point["tags"]["job_status"])
            .tag("quality", point["tags"]["quality"])
            .field("file_size", point["fields"]["file_size"])
            .field("frame_number", point["fields"]["frame_number"])
            .field("aa_samples", point["fields"]["aa_samples"])
            .field("polygon_count", point["fields"]["polygon_count"])
            .field("light_count", point["fields"]["light_count"])
            .field("aov_count", point["fields"]["aov_count"])
            .field("render_time", point["fields"]["render_time"])
            .field("total_render_time", point["fields"]["total_render_time"])
            .time(point["time"])
        )
        write_api.write(bucket=bucket, org=org, record=influx_point)


# Generate and write 10,000 render job data points
data = []
active_jobs_count = 0

# Generate jobs for multiple submissions of the same shot with varying quality settings
for _ in range(10000 // len(quality_settings)):
    project = random.choice(project_names)
    task = random.choice(tasks)
    frame_number = random.randint(1000, 1050)
    for quality_setting in quality_settings:
        job_data, active_jobs_count = generate_render_job_data(active_jobs_count, project, task, frame_number,
                                                               quality_setting)
        data.append(job_data)

write_data_to_influxdb(data)
print("Test data written to InfluxDB successfully.")
