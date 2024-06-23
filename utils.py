from influxdb_client.client.influxdb_client import InfluxDBClient


def get_client():
    bucket = "example_renderdata"
    org = "bepic"
    token = "zVcPT1ghC101fnl6TjLEGR8X7LQ7CWEBlTZ3P8Ufxq-YsmeMaI3DY02lpP0DTMraygyf0ZVstaX_9aLSyXUuJw=="
    url = "http://influxdb:8086"

    # Initialize InfluxDB client
    return InfluxDBClient(url=url, token=token, org=org), org, bucket