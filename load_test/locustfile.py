import random
from collections import Counter
from pathlib import Path

from locust import HttpUser, between, events, task

ROOT = Path(__file__).resolve().parent.parent
IMAGES = [p.read_bytes() for p in sorted((ROOT / "data" / "test").rglob("*.png"))[::6]]
served_by = Counter()


class PredictionUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        with self.client.post(
            "/predict", files={"file": ("cell.png", random.choice(IMAGES), "image/png")}, catch_response=True
        ) as response:
            if response.status_code == 200:
                served_by[response.json()["instance"]] += 1
            else:
                response.failure(f"status {response.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health")


@events.quitting.add_listener
def report_distribution(environment, **kwargs):
    print("Predictions served per container:", dict(served_by))
