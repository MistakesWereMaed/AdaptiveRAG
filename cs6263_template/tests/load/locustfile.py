"""Locust load test driver.

Run from the repo root with the app already up:

    locust -f tests/load/locustfile.py --headless -u 20 -r 5 -t 60s \\
           --host=http://localhost:8080 --csv=reports/loadtest

Rubric thresholds:
  * sustained ≥ 10 req/s with < 5% error rate over a 60-second window
"""
from locust import HttpUser, task, between


class HeadlineUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def submit_query(self):
        self.client.post("/api/query", json={"text": "What is FIPS 140-3?"})

    @task(1)
    def health(self):
        self.client.get("/health")
