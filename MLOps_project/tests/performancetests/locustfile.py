import random
from locust import HttpUser, between, task

class MyUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        self.client.get("/")

    @task(3)
    def get_item(self) -> None:
        item_id = random.randint(1, 10)
        self.client.get(f"/items/{item_id}")
