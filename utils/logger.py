import os
import json

class Logger:
    def __init__(self, output_dir):
        self.log_file = os.path.join(output_dir, "training.log")

    def log(self, message):
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def save_logs(self, history):
        logs = {key: list(value) for key, value in history.history.items()}
        with open(self.log_file, "a") as f:
            json.dump(logs, f, indent=4)
