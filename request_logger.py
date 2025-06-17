import os
import json
from datetime import datetime
from threading import Lock

class RequestLogger:
    """
    Manages the creation and writing of a comprehensive log for a single agent request.
    This class is designed to be thread-safe for use in a web server environment.
    """
    LOG_DIR = "Request_Logs"

    def __init__(self, request_data: dict):
        """
        Initializes the logger for a new request.

        Args:
            request_data (dict): The initial JSON data received from the POST request.
        """
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)

        self._lock = Lock()
        self.start_time = datetime.now()
        self.filename = f"Request_Log_{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        self.log_path = os.path.join(self.LOG_DIR, self.filename)

        # Initialize the structured log content
        self.log_content = {
            "request_info": {
                "timestamp_utc": self.start_time.isoformat(),
                "prompt": request_data.get('prompt'),
                "image_data_present": request_data.get('image_data') is not None,
                "deep_reasoning_requested": request_data.get('deep_reasoning', False)
            },
            "server_log": [],
            "agent_execution_log": [],
            "final_summary": None
        }
        self.log("Logger initialized.", "RequestLogger")

    def log(self, message: str, source: str = "Werkzeug"):
        """
        Adds a generic log entry (e.g., from the web server).

        Args:
            message (str): The log message.
            source (str): The source of the log (e.g., 'Werkzeug', 'App').
        """
        with self._lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "message": message
            }
            self.log_content["server_log"].append(entry)

    def log_agent_broadcast(self, data: dict):
        """
        Adds a log entry that was broadcast to the UI (e.g., plans, thoughts, results).

        Args:
            data (dict): The data dictionary that was sent to the WebSocket.
        """
        with self._lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": data.get('type'),
                "payload": data.get('payload')
            }
            self.log_content["agent_execution_log"].append(entry)

    def set_final_summary(self, summary: dict):
        """
        Adds the final outcome of the agent's task.

        Args:
            summary (dict): A summary containing the status and a final message.
        """
        with self._lock:
            self.log_content["final_summary"] = summary
        self.save()

    def save(self):
        """
        Saves the complete log content to its JSON file.
        This is typically called once at the very end of the request lifecycle.
        """
        with self._lock:
            self.log_content["request_info"]["duration_seconds"] = (datetime.now() - self.start_time).total_seconds()
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self.log_content, f, indent=4)
        print(f"Request log saved to {self.log_path}")

