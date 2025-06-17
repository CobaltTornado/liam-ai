import os
import asyncio
import threading
import logging
from flask import Flask, render_template, request, jsonify
from flask_sock import Sock
from main_agent import ChiefArchitectAgent, ProgressManager
from request_logger import RequestLogger  # Import the new logger

# --- Global State ---
# This is a simple way to hold a reference to the current logger per request.
# In a more complex app, you might use Flask's 'g' object or context locals.
current_request_logger = None

# --- Flask App and Logging Setup ---
app = Flask(__name__)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
sock = Sock(app)

# Intercept Werkzeug logs to add them to our custom request log
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.INFO)


def werkzeug_log_interceptor(message, *args):
    if current_request_logger:
        # Format the message if args are present
        log_message = message % args
        current_request_logger.log(log_message, 'Werkzeug')


# Replace the default handler with our interceptor
werkzeug_logger.info = werkzeug_log_interceptor

# --- Global Agent State ---
progress_manager = ProgressManager()
agent_thread = None


# --- Agent Lifecycle Management ---
def run_agent_task(prompt_text, image_data, deep_reasoning, logger: RequestLogger):
    """Function to run in a separate thread, now accepting a logger instance."""
    agent = ChiefArchitectAgent(progress_manager, logger)  # Pass the logger to the agent
    try:
        asyncio.run(agent.execute_task(prompt=prompt_text, image_data=image_data, deep_reasoning=deep_reasoning))
    except Exception as e:
        app.logger.error(f"Exception in agent thread: {e}", exc_info=True)
        logger.log(f"CRITICAL AGENT ERROR: {e}", "AgentThread")
    finally:
        # Ensure the log is saved even if the agent crashes
        logger.save()


# --- HTTP Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Receives the chat message, creates a logger, and starts the agent."""
    global agent_thread, current_request_logger
    data = request.json

    # --- Create a new logger for this specific request ---
    current_request_logger = RequestLogger(data)
    current_request_logger.log(f"Received data payload.", "App")

    prompt = data.get('prompt')
    image_data = data.get('image_data')
    deep_reasoning = data.get('deep_reasoning', False)

    if not prompt and not image_data:
        err_msg = "Prompt or image is required."
        current_request_logger.log(f"Request failed: {err_msg}", "App")
        current_request_logger.save()
        return jsonify({"error": err_msg}), 400

    if agent_thread and agent_thread.is_alive():
        err_msg = "Agent is already running. Please wait."
        current_request_logger.log(f"Request failed: {err_msg}", "App")
        current_request_logger.save()
        return jsonify({"status": err_msg}), 429

    current_request_logger.log("Starting background agent thread.", "App")
    # Pass the logger instance to the agent's execution thread
    agent_thread = threading.Thread(target=run_agent_task,
                                    args=(prompt, image_data, deep_reasoning, current_request_logger), daemon=True)
    agent_thread.start()

    return jsonify({"status": "Agent task has been started."})


# --- WebSocket Route ---
@sock.route('/ws')
def ws(socket):
    """Handles WebSocket connections for real-time updates."""
    app.logger.info("WebSocket client connected.")
    progress_manager.add_connection(socket)
    try:
        while True:
            socket.receive(timeout=60 * 5)
    except Exception:
        app.logger.info("WebSocket connection timed out or closed by client.")
    finally:
        progress_manager.remove_connection(socket)
        app.logger.info("WebSocket client removed from manager.")


# --- Main Execution ---
if __name__ == '__main__':
    extra_dirs = ['.']
    extra_files = extra_dirs[:]
    for extra_dir in extra_dirs:
        for dirname, dirs, files in os.walk(extra_dir):
            if 'workspace' in dirname or 'Request_Logs' in dirname:  # Also ignore the new log directory
                continue
            for filename in files:
                filename = os.path.join(dirname, filename)
                if os.path.isfile(filename):
                    extra_files.append(filename)

    app.run(host='0.0.0.0', port=5000, debug=True, extra_files=extra_files)
