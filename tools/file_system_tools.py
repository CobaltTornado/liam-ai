import os
import logging
from pydantic import BaseModel, Field

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Workspace Setup ---
# Define a dedicated directory for the agent's file operations.
# This is a crucial security and stability measure.
WORKSPACE_DIR = "workspace"

# Ensure the workspace directory exists.
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def _get_safe_path(file_path: str) -> str:
    """
    Joins the provided file path with the workspace directory and resolves it to an absolute path.
    This prevents directory traversal attacks and ensures operations are contained.
    """
    # os.path.join is used to safely combine path components.
    # os.path.abspath resolves the path to an absolute one.
    # os.path.normpath cleans up the path (e.g., handles ".." or redundant separators).
    safe_path = os.path.normpath(os.path.abspath(os.path.join(WORKSPACE_DIR, file_path)))

    # Security Check: Ensure the resolved path is still inside the workspace.
    if not safe_path.startswith(os.path.abspath(WORKSPACE_DIR)):
        raise ValueError("Attempted to access a file outside of the designated workspace.")
    return safe_path

# --- Standalone Tool Functions ---

def create_file(file_path: str, content: str = "") -> dict:
    """
    Creates a new file with specified content at a given path within the workspace.
    """
    try:
        safe_path = _get_safe_path(file_path)
        logger.info(f"TOOL: Creating new file at {safe_path}")
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"status": "success", "message": f"Successfully created file: {file_path}"}
    except Exception as e:
        logger.error(f"Failed to create file: {e}")
        # Return a dictionary for consistent error handling
        return {"status": "error", "reason": f"Failed to create file at '{file_path}'. Reason: {e}"}

def read_file(file_path: str) -> dict:
    """
    Reads the content of a file from a given path within the workspace.
    """
    try:
        safe_path = _get_safe_path(file_path)
        logger.info(f"TOOL: Reading file from {safe_path}")
        if not os.path.exists(safe_path):
            return {"status": "error", "reason": f"File not found at '{file_path}'."}
        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"status": "success", "content": content}
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return {"status": "error", "reason": f"Failed to read file at '{file_path}'. Reason: {e}"}

def update_file(file_path: str, content: str) -> dict:
    """
    Updates (overwrites) the content of an existing file at a given path within the workspace.
    """
    try:
        safe_path = _get_safe_path(file_path)
        logger.info(f"TOOL: Updating file at {safe_path}")
        if not os.path.exists(safe_path):
            return {"status": "error", "reason": f"File not found at '{file_path}'. Cannot update."}
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"status": "success", "message": f"Successfully updated file: {file_path}"}
    except Exception as e:
        logger.error(f"Failed to update file: {e}")
        return {"status": "error", "reason": f"Failed to update file at '{file_path}'. Reason: {e}"}

def delete_file(file_path: str) -> dict:
    """
    Deletes a file from a given path within the workspace.
    """
    try:
        safe_path = _get_safe_path(file_path)
        logger.info(f"TOOL: Deleting file from {safe_path}")
        if not os.path.exists(safe_path):
            return {"status": "error", "reason": f"File not found at '{file_path}'. Cannot delete."}
        os.remove(safe_path)
        return {"status": "success", "message": f"Successfully deleted file: {file_path}"}
    except Exception as e:
        logger.error(f"Failed to delete file: {e}")
        return {"status": "error", "reason": f"Failed to delete file at '{file_path}'. Reason: {e}"}

def list_files(directory_path: str = ".") -> dict:
    """
    Lists all files and directories within a given path inside the workspace.
    """
    try:
        safe_path = _get_safe_path(directory_path)
        logger.info(f"TOOL: Listing files in directory {safe_path}")
        if not os.path.isdir(safe_path):
            return {"status": "error", "reason": f"Directory not found at '{directory_path}'."}
        entries = os.listdir(safe_path)
        if not entries:
            return {"status": "success", "content": f"Directory '{directory_path}' is empty."}
        return {"status": "success", "content": "\n".join(entries)}
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return {"status": "error", "reason": f"Failed to list files in directory '{directory_path}'. Reason: {e}"}

def get_fs_state_str(directory_path: str = ".") -> str:
    """
    Gets a string representation of the file system state for the agent's context.
    """
    result = list_files(directory_path)
    if result["status"] == "success":
        return result["content"]
    else:
        return result["reason"]

