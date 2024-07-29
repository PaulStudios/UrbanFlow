import json
import logging
import uuid

import pandas as pd
import requests

api_url = "http://localhost:8000"
project_id = 604722

def delete_project_data(project_id):
    """
    Delete all data for a given project.

    Args:
        project_id (int): The ID of the project.

    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    url = f"{api_url}/project/{str(project_id)}/delete_data"
    r = requests.delete(url)
    if r.status_code == 200:
        logging.info(f"Deleted all data for project ID {project_id}")
        return True
    logging.error(f"Failed to delete data for project ID {project_id}")
    logging.error(f"Status code: {r.status_code}")
    return False


def upload_user_data(project_id, user_id, upload_id, user_data):
    """
    Upload user data to a project.

    Args:
        project_id (int): The ID of the project.
        user_id (int): The ID of the user.
        upload_id (str): The ID of the upload batch.
        user_data (str): User data entries to be uploaded as a JSON string.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    url = f"{api_url}/project/{str(project_id)}/user_data"
    data = {
        "user_id": user_id,
        "upload_id": upload_id,
        "user_data": user_data
    }
    r = requests.post(url, data=data)
    if r.status_code == 200:
        logging.info(f"Uploaded data for project ID {project_id}, user ID {user_id}, upload ID {upload_id}")
        return True
    logging.error(f"Failed to upload data for project ID {project_id}, user ID {user_id}, upload ID {upload_id}")
    logging.error(f"Status code: {r.status_code}")
    return False


def run(csv_file):
    df = pd.read_csv(csv_file)
    if not delete_project_data(project_id):
        logging.error("Failed to delete project data. Exiting.")
        exit(1)

    # Group data by upload_id
    grouped_data = df.groupby('upload_id')

    # Reupload cleaned data under user ID 000000
    new_user_id = 000000
    for upload_id, group in grouped_data:
        # If the group has more than 25 entries, chunk it
        if len(group) > 25:
            chunks = [group[i:i + 25] for i in range(0, len(group), 25)]
            for chunk in chunks:
                user_data = {
                    "entries": chunk.to_dict(orient="records")
                }
                if not upload_user_data(project_id, new_user_id, upload_id, json.dumps(user_data)):
                        logging.error(f"Failed to upload chunk for upload_id {upload_id}")
            else:
                user_data = {
                    "entries": group.to_dict(orient="records")
                }
                if not upload_user_data(project_id, new_user_id, upload_id, json.dumps(user_data)):
                    logging.error(f"Failed to upload data for upload_id {upload_id}")

        logging.info("Cleaned data reuploaded successfully under user ID 000000")
    logging.info("Cleaned data reuploaded successfully under user ID 000000")