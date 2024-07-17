import requests
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_url = "https://glowworm-known-raven.ngrok-free.app"
project_id = 604722


def check_project_id(project_id):
    """
    Check if the given project ID exists and corresponds to 'UrbanFlow'.

    Args:
        project_id (int): The ID of the project to check.

    Returns:
        bool: True if the project exists and is 'UrbanFlow', False otherwise.
    """
    url = f"{api_url}/project/{str(project_id)}"
    r = requests.get(url)
    try:
        if r.status_code == 200 and "UrbanFlow" in r.json()["project_name"]:
            logging.info(f"Project ID {project_id} is valid")
            return True
    except KeyError:
        pass
    logging.error(f"Project ID {project_id} not found or does not correspond to 'UrbanFlow'")
    return False


def check_api_status():
    """
    Check the status of the API.

    Returns:
        bool: True if the API status is OK, False otherwise.
    """
    url = f"{api_url}/status"
    r = requests.get(url)
    if r.status_code == 200 and "OK" in r.text:
        logging.info("API is up")
        return True
    logging.error("API is down")
    return False


def get_user_list(project_id):
    """
    Retrieve the list of users for the given project ID.

    Args:
        project_id (int): The ID of the project.

    Returns:
        list: A list of users for the project.
    """
    url = f"{api_url}/project/{str(project_id)}/get_data"
    r = requests.get(url)
    if r.status_code == 200:
        logging.info(f"Retrieved user list for project ID {project_id}")
        return r.json()
    logging.error(f"Failed to retrieve user list for project ID {project_id}")
    return None


def get_data_links(user_id, project_id):
    """
    Retrieve data links for a given user ID and project ID.

    Args:
        user_id (int): The ID of the user.
        project_id (int): The ID of the project.

    Returns:
        list: A list of data links for the user.
    """
    url = f"{api_url}/project/{str(project_id)}/get_data/{str(user_id)}"
    r = requests.get(url)
    if r.status_code == 200:
        data_list = r.json()
        link_list = []
        for data in data_list:
            data_id = data["upload_id"]
            url = f"{api_url}/project/{str(project_id)}/get_data/{str(user_id)}/{str(data_id)}"
            link_list.append(url)
        logging.info(f"Retrieved data links for user ID {user_id}")
        return link_list
    logging.warning(f"Failed to retrieve data links for user ID {user_id}")
    return []


def clean_data(df):
    """
    Perform basic data cleaning operations on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df.dropna(inplace=True)  # Remove rows with missing values
    df.drop_duplicates(inplace=True)  # Remove duplicates
    logging.info("Data cleaning complete")
    return df


def run():
    if not check_api_status():
        exit(1)

    if not check_project_id(project_id):
        exit(1)

    user_list = get_user_list(project_id)
    if user_list is None:
        exit(1)

    data_links = []
    for user in user_list:
        user_id = user["user_id"]
        data_links.extend(get_data_links(user_id, project_id))

    if not data_links:
        logging.error("No data links found.")
        exit(1)

    logging.info(f"Retrieved {len(data_links)} data links")

    data_set = []
    for i, link in enumerate(data_links, 1):
        response = requests.get(link)
        if response.status_code == 200:
            data_set.extend(response.json())
        else:
            logging.warning(f"Failed to retrieve data from {link}")
        logging.info(f"Progress: {i}/{len(data_links)} links processed")

    if not data_set:
        logging.error("No data retrieved from the links.")
        exit(1)

    df = pd.DataFrame(data_set)

    df = clean_data(df)

    df.to_csv('final_data.csv', index=False)
    logging.info("Data saved to final_data.csv")


if __name__ == "__main__":
    run()
