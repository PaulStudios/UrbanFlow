import logging

import pandas as pd
import requests
from haversine import haversine, Unit

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
    logging.error(f"Status code: {r.status_code}")
    return []


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
        # Dictionary to store unique entries by timestamp
        unique_data = {}

        # Iterate over the data and store entries with unique timestamps
        for entry in data_list:
            timestamp = entry['latest_timestamp']
            if timestamp not in unique_data:
                unique_data[timestamp] = entry

        # Convert the dictionary back to a list
        filtered_data = list(unique_data.values())
        data_list = filtered_data

        link_list = []
        for data in data_list:
            data_id = data["upload_id"]
            url = f"{api_url}/project/{str(project_id)}/get_data/{str(user_id)}/{str(data_id)}"
            link_list.append(url)
        logging.info(f"Retrieved data links for user ID {user_id}")
        return link_list
    logging.warning(f"Failed to retrieve data links for user ID {user_id}")
    logging.error(f"Status code: {r.status_code}")
    return []


def calculate_distance(coords1, coords2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface using the Haversine formula.

    Parameters:
    - coords1 (tuple): Latitude and longitude of point 1 in decimal degrees (lat, lon).
    - coords2 (tuple): Latitude and longitude of point 2 in decimal degrees (lat, lon).

    Returns:
    - distance (float): Distance between the two points in meters.
    """
    return haversine(coords1, coords2, unit=Unit.METERS)


def filter_standing_still(df, threshold=5):
    """
    Filter out datasets where movement is negligible (standing still) based on distance thresholds.
    If any point in an upload_id set is below the threshold, the entire set is removed.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing columns: longitude, latitude, timestamp, upload_id.
    - threshold (float): Minimum distance threshold in meters to consider movement significant.
                        Defaults to 5 meters.

    Returns:
    - filtered_df (pd.DataFrame): Filtered DataFrame containing rows where movement is significant.
    Raises:
    - RuntimeError: If an unexpected error occurs during filtering.
    """
    logging.info("Filtering standing still data...")

    try:
        # Ensure columns are named correctly (adjust if necessary)
        required_columns = ['longitude', 'latitude', 'timestamp', 'upload_id']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                "Input DataFrame does not contain required columns: longitude, latitude, timestamp, upload_id")

        # Sort dataframe by upload_id and timestamp
        df.sort_values(by=['upload_id', 'timestamp'], inplace=True)

        # Calculate distances between consecutive points for each upload_id
        df['next_latitude'] = df.groupby('upload_id')['latitude'].shift(-1)
        df['next_longitude'] = df.groupby('upload_id')['longitude'].shift(-1)

        def calculate_distance_row(row):
            """
            Calculate distance between current and next coordinates in a dataframe row.

            Parameters:
            - row (pd.Series): Row containing latitude, longitude, next_latitude, next_longitude.

            Returns:
            - bool: True if distance is above threshold, False otherwise.
            """
            coords1 = (row['latitude'], row['longitude'])
            coords2 = (row['next_latitude'], row['next_longitude'])
            dist = calculate_distance(coords1, coords2) if pd.notnull(row['next_latitude']) else None
            return dist >= threshold if dist is not None else False

        # Apply function to calculate distance for each row
        df['validity'] = df.apply(calculate_distance_row, axis=1)

        # Filter rows where movement is significant (validity is True)
        valid_df = df[df['validity'] == True]

        # Filter upload_id groups with more than 5 rows
        filtered_df = valid_df.groupby('upload_id').filter(lambda x: len(x) > 5)

        # Drop temporary columns
        filtered_df.drop(['next_latitude', 'next_longitude', 'validity'], axis=1, inplace=True)

        logging.info("Data filtered successfully")
        logging.info(f"Rows before filtering: {len(df)}, after filtering: {len(filtered_df)}")

        return filtered_df

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise RuntimeError(f"Error occurred: {str(e)}")


def clean_data(df):
    """
    Perform data cleaning operations on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df.dropna(inplace=True)  # Remove rows with missing values
    df.drop("data_id", axis="columns", inplace=True)
    df.drop("id", axis="columns", inplace=True)
    df.drop("user_id", axis="columns", inplace=True)
    df.drop("project_id", axis="columns", inplace=True)
    df.drop_duplicates(inplace=True)  # Remove duplicates
    clean_df = filter_standing_still(df)
    logging.info("Data cleaning complete")
    return clean_df


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
            logging.error(f"Status code: {response.status_code}")
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
