#!/usr/bin/env python3
"""
Script to fetch and display information about the upcoming SpaceX launch.
"""

import requests
from datetime import datetime


def upcoming_launch():
    """
    Fetches upcoming SpaceX launch information from the API.

    Returns:
        str: Formatted string containing launch information.
    """
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)
    launches = response.json()

    if not launches:
        return None

    # Sort launches by date_unix and select the soonest one
    upcoming_launch = min(launches, key=lambda launch: launch['date_unix'])

    # Extract relevant information
    launch_name = upcoming_launch['name']
    utc_date = upcoming_launch['date_utc']
    rocket_name = upcoming_launch['rocket']['name']
    launchpad_name = upcoming_launch['launchpad']['name']
    launchpad_locality = upcoming_launch['launchpad']['locality']

    # Convert UTC date to local time
    date_utc = datetime.fromisoformat(utc_date[:-1])  # Remove last 'Z' character
    date_local = date_utc.strftime('%Y-%m-%dT%H:%M:%S')

    # Format output
    output = "{} ({}) {} - {} ({})".format(launch_name, date_local, rocket_name, launchpad_name, launchpad_locality)
    return output


if __name__ == '__main__':
    upcoming_info = upcoming_launch()
    if upcoming_info:
        print(upcoming_info)
