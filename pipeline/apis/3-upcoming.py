#!/usr/bin/env python3
"""
Script to fetch and display the details of the upcoming SpaceX launch.
"""

import requests
from datetime import datetime, timezone


def fetch_upcoming_launch():
    """
    Fetches the upcoming launch details from SpaceX API.
    Returns:
        str: Formatted string with launch details.
    """
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    response = requests.get(url)
    launches = response.json()

    if not launches:
        return None
    
    # Sort launches by date_unix to get the soonest launch
    upcoming_launch = min(launches, key=lambda x: x['date_unix'])

    # Convert UTC time to local time
    launch_date_utc = datetime.fromisoformat(upcoming_launch['date_utc'].replace('Z', '+00:00'))
    launch_date_local = launch_date_utc.astimezone().isoformat()

    # Format output
    launch_name = upcoming_launch['name']
    rocket_name = upcoming_launch['rocket']
    launchpad_name = upcoming_launch['launchpad']

    launch_details = f"{launch_name} ({launch_date_local}) {rocket_name} - {launchpad_name['name']} ({launchpad_name['locality']})"
    return launch_details


if __name__ == '__main__':
    upcoming_launch = fetch_upcoming_launch()
    if upcoming_launch:
        print(upcoming_launch)
    else:
        print("No upcoming launches found.")
