#!/usr/bin/env python3
"""
This script fetches and prints the location of a specific GitHub user using the GitHub API.
"""

import requests
import sys
import time


def get_user_location(url):
    """
    Fetches the location of a specific GitHub user.

    Args:
        url (str): The full API URL of the user.

    Returns:
        str: The location of the user or an appropriate message if not found or rate limited.
    """
    response = requests.get(url)
    
    if response.status_code == 404:
        return "Not found"
    elif response.status_code == 403:
        reset_time = int(response.headers.get('X-RateLimit-Reset', time.time()))
        wait_time = reset_time - int(time.time())
        return f"Reset in {wait_time // 60} min"
    elif response.status_code == 200:
        data = response.json()
        return data.get('location', 'Location not available')
    else:
        return "Error occurred"


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)
    
    user_url = sys.argv[1]
    location = get_user_location(user_url)
    print(location)
