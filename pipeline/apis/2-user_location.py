#!/usr/bin/env python3
"""
This script fetches and prints the location of a specific GitHub user
"""


import requests
import time
import datetime import datetime


def main(url):
    """
    Fetches the location of a specific GitHub user.

    Args:
        url (str): The full API URL of the user.



    Returns:
        str: The location of the user or an appropriate message if not
        found or rate limited.
    """
    response = requests.get(url)

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_timestamp = int(response.headers["X-RateLimit-Reset"])
        current_timestamp = int(time.time())
        reset_in_minutes = (reset_timestamp - current_timestamp) // 60
        print("Reset in {} min".format(reset_in_minutes))
    else:
        print(response.json()["location"])
        
        
        if __name__ == "__main__":
            import sys
            
            main(sys.argv[1])