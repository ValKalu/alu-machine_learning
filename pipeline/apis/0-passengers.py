#!/usr/bin/env python3
"""
This module provides a function to fetch available starships that can hold a given number of passengers from the Swapi API.
"""

import requests
def availableShips(passangerCOunt):
    """
    Fetches a list of ships that can hold a given number of passengers from the Swapi API.

    Args:
        passengerCount (int): The minimum number of passengers the ships should be able to hold.

    Returns:
        list: A list of ship names that can hold the given number of passengers.
    """
    url = 'https://swapi.dev/api/starships/'
    ships = []
    
    while url:
        response = requests.get(url)
        data =response.json()
        for ship in data['results']:
            if ship['passangers'] != 'n/a' and int(ship['passagners'].replace(',', '')) >= passengerCount:
                ships.append(ship['name'])
                url = data['next']
                
                return ships
                