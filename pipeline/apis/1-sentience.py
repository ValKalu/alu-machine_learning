#!/usr/bin/env python3
"""
This module provides a function to fetch the names of the home planets of all 
sentient species from the Swapi API.
"""

import requests


def sentientPlanets():
    """
    Fetches the names of the home planets of all sentient species from the 
    Swapi API.

    Returns:
        list: A list of names of home planets of all sentient species.
    """
    url = 'https://swapi.dev/api/species/'
    planets = []

    while url:
        response = requests.get(url)
        data = response.json()
        for species in data['results']:
            if species['designation'] == 'sentient' and species['homeworld']:
                homeworld_response = requests.get(species['homeworld'])
                if homeworld_response.status_code == 200:
                    homeworld_data = homeworld_response.json()
                    planets.append(homeworld_data['name'])
                else:
                    planets.append('unknown')
                    print(f"Error fetching homeworld for species: {species['name']}")
                    print(f"Response status code: {homeworld_response.status_code}")
                    print(f"Response content: {homeworld_response.content}")
            else:
                planets.append('unknown')
        url = data['next']

    return planets


if __name__ == '__main__':
    planets = sentientPlanets()
    for planet in planets:
        print(planet)
