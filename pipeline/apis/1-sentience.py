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

        if 'results' not in data:
            break

        for species in data['results']:
            if species.get('designation') == 'sentient' and species.get('homeworld'):
                homeworld_response = requests.get(species['homeworld'])
                if homeworld_response.status_code == 200:
                    homeworld_data = homeworld_response.json()
                    planets.append(homeworld_data['name'])
                else:
                    print("Error fetching homeworld for species: {}".format(species['name']))
            elif species.get('designation') == 'sentient' and not species.get('homeworld'):
                planets.append(species['name'])

        url = data.get('next')

    return planets


if __name__ == '__main__':
    planets = sentientPlanets()
    for planet in planets:
        print(planet)
