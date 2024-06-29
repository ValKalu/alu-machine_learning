#!/usr/bin/env python3
"""
Return the list of names of the home
planets of all sentient species.
"""

import requests

def sentientPlanets():
    """
    Return the list of names of the home
    planets of all sentient species.
    """
    url = "https://swapi-api.alx-tools.com/api/species/"
    sentient_planets = []

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data["results"]:
            if species["classification"].lower() == "sentient" and species["homeworld"]:
                homeworld_url = species["homeworld"]
                homeworld_response = requests.get(homeworld_url)
                if homeworld_response.status_code == 200:
                    homeworld_data = homeworld_response.json()
                    if 'name' in homeworld_data:
                        sentient_planets.append(homeworld_data["name"])
                    else:
                        sentient_planets.append('unknown')
                else:
                    sentient_planets.append('unknown')

        url = data["next"]

    return sentient_planets

if __name__ == "__main__":
    planets = sentientPlanets()
    for planet in planets:
        print(planet)
