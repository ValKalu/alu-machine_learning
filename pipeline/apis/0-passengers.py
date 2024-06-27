#!/usr/bin/env python3
import requests
def availableShips(passangerCOunt):
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
                