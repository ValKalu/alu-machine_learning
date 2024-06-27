                           APIS

Requests Package Basics
The requests package in Python is a powerful tool for making HTTP requests. Here are the basics:

Making a Get request

import requests

response = requests.get('https://api.example.com/data')
print(response.status_code)
print(response.json())  # If the response is in JSON format


Handling Rate Limits

if response.status_code == 403 and 'X-Ratelimit-Reset' in response.headers:
    reset_time = int(response.headers['X-Ratelimit-Reset'])
    print(f"Reset in {reset_time} seconds")


Handling Pagination

def fetch_all_pages(url):
    results = []
    
    while url:
        response = requests.get(url)
        data = response.json()
        results.extend(data['results'])
        url = data['next']  # Assuming 'next' contains the URL for the next page
    return results
