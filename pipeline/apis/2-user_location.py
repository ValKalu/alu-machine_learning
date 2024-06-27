#!/usr/bin/env python3
import requests  
import time  
import sys  
  
def get_user_location(api_url):  
    response = requests.get(api_url)  
    if response.status_code == 200:  
        user_data = response.json()  
        if 'location' in user_data:  
            print(user_data['location'])  
        else:  
            print("Not found")  
    elif response.status_code == 403:  
        reset_time = int(response.headers['X-Ratelimit-Reset']) - int(time.time())  
        print(f"Reset in {reset_time // 60} min")  
    else:  
        print("Error:", response.status_code)  
  
if __name__ == '__main__':  
    if len(sys.argv)!= 2:  
        print("Usage:./2-user_location.py ")  
        sys.exit(1)  
    get_user_location(sys.argv)  
