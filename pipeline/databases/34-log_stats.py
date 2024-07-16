#!/usr/bin/env python3
"""
Provides some stats about Nginx logs stored in MongoDB
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def insert_test_data(logs_coll):
    """
    Inserts test data into the logs.nginx collection if it's empty.
    """
    if logs_coll.count_documents({}) == 0:
        test_data = [
            {"method": "GET", "path": "/"},
            {"method": "POST", "path": "/submit"},
            {"method": "GET", "path": "/status"},
            {"method": "PUT", "path": "/update"},
            {"method": "DELETE", "path": "/delete"},
            {"method": "GET", "path": "/home"},
            {"method": "GET", "path": "/about"},
            {"method": "POST", "path": "/login"},
            {"method": "PUT", "path": "/edit"},
            {"method": "DELETE", "path": "/remove"},
        ]
        logs_coll.insert_many(test_data)
        print("Inserted test data into the collection.")

if __name__ == "__main__":
    """
    Gets stats about Nginx logs stored in MongoDB
    """
    try:
        client = MongoClient('mongodb://127.0.0.1:27017', serverSelectionTimeoutMS=5000)
        client.server_info()  # Force connection on a request as the ping is lazily connected
        print("Successfully connected to MongoDB.")
    except ConnectionFailure:
        print("Failed to connect to MongoDB server.")
        exit(1)

    try:
        logs_coll = client.logs.nginx

        # Insert test data if collection is empty
        insert_test_data(logs_coll)

        doc_count = logs_coll.count_documents({})
        print("{} logs".format(doc_count))
        print("Methods:")
        methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        for method in methods:
            method_count = logs_coll.count_documents({"method": method})
            print("\tmethod {}: {}".format(method, method_count))

        filter_path = {"method": "GET", "path": "/status"}
        path_count = logs_coll.count_documents(filter_path)
        print("{} status check".format(path_count))

    except Exception as e:
        print(f"An error occurred: {e}")
