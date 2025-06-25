import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime

# --- CONFIGURATION ---
# Paste the exact MongoDB URI you provided.
# NOTE: In a real application, never hardcode credentials. Use environment variables.
# This is only for a quick, local test script.
MONGO_URI = "mongodb+srv://98sh32:SHANNU@nlp.fsz4tmh.mongodb.net/?retryWrites=true&w=majority&appName=NLP"

# These should match the DB and Collection names from your app.py
DB_NAME = "llm_feedback_db"
COLLECTION_NAME = "dpo_feedback"


def check_mongo_connection():
    """
    Connects to MongoDB, inserts a single dummy document, and prints the result.
    """
    print("--- MongoDB Connection Test ---")
    print(f"Attempting to connect to MongoDB Atlas...")
    
    client = None  # Initialize client to None for the 'finally' block

    try:
        # 1. Create a new client and connect to the server
        client = MongoClient(MONGO_URI)

        # 2. The ismaster command is cheap and does not require auth. It forces a
        # connection to be established and will raise an error if it fails.
        # A more modern approach is to use ping.
        client.admin.command('ping')
        print("✅ Connection Successful: Ping to the database was successful.")

        # 3. Get the database and collection
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print(f"✅ Successfully accessed database '{DB_NAME}' and collection '{COLLECTION_NAME}'.")

        # 4. Create a dummy document to insert
        dummy_document = {
            "prompt": "Connection Test",
            "chosen": "Success",
            "rejected": "Failure",
            "timestamp": datetime.utcnow(),
            "model_version": "N/A - Connection Test"
        }
        
        print("\nAttempting to insert a dummy document...")
        print("Document to be inserted:", dummy_document)

        # 5. Insert the document
        result = collection.insert_one(dummy_document)

        print("\n--- ✅ SUCCESS! ---")
        print(f"Document inserted successfully into the '{COLLECTION_NAME}' collection.")
        print(f"Inserted Document ID: {result.inserted_id}")
        print("\nCheck your MongoDB Atlas dashboard to confirm the new document exists.")

    except ConnectionFailure as e:
        print("\n--- ❌ CONNECTION FAILED ---")
        print("Could not connect to MongoDB.")
        print("Please check the following:")
        print("  1. Your internet connection.")
        print("  2. The IP Access List in your MongoDB Atlas project settings. Is your current IP address allowed?")
        print("  3. The MONGO_URI string for any typos.")
        print(f"\nError details: {e}")

    except OperationFailure as e:
        print("\n--- ❌ OPERATION FAILED (Likely Authentication) ---")
        print("Connected to the server, but the operation failed.")
        print("This is often due to incorrect username or password.")
        print("Please double-check your credentials in the MONGO_URI.")
        print(f"\nError details: {e}")

    except Exception as e:
        print(f"\n--- ❌ AN UNEXPECTED ERROR OCCURRED ---")
        print(f"Error details: {e}")
        
    finally:
        # 6. Ensure the connection is closed
        if client:
            client.close()
            print("\nConnection closed.")

# --- Run the test ---
if __name__ == "__main__":
    check_mongo_connection()