# setup_vector_db.py

import os
import json
import pandas as pd
from google.cloud import storage
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from google.cloud import aiplatform
from langchain_community.document_loaders import DataFrameLoader

def setup_vector_db():
    # Set up the client
    storage_client = storage.Client()

    # Get the bucket
    bucket_name = 'storybook-storage-1'
    bucket = storage_client.bucket(bucket_name)

    # Create empty list and loop through the 10 json files
    data = []
    for i in range(3): # taking just 3 out of 10
        blob_name_i = f'Children-Stories-Collection/Children-Stories-{i}-Final.json'
        blob_i = bucket.blob(blob_name_i)
        elevations_i = blob_i.download_as_string()
        data_i = json.loads(elevations_i)
        data.append(data_i)

    flattened_data = [item for sublist in data for item in sublist]
    df = pd.DataFrame(flattened_data)

    # Ensure environment variables are set
    assert os.environ["GOOGLE_PROJECT_ID"], "GOOGLE_PROJECT_ID not set"
    assert os.environ["GOOGLE_PROJECT_REGION"], "GOOGLE_PROJECT_REGION not set"
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"], "GOOGLE_APPLICATION_CREDENTIALS not set"

    # Initialize Vertex AI
    aiplatform.init(project=os.environ["GOOGLE_PROJECT_ID"], location=os.environ["GOOGLE_PROJECT_REGION"])

    # Initialize embeddings
    embeddings = VertexAIEmbeddings(project=os.environ["GOOGLE_PROJECT_ID"], model_name="textembedding-gecko@003")

    # Ensure 'prompt' column exists in df
    assert 'text' in df.columns, "'text' column not found in DataFrame"

    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()

    # Create vector database
    vector_db = Chroma.from_documents(documents, embeddings, persist_directory="./raw_data/chroma_db")

    # Return the vector_db for further use
    return vector_db

if __name__ == "__main__":
    vector_db = setup_vector_db()
    print("Vector database setup completed.")
