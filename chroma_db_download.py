import os
from google.cloud import storage


def download_folder_from_gcs(bucket_name, gcs_folder, local_folder):
    """
    Downloads a folder and its contents from Google Cloud Storage to a local directory.

    Args:
    bucket_name (str): Name of the GCS bucket.
    gcs_folder (str): GCS folder path to download from.
    local_folder (str): Path to the local folder.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_folder)

    for blob in blobs:
        # Get the relative path in the bucket
        relative_path = os.path.relpath(blob.name, gcs_folder)

        # Construct the local path
        local_path = os.path.join(local_folder, relative_path)

        # Ensure the local directory exists
        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Download the blob to the local path
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}.")

if __name__ == "__main__":
    LOCAL_FOLDER = "raw_data/chroma_db"  # Replace with your local folder path
    BUCKET_NAME = "storybook-storage-1"       # Replace with your GCS bucket name
    GCS_FOLDER = "raw_data/chroma_db"                 # Replace with your desired GCS folder path

    download_folder_from_gcs(BUCKET_NAME, GCS_FOLDER, LOCAL_FOLDER)
