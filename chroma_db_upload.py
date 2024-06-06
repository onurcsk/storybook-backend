import os
from google.cloud import storage

def upload_folder_to_gcs(local_folder, bucket_name, gcs_folder):
    """
    Uploads a folder and its contents to Google Cloud Storage.

    Args:
    local_folder (str): Path to the local folder.
    bucket_name (str): Name of the GCS bucket.
    gcs_folder (str): GCS folder path to upload to.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, dirs, files in os.walk(local_folder):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_folder)
            gcs_path = os.path.join(gcs_folder, relative_path)

            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)

            print(f"File {local_path} uploaded to {gcs_path}.")

if __name__ == "__main__":
    LOCAL_FOLDER = "raw_data/chroma_db"  # Replace with your local folder path
    BUCKET_NAME = "storybook-storage-1"       # Replace with your GCS bucket name
    GCS_FOLDER = "raw_data/chroma_db"                 # Replace with your desired GCS folder path

    upload_folder_to_gcs(LOCAL_FOLDER, BUCKET_NAME, GCS_FOLDER)
