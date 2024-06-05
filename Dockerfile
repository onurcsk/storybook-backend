# Use the official Python image from the Docker Hub with Python version 3.10 based on the Bookworm release.
FROM python:3.10-bookworm

# Copy the requirements.txt file from your local machine to the Docker image.
COPY requirements.txt /requirements.txt

# Upgrade pip to the latest version.
RUN pip install --upgrade pip

# Install the Python packages specified in the requirements.txt file.
RUN pip install -r requirements.txt

# Install the pysqlite3-binary package separately for use with Chroma
RUN pip install pysqlite3-binary

# Copy the chroma_db_download.py script to the Docker image
COPY chroma_db_download.py /chroma_db_download.py

# Run the chroma_db_download.py script to download and set up the vector database for the Retrieval-augmented generation (RAG)
# using a database of chidlren stories from Huggingface: https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection
RUN python chroma_db_download.py

# Copy the backend.py script to the Docker image.
COPY backend.py /backend.py

# Set the command to run the backend application using uvicorn.
CMD uvicorn backend:app --host 0.0.0.0 --port $PORT
