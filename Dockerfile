FROM python:3.10-bookworm
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pysqlite3-binary
# # Set build argument for Google Cloud authentication
# ENV GOOGLE_APPLICATION_CREDENTIALS=credentials.json
# COPY raw_data/credentials.json /credentials.json
COPY chroma_db_download.py /chroma_db_download.py
RUN python chroma_db_download.py
COPY backend.py /backend.py
CMD uvicorn backend:app --host 0.0.0.0 --port $PORT
