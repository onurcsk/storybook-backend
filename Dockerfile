FROM python:3.10.6-buster
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY backend.py /backend.py
CMD uvicorn backend:app --host 0.0.0.0 --port $PORT