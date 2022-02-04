FROM alpine:latest

WORKDIR /code

COPY requirements.txt .
COPY src/ .

RUN pip install -r requirements.txt
