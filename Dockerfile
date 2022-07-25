FROM python:3.10-buster

RUN mkdir -p /classification_project
WORKDIR /classification_project

COPY . .

RUN pip install -e .

RUN python train.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]