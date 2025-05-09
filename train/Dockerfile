FROM python:3.10
WORKDIR /app
COPY wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt
COPY . .
CMD ["python", "train.py"]