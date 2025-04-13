FROM python:3.8-slim
WORKDIR /app
COPY aggregator.py /app
RUN pip install flask numpy
CMD ["python", "aggregator.py"]
