FROM python:3.8-slim
WORKDIR /app
COPY updater.py /app
RUN pip install flask tensorflow numpy
CMD ["python", "updater.py"]
