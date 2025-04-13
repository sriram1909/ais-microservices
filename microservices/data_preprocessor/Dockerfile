FROM python:3.8-slim
WORKDIR /app
COPY preprocessor.py /app
RUN pip install flask pandas
CMD ["python", "preprocessor.py"]
