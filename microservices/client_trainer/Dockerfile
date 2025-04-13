FROM python:3.8-slim
WORKDIR /app
COPY trainer.py /app
RUN pip install flask tensorflow
CMD ["python", "trainer.py"]
