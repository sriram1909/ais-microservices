FROM python:3.8-slim
WORKDIR /app
COPY explain_model.py /app
RUN pip install flask shap tensorflow
CMD ["python", "explain_model.py"]
