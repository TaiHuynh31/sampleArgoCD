FROM python:3.10
WORKDIR /app
COPY evaluate.py /app
RUN pip install tensorflow numpy scikit-learn kfp
ENTRYPOINT ["python", "evaluate.py"]
