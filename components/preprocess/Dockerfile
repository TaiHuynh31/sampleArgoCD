FROM python:3.10
WORKDIR /app
COPY preprocess.py /app
RUN pip install scikit-learn numpy kfp
ENTRYPOINT ["python", "preprocess.py"]
