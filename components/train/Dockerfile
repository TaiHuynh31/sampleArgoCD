FROM python:3.10
WORKDIR /app
COPY train.py /app
RUN pip install tensorflow numpy kfp
ENTRYPOINT ["python", "train.py"]
