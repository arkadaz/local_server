FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
WORKDIR /app
RUN apt-get update && apt-get install -y \
    python3-pip
COPY requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt
COPY . ./
CMD ["python3", "main.py"]