FROM python:3.11-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 build-essential -y

COPY handler.py ./
COPY requirements.txt ./
COPY pose_landmarker.task ./
COPY image.jpg ./
COPY Makefile ./
COPY pthread_shim.c ./
RUN make pthread_shim.so && cp pthread_shim.so /opt

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD ["handler.handler"]
