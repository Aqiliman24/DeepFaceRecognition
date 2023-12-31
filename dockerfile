FROM python:3.10.6
WORKDIR /DeepLearningModelFaceImages

ENV AWS_ACCESS_KEY_ID_FACEDETECTION = AWS_ACCESS_KEY_ID_FACEDETECTION
ENV AWS_SECRET_ACCESS_KEY_FACEDETECTION = AWS_SECRET_ACCESS_KEY_FACEDETECTION
ENV AWS_REGION_FACEDETECTION=AWS_REGION_FACEDETECTION
ENV AWS_BUCKET_FACEDETECTION=AWS_BUCKET_FACEDETECTION
ENV PORT_FACEDETECTION = PORT_FACEDETECTION

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .
CMD ["python","app.py"]


