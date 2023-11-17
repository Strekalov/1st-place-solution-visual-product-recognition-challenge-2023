FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN apt-get update && apt-get install libturbojpeg


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -U git+https://github.com/albumentations-team/albumentations --no-binary qudida,albumentations
RUN pip install timm --pre -U
