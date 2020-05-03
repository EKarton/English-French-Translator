FROM pytorch/pytorch:latest

WORKDIR /app

COPY ["./Language-Detector/models/Hansard-Multi30k", "./Language-Detector/models/Hansard-Multi30k"]
COPY ["./Language-Detector/src/dataloader", "./Language-Detector/src/dataloader"]
COPY ["./Language-Detector/src/model", "./Language-Detector/src/model"]

COPY ["./Language-Detector-Webapi/app.py", "./Language-Detector-Webapi/app.py"]
COPY ["./Language-Detector-Webapi/models.py", "./Language-Detector-Webapi/models.py"]
COPY ["./Language-Detector-Webapi/requirements.txt", "./Language-Detector-Webapi/requirements.txt"]

WORKDIR "/app/Language-Detector-Webapi"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python3 -m nltk.downloader punkt

CMD python app.py
