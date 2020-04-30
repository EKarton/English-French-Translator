FROM pytorch/pytorch:latest

ENV EN_FR_TRANSLATOR_ENDPOINT http://localhost:5001
ENV FR_EN_TRANSLATOR_ENDPOINT http://localhost:5002

WORKDIR /app

COPY ["./Translator/models/Hansard-Multi30k", "./Translator/models/Hansard-Multi30k"]

COPY ["./Translator/src/dataloader", "./Translator/src/dataloader"]
COPY ["./Translator/src/model", "./Translator/src/model"]

COPY ["./Webapp/static", "./Webapp/static"]
COPY ["./Webapp/templates", "./Webapp/templates"]
COPY ["./Webapp/app.py", "./Webapp/app.py"]
COPY ["./Webapp/models.py", "./Webapp/models.py"]
COPY ["./Webapp/requirements.txt", "./Webapp/requirements.txt"]

WORKDIR "/app/Webapp"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm

CMD python app.py
