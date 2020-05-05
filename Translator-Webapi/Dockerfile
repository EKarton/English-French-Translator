FROM pytorch/pytorch:latest

ENV SOURCE_LANG en
ENV TARGET_LANG fr

WORKDIR /app

COPY ["./Translator/models/Hansard-Multi30k", "./Translator/models/Hansard-Multi30k"]
COPY ["./Translator/src/dataloader", "./Translator/src/dataloader"]
COPY ["./Translator/src/model", "./Translator/src/model"]

COPY ["./Translator-Webapi/app.py", "./Translator-Webapi/app.py"]
COPY ["./Translator-Webapi/models.py", "./Translator-Webapi/models.py"]
COPY ["./Translator-Webapi/requirements.txt", "./Translator-Webapi/requirements.txt"]

WORKDIR "/app/Translator-Webapi"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm

CMD python app.py --source-lang $SOURCE_LANG --target-lang $TARGET_LANG
