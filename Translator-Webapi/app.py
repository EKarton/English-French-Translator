import argparse
from os import environ

from flask import Flask, escape, request, render_template

from models import Translator

translator = None

app = Flask(__name__)


@app.route("/api/v1/translate", methods=["POST"])
def get_translation():
    global translator

    json_body = request.get_json()

    if "input_text" not in json_body:
        return "Must have input text", 400

    text = json_body["input_text"]

    if len(text) == 0:
        return "Must have a character in the input text", 400

    print(f"input_text={text}")

    translated_text = translator.translate_text(text)

    print(f"translated_text={translated_text}")
    return translated_text, 200


@app.route("/api/v1/status", methods=["GET"])
def get_status():
    return "ok", 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translator Web Api")
    parser.add_argument("--source-lang", type=str, help="Source language")
    parser.add_argument("--target-lang", type=str, help="Target language")

    opts = parser.parse_args()

    if opts.source_lang == "en" and opts.target_lang == "fr":
        print("Making english to french translator")
        translator = Translator(
            opts.source_lang,
            opts.target_lang,
            "../Translator/models/Hansard-Multi30k/vocab.inf.5.english.gz",
            "../Translator/models/Hansard-Multi30k/vocab.inf.5.french.gz",
            "../Translator/models/Hansard-Multi30k/model.en.fr.pt",
        )

    elif opts.source_lang == "fr" and opts.target_lang == "en":
        print("Making french to english translator")
        translator = Translator(
            opts.source_lang,
            opts.target_lang,
            "../Translator/models/Hansard-Multi30k/vocab.inf.5.french.gz",
            "../Translator/models/Hansard-Multi30k/vocab.inf.5.english.gz",
            "../Translator/models/Hansard-Multi30k/model.fr.en.pt",
            num_encoder_layers=2,
            num_decoder_layers=2,
        )

    else:
        raise ValueError(
            f"Cannot handle translation from {opts.source_lang} to {opts.target_lang}!"
        )

    app.run(host="0.0.0.0", port=environ.get("PORT", 5001))
