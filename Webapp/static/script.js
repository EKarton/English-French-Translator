"use strict";

$('document').ready(() => {
    class StreamableTranslator {
        constructor(onReceiveHandler, onErrorHandler) {
            this.dataToSend = null;
            this.inFlight = false;

            this.onReceiveHandler = onReceiveHandler;
            this.onErrorHandler = onErrorHandler;

            // this.host_url = "https://en-fr-translator.herokuapp.com"
            this.host_url = "http://localhost:5000"
        }

        runTranslation() {
            if (!this.inFlight && this.dataToSend != null) {

                let data = this.dataToSend;

                this.inFlight = true;
                this.dataToSend = null;

                console.log("Sending request to API: " + JSON.stringify(data));

                const url = this.host_url + "/api/v1/translate";

                axios.post(url, data)
                    .then(data => {
                        this.onReceiveHandler(data.data);
                        this.inFlight = false;
                        this.runTranslation();
                    })
                    .catch(error => {
                        this.onErrorHandler(error);
                        this.inFlight = false;
                        this.runTranslation();
                    });
            }
        }

        translate(text, source_lang, target_lang) {
            console.log(this.inFlight + " | " + JSON.stringify(this.postponedData) + " | " + this.text);
            this.dataToSend = {
                'input_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang
            };

            this.runTranslation();
        }

        getStatus(onSuccessHandler, onErrorHandler) {
            const url = this.host_url + "/api/v1/status";
            axios.get(url)
                .then(() => onSuccessHandler)
                .catch(error => {
                    onErrorHandler(error);
                });
        }
    }

    var sourceLangOption = $("#source-lang-dropdown")
    var targetLangOption = $("#target-lang-dropdown")
    var sourceTextbox = document.getElementById('source-text-txtbox')
    var targetTextbox = document.getElementById('target-text-txtbox');

    var translator = new StreamableTranslator(
        translatedText => {
            targetTextbox.value = translatedText;
        }, error => {
            console.error(error.response);
        }
    );

    translator.getStatus(() => {
        console.log("Success in contacting with server");
    }, error => {
        console.error("Error in getting status from server:");
        console.error(error);
    });

    $(".dropdown-item").click(function () {

        // Add the value of the dropdown item to the parent's data-value attribute
        var selectedOption = ($(this).data("value"));
        $(this).parents(".dropdown").data("value", selectedOption);

        // Change the text of dropdown to the selected option
        $(this).parents(".dropdown").find('.btn').html($(this).text() + '<span class="caret"></span>');
        $(this).parents(".dropdown").find('.btn').val($(this).data('value'));

        // Set the "active" classname to the selected dropdown item
        $(this).parents(".dropdown-menu").find('a').removeClass('active');
        $(this).addClass('active');
    });

    sourceTextbox.addEventListener('keyup', function (e) {
        let source_text = e.target.value;
        let source_lang = sourceLangOption.data("value");
        let target_lang = targetLangOption.data("value");

        // Remove the newline character
        source_text = source_text.replace(/(\r\n|\n|\r)/gm, "");

        // Replace all double white spaces with single spaces
        source_text = source_text.replace(/\s+/g, " ");

        if (source_text.length == 0) {
            targetTextbox.value = "";

        } else {
            console.log('Trying to translate: ' + source_text + ' from ' + source_lang + " to " + target_lang);
            translator.translate(source_text, source_lang, target_lang);
        }
    });
});