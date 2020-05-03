# English French Translator
## Description

The English French Translator is a web application that tries to make good translations between English and French sentences. It is equipped with two Transformer models - one for translating from English to French, and another from French to English - and was trained on the Hansard+Multi30K dataset. The model that translates from English to French achieved a BLEU score of 35.26, while the model that translates from French to English achieved a BLEU score of  35.20.

## Table of Contents
* Walkthrough
* Getting Started
* Usage
* Credits
* License

## Walkthrough

This project consists of several components, each responsible for performing a certain task to make good translations. The image below illustrates the system architecture of the project.

The web application looks like this:

<div width="100%">
    <p align="center">
<img src="Webapp/docs/Homepage.png" width="100%"/>
    </p>
</div>

It should detect the language and translate your text as you type.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
* Unix machine
* Python 3
* Pip 3

### Using Pre-Trained Models:
You can use pre-trained models instead of training one from scratch.
1. Go to https://drive.google.com/drive/folders/1C3PZYT0csQmDUoijvqnfr5w5wiZprDH_?usp=sharing
2. Navigate to the ```Hansard-Multi30k``` folder
3. Download its contents to your local machine at ```models/Hansard-Multi30k/```

### Training the model [Optional]:
Ignore this step if you used pre-trained models. There are two ways to train the model:

1. On Google Colab:

	This is the preferred way as there are free (limited) GPUs on Google Colab

	* ##### Setting up the environment

		1. Make the following directories from your root project directory by running the command:
			```
			mkdir models
			cd models
			mkdir Hansard
			mkdir Multi30k
			mkdir Hansard-Multi30k
			```

		2. Make a new Google Account to host this repo (it will be ~10gb large)
		3. Upload the entire project directory to Google Drive

	* ##### Running on Colab
		1. On Google Drive, navigate to the ```English-French-Translator/scripts``` folder
		2. Click on the ```Notebook - Hansard+Multi30k.ipymb``` file and open it in Google Colaboratory
		3. Follow the steps in the Notebook

2. On a local machine:

	* ##### Setting up the environment

		1. Create a virtual environment in the ```src``` directory by running the command:

			```
			cd src
			virtualenv -p python3.7 .
			source bin/activate
			```

		2. Install the dependencies by running the command:
			```
			pip3 install -r requirements.txt
			
			python3.7 -m spacy download en_core_web_sm
			python3.7 -m spacy download fr_core_news_sm
			```

		3. Make the following directories from your root project directory by running the command:
			```
			mkdir models
			cd models
			mkdir Hansard
			mkdir Multi30k
			mkdir Hansard-Multi30k
			```


	* ##### Building the vocabulary
	  In the root project directory, run the following command:
	  ```
	  sh scripts/build-vocabs.sh
	  ```


	* ##### Training the model
	  In the root project directory, run the following command:
	  ```
	  sh scripts/train.sh
	  ```

	* ##### Testing the model
	  In the root project directory, run the following command:
	  ```
	  sh scripts/test.sh
	  ```

### Running the Web App:

There are two ways in running the Web App:

1. Through manual setup of microservices:

	* ##### Getting the English to French microservice running:

		In a new Bash window, run the command:
		```
		cd src
		source bin/activate
		export PORT=5000
		python3.7 webapp/app.py
		```

	* ##### Getting the French to English microservice running:
		
		In a new Bash window, run the command:
		```
		cd src
		source bin/activate
		export PORT=5001
		python3.7 webapp/worker.py
		```

	* ##### Getting the front-facing microservice running:
		
		In a new Bash window, run the command:
		```
		cd src
		source bin/activate
		export PORT=5002
		python3.7 webapp/worker.py
		```

2. Through Docker Compose

	This is a preferred (but slower) way as it runs the app on an environment close to production

	* ##### Additional Pre-requisites

		You will need to have Docker installed on your local machine.

	* ##### Getting Docker-Compose running
		1. First, build the docker images by running the command:
			```
			docker-compose build
			```

		2. Next, run the docker images by running the command:
			```
			docker-compose up
			```

	* ##### Shutting down Docker-Compose
		Run the command:
		```
		docker-compose down
		```

## Experiments
Several experiments were conducted on the Multi30k dataset before using the Transformer as the model for the web app

Notebook links:
1. Experimenting NMT using RNNs: [Notebook](Translator/notebooks/NMT%20using%20RNNs.ipynb)
2. Experimenting NMT using RNNs with Attention: [Notebook](Translator/notebooks/NMT%20using%20RNNs%20with%20Attention.ipynb)
3. Experimenting NMT using Transformers: [Notebook](Translator/notebooks/NMT%20using%20Transformers.ipynb)
4. Experimenting Text Classification using Bag of Words: [Notebook](Language-Detector/notebooks/Text%20Classification%20using%20Bag%20of%20Words.ipynb)

Results:
|                         | NMT using RNNs | NMT using RNNs with Luong Attention | NMT using Transformers |
|-------------------------|----------------|-------------------------------------|------------------------|
| Dev Test Loss Score     | 2.029          | 1.989                               | 1.922                  |
| Dev Test Set BLEU Score | 24.14          | 27.24                               | 36.83                  |
| Test Set Loss Score     | 1.98           | 1.98                                | 1.574                  |
| Test Set BLEU Score     | 29.38          | 29.38                               | 42.94                  |

|                         | Text Classification using Bag of Words |
|-------------------------|----------------------------------------|
| Dev Test Loss Score     | 5.674499565238241e-10                  |
| Dev Test Set Accuracy   | 0.9931                                 |
| Test Set Loss Score     | 2.8081064123368606e-06                 |
| Test Set Accuracy       | 0.9941                                 |


## Deploying the Web App to Heroku
1. Login to Heroku with the command line arg:
	```
	heroku container:login
	```

1. Add these config (environment) variables in the ```fr-2-en-translator``` app:
	```
	SOURCE_LANG : fr
	TARGET_LANG : en
	```

2. Add these config (environment) variables in the ```en-2-fr-translator``` app:
	```
	SOURCE_LANG : en
	TARGET_LANG : fr
	```

3. Add these config (environment) variables in the ```en-fr-translator``` app:
	```
	EN_FR_TRANSLATOR_ENDPOINT : https://en-2-fr-translator.herokuapp.com
	FR_EN_TRANSLATOR_ENDPOINT : https://fr-2-en-translator.herokuapp.com
    LANGUAGE_DETECTOR_ENDPOINT : https://en-fr-language-detector.herokuapp.com
	```

4. Build, push, and deploy the Docker image to the ```fr-2-en-translator``` app:
	```
	docker build -t en2fr -f Translator-Webapi/Dockerfile .
	docker tag en2fr registry.heroku.com/en-2-fr-translator/web
	docker push registry.heroku.com/en-2-fr-translator/web
	heroku container:release web --app en-2-fr-translator
	```

5. Build, push, and deploy the Docker image to the ```en-2-fr-translator``` app:
	```
	docker build -t fr2en -f Translator-Webapi/Dockerfile .
	docker tag fr2en registry.heroku.com/fr-2-en-translator/web
	docker push registry.heroku.com/fr-2-en-translator/web
	heroku container:release web --app fr-2-en-translator
	```

6. Build, push, and deploy the Docker image to the ```en-fr-language-detector``` app:
	```
	docker build -t language-detector -f Language-Detector-Webapi/Dockerfile .
	docker tag language-detector registry.heroku.com/en-fr-language-detector/web
	docker push registry.heroku.com/en-fr-language-detector/web
	heroku container:release web --app en-fr-language-detector
	```

7. Build, push, and deploy the Docker image to the ```en-fr-translator``` app:
	```
	docker build -t en-fr -f Webapp/Dockerfile .
	docker tag en-fr registry.heroku.com/en-fr-translator/web
	docker push registry.heroku.com/en-fr-translator/web
	heroku container:release web --app en-fr-translator
	```

## Built With

* [PyTorch](https://pytorch.org/) - The machine learning framework used
* [Flask](https://flask.palletsprojects.com/en/1.1.x/) - The web framework used

## Authors

* Emilio Kartono

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* http://www.cs.toronto.edu/~frank/csc401/
* https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
* http://www.peterbloem.nl/blog/transformers
* https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
* https://homes.cs.washington.edu/~msap/notes/seq2seq-tricks.html
