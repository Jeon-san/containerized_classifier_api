# containerized_classifier_api
A simple classifier model (random forest trained in python using sklearn) API deployed using docker

## About each file:
* **Dockerfile** --- docker file used to generate a docker image that will run the API
* **classifier_model.sav** --- a pickled random forest model object used to predict probability of default
* **clean_data.py** --- python script to process the API input data from requests, the output will be fed to the random forest to make predictions
* **model.py** --- script that was used to train the random forest model
* **main.py** --- FastAPI script with a simple GET method
* **test_main.py** --- pytest unit test script (run the command "pytest" while in this directory to the test API)
* **requirements.txt** --- list of all dependencies used to build the docker image

## How to deploy this API 

### Pre-requisite
Make sure that docker is installed on your machine

### Clone the directory
Go to the directory where you want to create your project and run:

```bash
$ git clone https://github.com/Jeon-san/containerized_classifier_api
```

### Generate docker image
cd to the project folder, start docker in terminal, and run the following command:

```bash
$ docker build -t [name you want your image to be called] .
```
##### Note: If you get the error "Get "https://registry-1.docker.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)" try to follow the solutions outlined here https://stackoverflow.com/questions/48056365/error-get-https-registry-1-docker-io-v2-net-http-request-canceled-while-b

### Run the docker image:
Run the following commands:
```bash
# Check whether image was generated
$ docker images
# Run docker image
$ docker run -d -p 8000:8000 [image name]
```

### Check whether the API is running by using the command below, you should be able to see a container and its details:
```bash
$ docker container ls
```

## Using the API
### Location of API
The API can be accessed at http://127.0.0.1:8000/predict/ 

### Querying the API
A sample of a request URL is:
http://127.0.0.1:8000/predict/?contract=Month-to-month&dependents=Yes&deviceprotection=Yes&gender=Male&internetservice=Fiber%20optic&multiplelines=Yes&onlinebackup=Yes&onlinesecurity=Yes&paperlessbilling=Yes&partner=Yes&paymentmethod=Electronic%20check&phoneservice=Yes&seniorcitizen=Yes&streamingmovies=Yes&streamingtv=Yes&techsupport=Yes&tenure=3&monthlycharges=35&totalcharges=686

### Documentation
Documentation of the query variables could be found at **http://127.0.0.1:8000/docs**

### Model performance
The model performance is shown below:

![image](https://user-images.githubusercontent.com/77715245/154695649-dd8b6b8f-7b61-43d1-88d2-7e61dd5fd679.png)  

The confusion matrix:

![image](https://user-images.githubusercontent.com/77715245/154695714-9ac10ca2-0900-4704-9eeb-47be2a97ea07.png)



## Unit testing
In order to test weather the API is working correctly, install pytest if you haven't already and run the command in the project directory:
```bash
$ pytest
```


