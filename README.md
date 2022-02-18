# containerized_classifier_api
A simple classifier model (random forest) API deployed using docker

## About each file:
* **Dockerfile** --- docker file used to generate a docker image that will run the API
* **classifier_model.sav** --- a pickled random forest model object used to predict probability of default
* **clean_data.py** --- python script to process the API input data from requests, the output will be fed to the random forest to make predictions
* **model.py** --- script that was used to train the random forest model
* **main.py** --- FastAPI script with a simple GET method
* **test_main.py** --- pytest unit test script (run the command "pytest" while in this directory to test API)
* **requirements.txt** --- list of all dependencies used to build the docker image

## How to deploy this API 
### Clone the directory
Go to the directory where you want to create your project and run:

```bash
$ git clone https://github.com/Jeon-san/containerized_classifier_api
```

### Generate docker image
cd to the project folder and run the following command:

```bash
docker build -t [name you want your image to be called] .
```
##### Note: If you get the error "Get "https://registry-1.docker.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)" try to follow the solutions outlined here https://stackoverflow.com/questions/48056365/error-get-https-registry-1-docker-io-v2-net-http-request-canceled-while-b

### Run the docker image:
If you are using docker for windows, go to the images tab and check if the image has successfully been created:

![image](https://user-images.githubusercontent.com/77715245/154616561-e87489b9-e739-4c11-97f8-fc5dce93a5f7.png)

Once it is there, go to the containers/apps tab and run the image:

![image](https://user-images.githubusercontent.com/77715245/154616763-84dddfd4-8583-4830-b55b-6d8bf6ad9f05.png)

Alternatively, if you are using docker on linux, run the following command:
```bash
# Check whether image was generated
docker images
# Run docker image
docker run [name of image]
```

### Check whether the API is running by 


