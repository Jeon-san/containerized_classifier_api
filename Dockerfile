# Use an official Python runtime as a parent image.
FROM python:3.9

# Create an /code folder inside the container.
RUN mkdir /code

# Set the working directory inside the container to /code.
WORKDIR /code 

# Copy files from the current directory into the container's /code directory.
COPY ./requirements.txt /code
COPY ./main.py /code/main.py

# Install the packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the model and the clean data from the current directory into the container's /code directory.
COPY ./classifier_model.sav /code/classifier_model.sav
COPY ./clean_data.py /code/clean_data.py

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
