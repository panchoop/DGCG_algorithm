# set base image (host OS)
FROM python:3.6

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

# copy the content of the local src directory to the working directory
COPY . .

# command to run on container start
ENTRYPOINT ["python3"]
CMD [ "main.py"]
