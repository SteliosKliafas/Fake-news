# syntax=docker/dockerfile:1

FROM python:3.10.6

RUN mkdir -p /home/app/src/
RUN mkdir -p /home/app/models/
WORKDIR home/app/

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY models/ models/ 
COPY src/ src/
COPY . .

# Expose the port on which the application will run
EXPOSE 8084

CMD [ "python3" , "app.py"]