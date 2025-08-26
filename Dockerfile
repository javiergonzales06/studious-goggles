FROM ubuntu:20.04

RUN apt-get update && apt-get install -y curl unzip

COPY files/ /javier-files/

EXPOSE 80

CMD ["tail", "-f", "/dev/null"]