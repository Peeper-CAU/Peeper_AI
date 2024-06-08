FROM python:3.10.6

ENV TZ Asia/Seoul

WORKDIR /app
ADD ./ /app

RUN python3 -m pip install -r requirements.txt
RUN apt update
RUN apt install -y portaudio19-dev

EXPOSE 5000
EXPOSE 5001

ENTRYPOINT ["sh", "run_server.sh"]