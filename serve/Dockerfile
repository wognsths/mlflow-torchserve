FROM pytorch/torchserve:latest

WORKDIR /app
COPY . .

ENV PYTHONPATH="/app"

RUN pip install -r requirements.txt

ARG RUN_ID
RUN python make_mar.py ${run_id}

RUN python make_mar.py ${RUN_ID}
CMD ["torchserve","--start","--model-store","/home/model-server/model-store","--models","mnist=mnist.mar","--ncs"]