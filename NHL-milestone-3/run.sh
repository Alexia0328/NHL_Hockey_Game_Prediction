#!/bin/bash
docker run -it -p 3000:3000 --env COMET_API_KEY=$COMET_API_KEY --env LOG_FILE=./flask.log ift6758/serving:1.0.0
#docker run -it -p 3001:3001 ift6758/jupyter:1.0.0