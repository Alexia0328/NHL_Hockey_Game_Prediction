#!/bin/bash
docker build -f Dockerfile.serving -t ift6758/serving:1.0.0 .
#docker build -f Dockerfile.jupyter -t ift6758/jupyter:1.0.0 .
#echo "TODO: fill in the docker build command"