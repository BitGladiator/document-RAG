#!/bin/sh

echo "Starting Flask..."

PYTHONPATH=/workspace/src \
python -m flask \
--app document_rag.app \
run \
--host=0.0.0.0 \
--port=5500 &

echo "Starting Jupyter..."

exec jupyter lab \
--ip=0.0.0.0 \
--port=8888 \
--no-browser \
--allow-root
