#!/bin/bash

PORT=8889
VENV_PATH=".venv"

echo "Launching interactive CPU node..."

srun --nodes=1 \
     --qos=interactive \
     --time=04:00:00 \
     --constraint=cpu \
     -A desi  \
     --pty bash -i -c "
echo 'On node:'
hostname
module load python
source $VENV_PATH/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
"