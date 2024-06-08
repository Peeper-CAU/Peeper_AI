#!/bin/bash

set -m

python3 app.py &
streamlit run streamlit_demo.py --server.port 5001 &

wait -n