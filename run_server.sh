#!/bin/bash

set -m

python3 app.py &
streamlit run streamlit_demo.py --server.port 5001

fg %1