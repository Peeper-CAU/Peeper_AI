#!/bin/bash

set -m

flask run --host=0.0.0.0 &
streamlit run streamlit_demo.py --server.port 5001

fg %1