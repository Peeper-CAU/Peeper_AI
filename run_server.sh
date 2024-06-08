#!/bin/bash

set -m

flask run &
streamlit run streamlit_demo.py --server.port 5001

fg %1