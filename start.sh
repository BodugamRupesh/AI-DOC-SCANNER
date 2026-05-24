#!/bin/bash
# Start FastAPI backend in the background on localhost
uvicorn app.main:app --host 127.0.0.1 --port 8000 &

# Start Streamlit frontend in the foreground on Render's public port
streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
