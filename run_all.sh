#!/bin/bash

# Define ports
MODELEND_PORT=8091
BACKEND_PORT=8299
FRONTEND_PORT=8080

# Log directory
LOG_DIR="logs"

# Ensure the log directory exists
mkdir -p $LOG_DIR

start_services() {
    echo "Starting services..."

    for PORT in $MODELEND_PORT $BACKEND_PORT $FRONTEND_PORT; do
        PID=$(lsof -t -i:$PORT)
        if [ ! -z "$PID" ]; then
            echo "Port $PORT is in use, killing process $PID..."
            kill -9 $PID
        else
            echo "Port $PORT is free."
        fi
    done

    echo "Starting modelend on $MODELEND_PORT!"
    nohup python modelend/run_agent.py > $LOG_DIR/modelend.log 2>&1 &
    echo "Modelend started."

    echo "Starting backend on $BACKEND_PORT!"
    nohup python backend/run.py > $LOG_DIR/backend.log 2>&1 &
    echo "Backend started."

    echo "Starting frontend on $FRONTEND_PORT!"
    nohup streamlit run frontend/run.py --server.port $FRONTEND_PORT --server.headless true > $LOG_DIR/frontend.log 2>&1 &
    echo "Frontend started."

    echo "All services started."
}

stop_services() {
    echo "Stopping services..."

    for PORT in $MODELEND_PORT $BACKEND_PORT $FRONTEND_PORT; do
        PID=$(lsof -t -i:$PORT)
        if [ ! -z "$PID" ]; then
            echo "Killing process $PID on port $PORT..."
            kill -9 $PID
        else
            echo "Port $PORT is free."
        fi
    done

    echo "All services stopped."
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 {start|stop}"
    exit 1
fi

status=$1

if [ "$status" == "start" ]; then
    start_services
elif [ "$status" == "stop" ]; then
    stop_services
else
    echo "Usage: $0 {start|stop}"
    exit 1
fi