#!/bin/bash
echo "Building and starting ML Prototype..."
docker-compose up --build -d

echo "Waiting for service to start..."
sleep 5

echo "App running at http://localhost:8501"
echo "To stop: docker-compose down"
