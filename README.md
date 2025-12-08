# ML Automation Prototype

A lightweight, tabular-only ML automation tool built with Streamlit and Scikit-learn.

## Features
- **Data Upload**: Support for CSV and Parquet.
- **Preprocessing**: Configurable drop, imputation, scaling, and encoding.
- **Modeling**: Classification, Regression, and Clustering with Scikit-learn.
- **Iterative**: Train multiple models, compare results.
- **DevOps**: Dockerized, CI/CD with GitHub Actions.
- **Git Integration**: Push artifacts directly to a new branch.

## Quick Start via Docker

1.  **Clone the repository** (if not already):
    ```bash
    git clone <repo-url>
    cd <repo-name>
    ```

2.  **Run with Docker Compose**:
    ```bash
    ./demo_run.sh
    # OR
    docker-compose up --build
    ```

3.  **Access the App**:
    Open [http://localhost:8501](http://localhost:8501) in your browser.

## Local Development Setup

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the app:
    ```bash
    streamlit run app.py
    ```

4.  Run tests:
    ```bash
    pytest tests/
    ```

## Technology Stack Justification

-   **Streamlit**: Chosen for rapid prototyping and ease of creating data-centric UIs without complex frontend logic.
-   **Scikit-learn**: Standard, robust library for classical ML algorithms (Classification, Regression, Clustering).
-   **Pandas**: Essential for tabular data manipulation.
-   **GitHub Actions**: Integrated CI/CD for automated testing and container building.
-   **Docker**: Ensures reproducibility across environments.

## CI/CD Pipeline

-   **CI**: Runs on every push to `main` and PRs. Executes `ruff` (linting), `pytest` (unit tests), and builds the Docker image.
-   **CD**: Runs on tag pushes (e.g., `v1.0.0`). Builds and pushes the Docker image to GitHub Container Registry (ghcr.io).
