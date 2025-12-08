import os
import shutil
import joblib
import json
from datetime import datetime
from git import Repo, Actor
import matplotlib.pyplot as plt

def generate_run_summary(results, summary_path):
    """Writes a text summary of the run."""
    with open(summary_path, 'w') as f:
        f.write("ML Automation Run Summary\n")
        f.write("=========================\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        
        for model_name, res in results.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"Best Params: {res.get('best_params', 'N/A')}\n")
            f.write(f"Metrics: {res['metrics']}\n")
            f.write("-" * 20 + "\n")

def save_artifacts(results, artifact_dir):
    """Saves models, plots, and summary."""
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Save Summary
    generate_run_summary(results, os.path.join(artifact_dir, 'run_summary.txt'))
    
    # Save Models
    for model_name, res in results.items():
        safe_name = model_name.replace(" ", "_").lower()
        joblib.dump(res['model'], os.path.join(artifact_dir, f"{safe_name}.pkl"))

    return artifact_dir

def push_to_github(results, repo_name="ml-prototype", commit_msg="Auto-generated run"):
    """
    Pushes artifacts to a new branch in the user's repo.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    branch_name = f"prototype/ml-{timestamp}"
    artifact_dir = f"artifacts/{timestamp}"
    
    try:
        # Assume we are in the repo root or provided path
        repo_path = os.getcwd() 
        repo = Repo(repo_path)
        
        # Check if remote exists
        if 'origin' not in repo.remotes:
            return None # Can't push without remote
            
        # Create new branch
        current = repo.active_branch
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()
        
        # Save artifacts
        save_artifacts(results, artifact_dir)
        
        # Add and Commit
        repo.index.add([artifact_dir])
        repo.index.commit(commit_msg)
        
        # Push
        # Note: This relies on the environment having credentials (ssh or credential helper)
        origin = repo.remote(name='origin')
        origin.push(branch_name)
        
        # Return PR link (simulated)
        remote_url = origin.url
        if remote_url.endswith('.git'):
            remote_url = remote_url[:-4]
        
        # Determine base URL for PR
        # Supports simple github.com urls
        pr_link = f"{remote_url}/compare/main...{branch_name}?expand=1"
        
        # Switch back to main to avoid state issues in app
        current.checkout()
        
        return pr_link
        
    except Exception as e:
        print(f"Git Error: {e}")
        return None
