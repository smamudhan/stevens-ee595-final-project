#!/bin/bash
# Paste this in the runpod template
# Make sure the env vars are configured

export GITHUB_SSH_PRIVATE_KEY="GITHUB_DEPLOY_PRIVATE_KEY"
export GITHUB_BRANCH="branch_name"
export KAGGLE_USERNAME="smamudhan"
export KAGGLE_KEY="redacted"

# Setup GIT SSH Key
ssh-keyscan github.com >> ~/.ssh/known_hosts
touch /root/.ssh/id_rsa
echo "$GITHUB_SSH_PRIVATE_KEY" > /root/.ssh/id_rsa
chmod 600 /root/.ssh/id_rsa

# Install Tools
apt-get update
apt-get install -y git htop nvtop vim unzip wget tmux screen

# Clone Repo and Dataset
git clone --single-branch --branch "$GITHUB_BRANCH" git@github.com:smamudhan/stevens-ee595-final-project.git /workspace/repo
mv /workspace/dataset /workspace/repo/dataset

pip install --upgrade --no-cache-dir kaggle
kaggle datasets download nikhil2k3/artifact-customcurated-extras --force --unzip --path /workspace/repo/dataset/full
kaggle datasets download nikhil2k3/project-data --force --unzip --path /workspace/repo/dataset/reduced

# Install Python Deps
pip install --upgrade --no-cache-dir numpy matplotlib scikit-learn h5py opencv-python argparse kaggle tensorflow imbalanced-learn pillow 

echo "SETUP SCRIPT COMPLETE"
