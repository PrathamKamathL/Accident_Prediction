# Accident Prediction System

This project predicts accident severity based on input features using a machine learning model. It provides a REST API using FastAPI and is containerized using Docker with CI/CD deployment on AWS EC2.

---

## 1. Input and Output Specification

a. Input: Details of the accident with 18 features
b. Output: Multiclass classification of severity:

* Fatal injury – class 0
* Serious injury – class 1
* Slight injury – class 2

---

## 2. Dataset Source

https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents
Referenced by: https://data.mendeley.com/datasets/xytv86278f/1

---

## 3. Steps to Run Application

a. Fork the repository
b. Clone the repository

---

## Setup Instructions (Local)

### 1. Clone the repository

```
git clone https://github.com/PrathamKamathL/Accident_Prediction.git
cd Accident_Prediction
```

---

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Run FastAPI Locally

```
python -m uvicorn app.app:app --reload
```

Open in browser:

```
http://localhost:8000/docs
```

---

## Run Using Docker

### Build image

```
docker build -t fsml-img .
```

### Run container

```
docker run -p 80:8000 fsml-img
```

Access API:

```
http://localhost/docs
```

---

## CI/CD Pipeline

This project uses GitHub Actions for CI/CD.

Continuous Integration:

* Builds Docker image
* Pushes image to Docker Hub

Continuous Deployment:

* Connects to AWS EC2 using SSH
* Pulls latest Docker image
* Runs container automatically

---

## AWS EC2 Deployment Setup

### 1. Launch EC2 Instance

* Instance type: t2.micro
* OS: Ubuntu
* Allow ports:

  * 22 (SSH)
  * 80 (HTTP)

---

### 2. Install Docker on EC2

```
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu
newgrp docker
```

---

### 3. Create and Attach Elastic IP

* Go to EC2 → Elastic IPs
* Allocate new Elastic IP
* Associate with your instance

---

### 4. GitHub Actions Secrets Configuration

Go to:

Repository → Settings → Secrets and variables → Actions

Add the following secrets:

```
DOCKER_USERNAME = your_dockerhub_username
DOCKERHUB_TOKEN = your_dockerhub_access_token
EC2_IP = your_ec2_public_or_elastic_ip
EC2_USER = ubuntu
EC2_KEY = contents_of_your_.pem_file
```

Notes:

* EC2_KEY must include full private key with BEGIN and END lines
* Use Docker access token, not password

---

### 5. Deployment Process

On every push to main branch:

1. Docker image is built
2. Image is pushed to Docker Hub
3. GitHub Actions connects to EC2
4. Existing container is stopped and removed
5. Latest image is pulled and run

---

## Access Deployed API

```
http://<EC2-IP>/docs
```

---


---

## Logging

Logs are stored in:

```
logs/app.log
```

---

## Model

* Model file: models/model_v2.pkl
* Type: Multiclass classification model

---
