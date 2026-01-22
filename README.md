# MLOps project on Emotion Detection

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)

## Goal

This repository contains the final project for the DTU course MLOps (02476). The goal is to create an end-to-end machine learning project that combines initial data exploration, model training, evaluation, deployment and monitoring using techniques from the field of Machine Learning Operations (MLOps).

## Repository layout

Our repository contains both the course report material and the actual MLOps project code.

- The actual MLOps project (source code, tests, CI, Dockerfiles, configs and so on) is located in the `MLOps_project/` directory.
- The repository root contains the course report, checklist and written exam answers required for the final hand-in.

The `MLOps_project/` directory was generated using the official Cookiecutter template for the course.

## Frameworks

We have chosen torchvision and TIMM as our frameworks for this project, where torchvision will be used for data handling, and TIMM for model architectures and pretrained weights.

## Data

The data for this project is sourced from [Kaggle: Emotion Detection](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data). The dataset contains 35,685 examples of 48x48 pixel gray scale images of faces. Images are categorized based on the emotion shown in the facial expressions (happiness, neutral, sadness, anger, surprise, disgust, fear).

## Models

Image classification tasks will be performed on the emotion detection dataset using a pretrained ResNet model from the TIMM library.

## Project structure (inside `MLOps_project/`)

The following directory structure corresponds to the contents of the `MLOps_project/` folder:

```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Exam checklist

[Exam template for 02476 Machine Learning Operations](https://github.com/DitteGilsfeldt/MLOps_Group_50/tree/main/MLOps_project/reports)

## Project diagram

Below is a brief overview of the overall MLOps project structure and workflow:

![diagram](https://github.com/DitteGilsfeldt/MLOps_Group_50/tree/main/MLOps_project/reports/figures/MLOps_diagram.png)

This diagram is also found under the exam checklist as an answer to question 29.

---