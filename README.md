# MLOps_Group_50

## Goal

This repository contains the final project for the DTU course MLOps(02476). The goal is to create an end-to-end machine learning project that combines initial data exploration, model training, evaluation, deployment and monitoring using techniques from the field of Machine Learning Operations (MLOps). 

## Frameworks

TBD

## Data

The data for this project is sourced from [Kaggle: Emotion Detection](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data). The dataset contain 35,685 examples of 48x48 pixel gray scale images of faces. Images are categorized based on the emotion shown in the facial expressions (happiness, neutral, sadness, anger, surprise, disgust, fear).

The dataset is a collection of 35,685 images of 48x48 pixel grayscale images of faces. These faces are labeled, based on facial expressions, into the emotions happiness, neutral, sadness and anger.

## Models

TBD

## Project structure

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

