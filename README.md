# spotify-genre-classification
Machine learning models classifying Spotify songs into genres using audio features from a 50k-song dataset.
# Spotify Genre Classification

This project builds machine learning classification models to predict the genre of a song using Spotify audio features.

## Overview

The goal of this project is to classify songs into genres based on their audio and metadata features from a dataset of 50,000 songs.

Key challenges in this project include:

- handling missing values
- encoding categorical variables
- reducing dimensionality
- evaluating multi-class classification performance

## Dataset

The dataset contains 50k Spotify songs with features such as:

- acousticness
- danceability
- duration
- energy
- instrumentality
- liveness
- loudness
- speechiness
- tempo
- valence
- popularity

The target variable is the song's **genre**.

## Methods

The project includes:

- data preprocessing
- missing-value handling
- categorical encoding
- dimensionality reduction
- classification modeling

Models explored may include:

- Logistic Regression
- Random Forest
- SVM
- XGBoost

## Results

The project evaluates classification performance using AUC and multi-class classification metrics.

## Tools

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Project Structure

spotify-genre-classification/
├── data/
├── notebooks/
├── model/
├── app/
├── requirements.txt
└── README.md
