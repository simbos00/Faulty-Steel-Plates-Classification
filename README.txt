# Faulty Steel Plates Classification

## Overview
We fit different classification models (Logistic Regression, LDA, QDA, NB and KNN) to find the best supervised learning model to classify faulty steel plates into minor or major faults. The code is written in R.

## Dataset
The dataset `faults.csv` contains features extracted from steel plates and their corresponding fault types.
The source of the dataset is Kaggle: https://www.kaggle.com/datasets/uciml/faulty-steel-plates

## Requirements
- R
- R libraries: tidyverse, magrittr, corrplot, modelsummary, dplyr, margins, ROCR, MASS, caret, car, e1071, class.

## How to Run
1. Clone this repository.
2. Open the 'script.R' file in RStudio or any R environment.
3. Install the required packages.
4. Run the script.

## Results
The model achieved an accuracy of almost 70% in classifying minor and major faults using the second Logistic Regression Model. 
Thanks to this model we were also able to confirm that using a softer steel (a300) rather than a400 steel is associated with an higher probability of major fault.