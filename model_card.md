# Model Card

## Model Details

This is a Random forest classification model, v1, developed by Cooper Hepworth on 05/03/2024.

## Intended Use

This model is itended to predict a person's salary based on several features including age, workclass, education, marital-status, occupation, relationship, hours worked per week, and native country.

This model can be used by research firms, universities, or government bureacrats looking to understand salaries of individuals

## Training Data

This model was trained on 85% of the data found in census.csv

## Evaluation Data

This model was evaluated on 15% of the data found in census.csv

## Metrics

This model was evaluated on precision, recall, and F1. 

The model received the following scores:

    Precision: 0.7353

    Recall: 0.6344

    F1: 0.6751

## Ethical Considerations

The data used contains features including race, sex, country of origin, and relationship. To avoid discrimination, these features should not be included when predicting outcomes.

## Caveats and Recommendations

It is fair to say that the data is not complete. There are many other features that could be obtained to produce better observations and predictions of salary earned. The data also only contains 32,561 individuals which may not be a large enough sample size.

A random forest model also may not be the only model that is useful for predicting salary.