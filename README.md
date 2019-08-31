# PySpark Resumes text classification

A text classification script using PySpark.
Inspired by https://towardsdatascience.com/multi-class-text-classification-with-pyspark-7d78d022ed35

## Getting Started

Get PySpark up and running. If you're using Windows, you can follow steps 1-4 from this link:

https://medium.com/@naomi.fridman/install-pyspark-to-run-on-jupyter-notebook-on-windows-4ec2009de21f

Read this tutorial:

https://towardsdatascience.com/multi-class-text-classification-with-pyspark-7d78d022ed35

## Resumes Classification

Some Italian resumes were downloaded from indeed, and divided them into folders. Then, five different models of classification were applied.

Models used:

- Logistic Regression with Count Vector Features
- Logistic Regression with TF-IDF
- Cross Validation
- Naïve Bayes
- Random Forest