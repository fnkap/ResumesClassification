# -*- coding: utf-8 -*-
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics

import pandas as pd
import seaborn as sn

import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

sc = SparkContext()
sqlContext = SQLContext(sc)

sc.setLogLevel("ERROR")


def clean(string):
    """Return the document without last line. If document is empty, returns a empty string."""
    if (string is None) or (string == ""):
        return ""
    temp = string.split("\r\r\n")
    temp = temp[:-1]
    temp2 = ' '.join(temp)
    return temp2


print("\n\n\n")
print("###########################")
print("#####Script.py started#####")
print("###########################")
print("\n\n")

# Loading of text files that contain resumes
rdd = sc.wholeTextFiles('myresumes/*/*', use_unicode=True)

# Create a RDD ('Category', 'Text of the resume')
mapped_rdd = rdd.map(lambda x: (str(x[0].split('/')[-2].strip()), clean(x[1])))

before_cleaning = int(mapped_rdd.count())

# Filter out empty documents
mapped_rdd = mapped_rdd.filter(lambda x: x[1] is not '')

after_cleaning_empty = int(mapped_rdd.count())

# Filter out duplicates
mapped_rdd = mapped_rdd.distinct()

after_cleaning_duplicates = int(mapped_rdd.count())

print("Found resumes:\t\t%d" % before_cleaning)
print("After cleaning empty documents:\t%s" % after_cleaning_empty)
print("After cleaning duplicates:\t%s" % after_cleaning_duplicates)

print("\n")

schema = ["Categoria", "CV"]

# Create a dataframe from RDD using columns defined in variable schema
df = sqlContext.createDataFrame(mapped_rdd, schema)

# Regular Expression Tokenizer
# We extract tokens from input column CV and we put them into output column words.
# We're diving words by using Regex \W which selects non-words characters.
regexTokenizer = RegexTokenizer(inputCol="CV", outputCol="words", pattern="\W")

# Stop Words
# We found a list of italian stopwords here: https://github.com/stopwords-iso/stopwords-it/blob/master/stopwords-it.txt
# We load the file into a list
add_stopwords = []
with open("stopwords.txt", "r") as f:
    for line in f:
        add_stopwords.append(line.strip())

    # We remove stop words from column words and we output the result into column filtered
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)

# Extracts a vocabulary from document collections and generates a CountVectorizerModel.
# minDF: Specifies the minimum number of different documents a term must appear in to be included in the vocabulary.
# if this is an integer >= 1, this specifies the number of documents the term must appear in;
# CountVectorizer Example:
# +-----+---------------+-------------------------+
# |label|raw            |vectors                  |
# +-----+---------------+-------------------------+
# |0    |[a, b, c]      |(3,[0,1,2],[1.0,1.0,1.0])|
# |1    |[a, b, b, c, a]|(3,[0,1,2],[2.0,2.0,1.0])|
# +-----+---------------+-------------------------+
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

# A label indexer that maps a string column of labels to an ML column of label indices.
# If the input column is numeric, we cast it to string and index the string values.
# The indices are in [0, numLabels).
# By default, this is ordered by label frequencies so the most frequent label gets index 0.
# The ordering behavior is controlled by setting stringOrderType. Its default value is ‘frequencyDesc’.
label_stringIdx = StringIndexer(inputCol="Categoria", outputCol="label")

# A Pipeline consists of a sequence of stages
# When Pipeline.fit() is called, the stages are executed in order.
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])

# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(df)
dataset = pipelineFit.transform(df)
# dataset.show(n=1005)


print("\n")

# Set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(("Training Dataset Count: " + str(trainingData.count())))
print(("Test Dataset Count: " + str(testData.count())))

print("\n")
print("=" * 40)
print("Running Logistic Regression using Count Vector Features. Please wait.")
start = time.time()

# elasticNetParam corresponds to α and regParam corresponds to λ.
lr = LogisticRegression(maxIter=20, regParam=0.1, elasticNetParam=0.2)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)

# Evaluate the results
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
end = time.time()
print("Accuracy:\t\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))
print("Weighted Precision:\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})))
print("Weighted Recall:\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})))
print("F1 Measure:\t\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "f1"})))
print("Time to process:\t" + str(end - start))

# Metrics
predictionRDD = predictions.select(['label', 'prediction']) \
    .rdd.map(lambda line: (line[1], line[0]))
metrics = MulticlassMetrics(predictionRDD)

# Confusion Matrix
print((metrics.confusionMatrix().toArray()))

df_cm = pd.DataFrame(metrics.confusionMatrix().toArray(), index=[i for i in "ABCDEFGHIJKLMNO"],
                     columns=[i for i in "ABCDEFGHIJKLMNO"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.savefig('myfig_1.png')

print("\n")
print("=" * 40)
print("Running Logistic Regression using TF-IDF Features. Please wait.")
start = time.time()

# Maps a sequence of terms to their term frequencies using the hashing trick. 
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
# Compute the Inverse Document Frequency (IDF) given a collection of documents.
# minDocFreq: Minimum number of documents in which a term should appear for filtering'
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx])
pipelineFit = pipeline.fit(df)
dataset = pipelineFit.transform(df)
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
lr = LogisticRegression(maxIter=20, regParam=0.1, elasticNetParam=0.2)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
end = time.time()
print("Accuracy:\t\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))
print("Weighted Precision:\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})))
print("Weighted Recall:\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})))
print("F1 Measure:\t\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "f1"})))
print("Time to process:\t" + str(end - start))

# Metrics
predictionRDD = predictions.select(['label', 'prediction']) \
    .rdd.map(lambda line: (line[1], line[0]))
metrics = MulticlassMetrics(predictionRDD)

# Confusion Matrix
print((metrics.confusionMatrix().toArray()))

df_cm = pd.DataFrame(metrics.confusionMatrix().toArray(), index=[i for i in "ABCDEFGHIJKLMNO"],
                     columns=[i for i in "ABCDEFGHIJKLMNO"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.savefig('myfig_2.png')

print("\n")
print("=" * 40)
print("Running Cross-Validation. Please wait.")
start = time.time()

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
pipelineFit = pipeline.fit(df)
dataset = pipelineFit.transform(df)
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)

# Create ParamGrid for Cross Validation to test various parameters
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.3, 0.5])  # regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2])  # Elastic Net Parameter (Ridge = 0)
             #            .addGrid(model.maxIter, [10, 20, 50]) #Number of iterations
             #            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
cvModel = cv.fit(trainingData)

predictions = cvModel.transform(testData)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
end = time.time()
print("Accuracy:\t\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))
print("Weighted Precision:\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})))
print("Weighted Recall:\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})))
print("F1 Measure:\t\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "f1"})))
print("Time to process:\t" + str(end - start))

best_model = cvModel.bestModel
best_reg_param = best_model._java_obj.getRegParam()
best_elasticnet_param = best_model._java_obj.getElasticNetParam()

print("Best regularization parameter found:\t" + str(best_reg_param))
print("Best elasticnet parameter found:\t" + str(best_elasticnet_param))

# Metrics
predictionRDD = predictions.select(['label', 'prediction']) \
    .rdd.map(lambda line: (line[1], line[0]))
metrics = MulticlassMetrics(predictionRDD)

# Confusion Matrix
print((metrics.confusionMatrix().toArray()))

df_cm = pd.DataFrame(metrics.confusionMatrix().toArray(), index=[i for i in "ABCDEFGHIJKLMNO"],
                     columns=[i for i in "ABCDEFGHIJKLMNO"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.savefig('myfig_3.png')

print("\n")
print("=" * 40)
print("Running Naive Bayes. Please wait.")
start = time.time()

# Naive Bayes Classifiers.
nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData)
predictions = model.transform(testData)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
end = time.time()
print("Accuracy:\t\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))
print("Weighted Precision:\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})))
print("Weighted Recall:\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})))
print("F1 Measure:\t\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "f1"})))
print("Time to process:\t" + str(end - start))

# Metrics
predictionRDD = predictions.select(['label', 'prediction']) \
    .rdd.map(lambda line: (line[1], line[0]))
metrics = MulticlassMetrics(predictionRDD)

# Confusion Matrix
print((metrics.confusionMatrix().toArray()))

df_cm = pd.DataFrame(metrics.confusionMatrix().toArray(), index=[i for i in "ABCDEFGHIJKLMNO"],
                     columns=[i for i in "ABCDEFGHIJKLMNO"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.savefig('myfig_4.png')

print("\n")
print("=" * 40)
print("Running Random Forest Classifier. Please wait.")
start = time.time()

rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees=100, \
                            maxDepth=4, \
                            maxBins=32)
# Train model with Training Data
rfModel = rf.fit(trainingData)
predictions = rfModel.transform(testData)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
end = time.time()
print("Accuracy:\t\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))
print("Weighted Precision:\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})))
print("Weighted Recall:\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})))
print("F1 Measure:\t\t" + str(evaluator.evaluate(predictions, {evaluator.metricName: "f1"})))
print("Time to process:\t" + str(end - start))

# Metrics
predictionRDD = predictions.select(['label', 'prediction']) \
    .rdd.map(lambda line: (line[1], line[0]))
metrics = MulticlassMetrics(predictionRDD)

# Confusion Matrix
print((metrics.confusionMatrix().toArray()))

df_cm = pd.DataFrame(metrics.confusionMatrix().toArray(), index=[i for i in "ABCDEFGHIJKLMNO"],
                     columns=[i for i in "ABCDEFGHIJKLMNO"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.savefig('myfig_5.png')
