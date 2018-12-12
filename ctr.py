# imports
import re
import ast
import time
import numpy as np
import pandas as pd
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# start Spark Session
from pyspark.sql import SparkSession
app_name = "ctr_final_project"
master = "local[*]"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .master(master)\
        .getOrCreate()
sc = spark.sparkContext

FIELDS = ["label","I1","I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13", \
          "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15", \
          "C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26"]

# Parse the data, handle missing values and return an array of label & features
def parseDataPoint(line):
    values = line.replace('\n', '').split('\t')
    label = int(values[0])
    features = [label]

    # Replace missing integer features with 0.0
    for f in values[1:14]:
        if f == '':
            features.append(0.0)
        else:
            features.append(float(f))

    # Replace missing categorical features with new value 'missing'
    for f in values[14:]:
        if f == '':
            features.append('missing')
        else:
            features.append(f)

    return features

def parseTestDataPoint(line):
    values = line.replace('\n', '').split('\t')
    features = []

    # Replace missing integer features with 0.0
    for f in values[0:13]:
        if f == '':
            features.append(0.0)
        else:
            features.append(float(f))

    # Replace missing categorical features with new value 'missing'
    for f in values[13:]:
        if f == '':
            features.append('missing')
        else:
            features.append(f)

    return features

def initGraph(dataRDD, typ):
    if typ == 'train':
        graphRDD = dataRDD.map(parseDataPoint).cache()
    else:
        graphRDD = dataRDD.map(parseTestDataPoint).cache()

    return graphRDD

def runCTR(initRDD, maxIter, regParam):
    """
    Spark job to implement click through rate prediction
    Args:

    """
    SEED = 42
    df = spark.createDataFrame(initRDD, FIELDS)
    training, validation = df.randomSplit([0.75, 0.25], SEED)

    c9_indexer = StringIndexer(inputCol="C9", outputCol="C9Indexed")
    c17_indexer = StringIndexer(inputCol="C17", outputCol="C17Indexed")
    c20_indexer = StringIndexer(inputCol="C20", outputCol="C20Indexed")

    encoder1 = OneHotEncoder(inputCol="C9Indexed", outputCol="C9Encoded")
    encoder2 = OneHotEncoder(inputCol="C17Indexed", outputCol="C17Encoded")
    encoder3 = OneHotEncoder(inputCol="C20Indexed", outputCol="C20Encoded")

    assembler = VectorAssembler(inputCols=["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", \
                                           "I12", "I13", "C9Encoded", "C17Encoded", "C20Encoded"],
                                outputCol="features")

    lr = LogisticRegression(maxIter=maxIter, regParam=regParam)

    pipeline = Pipeline(stages=[c9_indexer, c17_indexer, c20_indexer, encoder1,
        encoder2, encoder3, assembler, lr])

    model = pipeline.fit(training)
    predictions = model.transform(validation)

    testError = predictions.select("label", "prediction").filter(predictions["label"] != predictions["prediction"]).count() / predictions.count()

    return testError, model


def evaluate(model, testRDD):
    testDF = spark.createDataFrame(testRDD, FIELDS[1:])
    predictions = model.transform(testDF)

    return predictions.select("prediction").collect()

if __name__ == '__main__':
    output = 'gs://w261-final-project-team-8/results/'
    ctrTrain = sc.textFile('gs://w261-final-project-team-8/data/train.txt')
    ctrTest = sc.textFile('gs://w261-final-project-team-8/data/test.txt')

    nIter = 5
    regParam = 0.01

    start = time.time()
    graphRDD = initGraph(ctrTrain, 'train')
    testRDD = initGraph(ctrTest, 'test')

    testError, model = runCTR(graphRDD, nIter, regParam)

    print(f'...trained {nIter} iterations in {time.time() - start} seconds.')

    model.write().overwrite().save(output)
    predictions = evaluate(model, testRDD)

    print(f'Test Error on 75-25 split: {testError}')

    print(predictions)
