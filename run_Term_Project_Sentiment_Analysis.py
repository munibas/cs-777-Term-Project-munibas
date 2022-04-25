
from __future__ import print_function
import os
import sys
import requests
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.feature import ChiSqSelector
from pyspark.mllib.evaluation import MulticlassMetrics
import time

# Function to predict the sentiment(positive, negative) of text for different classifiers
def getPrediction(text, model):
    # Check if text is one review or a list of reviews
    if (isinstance(text, str)):
        review = [text]
    else:
        review = text

    # Create a dataframe of review list
    df_new_data = spark.createDataFrame(review, StringType()) 

    # Rename the dataframe column
    df_new_data = df_new_data.withColumnRenamed("value", "review")

    # Predict sentiment using the SVM model
    predict = model.transform(df_new_data)
    predict = predict.select("review", "prediction")
    return predict

# main() starts here
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    # Create spark context   
    spark = SparkSession \
        .builder \
        .appName("Sentiment Analysis of IMDB Dataset") \
        .getOrCreate()

    # Set your file path here 
    # Google cloud path
    data_file = "gs://muniba-met-cs-777/CS777-Term-Project-Sentiment-Analysis/IMDB_Dataset.csv"

    #data_file = "/Users/munibasiddiqi/Desktop/BUCS777/Homework_Assignments/Term_Project/IMDB_Dataset.csv"
    
    # Colab path
    #data_file = "IMDB_Dataset.csv"

    # Upload data into a dataframe
    spark_df = spark.read.format("csv").option("header", "true").option("escape","\"").option("multiLine","true").load(data_file)

    # Add label column to dataframe (Positive sentiment = 1, Negative sentiment = 0)
    df = spark_df.withColumn("label", F.when(F.col("sentiment")=="positive",1).otherwise(0)).cache()
    df.show(10)

    # Split the dataset into training and test set
    df_train, df_test = df.randomSplit(weights=[0.7, 0.3], seed=100)

    # Check if the dataset is balanced or imbalanced
    print("Taining Data")
    df_train.groupby("label").count().show()
    print("Test Data")
    df_test.groupby("label").count().show()

    ############################### Data Preprocessing #######################################
    
    # Converts text to lowercase and split text on non-word character
    regexTokenizer = RegexTokenizer(inputCol="review", outputCol="words", pattern="\\W")
    
    # Remove stopwords
    remover = StopWordsRemover(inputCol = regexTokenizer.getOutputCol(), outputCol="filtered")

    # Remove stopWords=["br", 'm', 've', 're', 'll', 'd']
    remover2 = StopWordsRemover(inputCol= remover.getOutputCol(), outputCol="token",stopWords=["br", 'm', 've', 're', 'll', 'd'])

    # Extracts a vocabulary from document collections and generates a CountVectorizerModel
    # During the fitting process, CountVectorizer will select the top vocabSize words ordered 
    # by term frequency across the corpus.
    countVectorizer = CountVectorizer(inputCol= remover2.getOutputCol(), outputCol="rawFeatures", vocabSize=5000)
    
    # The IDF Model takes feature vectors and scales each feature. 
    # Intuitively, it down-weights features which appear frequently in a corpus
    idf = IDF(inputCol= countVectorizer.getOutputCol(), outputCol="featuresIDF")

    # Chi-Squared feature selection. It operates on labeled data with categorical features. 
    # ChiSqSelector uses the Chi-Squared test of independence to decide which features to choose. 
    selector = ChiSqSelector(numTopFeatures=500, featuresCol=idf.getOutputCol(),
                         outputCol="features", labelCol="label")

############################### LOGISTIC REGRESSION ######################################

    ##################### Training Model ####################

    # Start time
    begin_time = time.time()

    # LogisticRegression classifier
    classifier_logreg = LogisticRegression(maxIter=20)

    # Chain indexers and classifier_logreg in a Pipeline
    pipeline_logreg = Pipeline(stages=[regexTokenizer, remover, remover2, countVectorizer, idf, selector, classifier_logreg])
    
    # Train model. 
    model_logreg = pipeline_logreg.fit(df_train)

    # Print the coefficients and intercept for linear SVC
    print("Logistic Regression Model")
    print("First 10 Coefficients: " + str(model_logreg.stages[6].coefficients[:10]))
    print("Intercept: " + str(model_logreg.stages[6].intercept))

    # Top 20 vocabulary words
    #pipeline_logreg.getStages()
    vocabulary = model_logreg.stages[3].vocabulary
    print("Top twenty vocabulary words", vocabulary[0:20])

    # End time
    end_time = time.time() - begin_time
    print("Total execution time to train logistic regression model on the train data: ", end_time)
   
    # Create a dataframe of top 20 vocabulary words to save as csv file
    df_top20 = spark.createDataFrame(vocabulary[0:20], StringType())

    # Store this result in a single file on the cluster
    df_top20.coalesce(1).write.format("csv").option("header",True).save(sys.argv[1]+'.top20_words_IDF')

    ##################### Model Testing ####################

    # Start time
    begin_time = time.time()

    # Make predictions.
    predictions_logreg = model_logreg.transform(df_test).cache()

    # End time
    end_time = time.time() - begin_time
    print("Total execution time to test logistic regression model on the test data: ", end_time)

    ##################### Model evaluation ####################

    # Start time
    begin_time = time.time()
    
    # Covert dataframe to RDD for Model evaluation
    predictionAndLabels_logreg = predictions_logreg.select("label",  "prediction").rdd.map(lambda x : (float(x[0]), float(x[1]))).cache()

    # Instantiate metrics object
    metrics_logreg = MulticlassMetrics(predictionAndLabels_logreg)

    # Statistics by class
    #labels = data.map(lambda lp: lp.label).distinct().collect()
    print("Summary statistics for Logistic regression classifier")
    labels = [0.0, 1.0]
    for label in sorted(labels):
        print("Class %s precision = %s" % (label, metrics_logreg.precision(label)))
        print("Class %s recall = %s" % (label, metrics_logreg.recall(label)))
        print("Class %s F1 Measure = %s" % (label, metrics_logreg.fMeasure(label, beta=1.0)))
    
    print("Accuracy = %s" % metrics_logreg.accuracy)
    print("Confusion Matrix")
    print(metrics_logreg.confusionMatrix().toArray().astype(int))

    # End time
    end_time = time.time() - begin_time
    print("Total execution time to evalute the performance of logistic regression model on test data: ", end_time)

    # Create a dataframe to store the summary of results
    data = [("Accuracy", str(metrics_logreg.accuracy)),("Confusion Matrix",str(metrics_logreg.confusionMatrix().toArray()))]
    df = spark.createDataFrame(data)

    # Store this result in a single file on the cluster
    df.coalesce(1).write.format("csv").option("header",True).save(sys.argv[1]+'.logreg_statistics')



 ############################### SUPPORT VECTOR MACHINE ######################################

    ##################### Training Model ####################

    # Start time
    begin_time = time.time()

    # SVM classifier
    classifier_lsvc = LinearSVC(maxIter=20)

    # Fit the model
    #lsvcModel = classifier.fit(df_train)

    # Chain indexers and classifier_lsvc in a Pipeline
    pipeline_lsvc = Pipeline(stages=[regexTokenizer, remover, remover2, countVectorizer, idf, selector, classifier_lsvc])

    # Train model. 
    model_lsvc = pipeline_lsvc.fit(df_train)

    # Print the coefficients and intercept for linear SVC
    print("Support Vector Machine Model")
    print("First 10 Coefficients: " + str(model_lsvc.stages[6].coefficients[:10]))
    print("Intercept: " + str(model_lsvc.stages[6].intercept))

    # End time
    end_time = time.time() - begin_time
    print("Total execution time to train SVM model on the train data: ", end_time)

    ##################### Model Testing ####################

    # Start time
    begin_time = time.time()

    # Make predictions.
    predictions_lsvc = model_lsvc.transform(df_test).cache()

    # End time
    end_time = time.time() - begin_time
    print("Total execution time to test SVM model on the test data: ", end_time)

    ##################### Model evaluation ####################

    # Start time
    begin_time = time.time()
    
    # Covert dataframe to RDD for Model evaluation
    predictionAndLabels_lsvc = predictions_lsvc.select("label",  "prediction").rdd.map(lambda x : (float(x[0]), float(x[1]))).cache()

    # Instantiate metrics object
    metrics_lsvc = MulticlassMetrics(predictionAndLabels_lsvc)


    # Statistics by class
    #labels = data.map(lambda lp: lp.label).distinct().collect()
    labels = [0.0, 1.0]
    for label in sorted(labels):
        print("Class %s precision = %s" % (label, metrics_lsvc.precision(label)))
        print("Class %s recall = %s" % (label, metrics_lsvc.recall(label)))
        print("Class %s F1 Measure = %s" % (label, metrics_lsvc.fMeasure(label, beta=1.0)))
    
    print("Accuracy = %s" % metrics_lsvc.accuracy)
    print(metrics_lsvc.confusionMatrix().toArray().astype(int))

    # End time
    end_time = time.time() - begin_time
    print("Total execution time to evalute the performance of SVM model on test data: ", end_time)

    # Create a dataframe to store the summary of results
    data = [("Accuracy", str(metrics_lsvc.accuracy)),("Confusion Matrix",str(metrics_lsvc.confusionMatrix().toArray()))]
    df = spark.createDataFrame(data)

    # Store this result in a single file on the cluster
    df.coalesce(1).write.format("csv").option("header",True).save(sys.argv[1]+'.SVM_statistics')


########################## Predicting Sentiments on New Data (Reviews)  ############################

   # A list of reviews 
    new_data = ['This movie was horrible, plot was boring, acting was okay.',
                'The film really sucked. I want my money back',
                'What a beautiful movie. Great plot, great acting.',
                'Harry Potter was a good movie.'
                ]

    ################# Prediction using logistic regression model ###############

    # Call to getPrediction function with a list of reviews and logistic regression model
    predict = getPrediction(new_data, model_logreg)
    print("Prediction using Logistic Regression model:")
    predict.show(truncate=0)

    # Store this result in a single file on the cluster
    predict.coalesce(1).write.format("csv").option("header",True).save(sys.argv[1]+'.logreg_prediction')

    ##################### Prediction using SVM model model ####################

    # Call to getPrediction function with a list of reviews and SVM model
    predict = getPrediction(new_data, model_lsvc)
    print("Prediction using SVM model:")
    predict.show(truncate=0)

    # Store this result in a single file on the cluster
    predict.coalesce(1).write.format("csv").option("header",True).save(sys.argv[1]+'.SVM_prediction')

    # Stop spark context   
    spark.stop()
