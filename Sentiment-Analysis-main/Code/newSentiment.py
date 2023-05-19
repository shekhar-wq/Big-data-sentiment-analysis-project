from pyspark.sql import SparkSession
import json
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from nltk.sentiment import SentimentIntensityAnalyzer
import sys

def main():
    SentimentAnalyser = SentimentIntensityAnalyzer()
    def classify(tweet):
        score = SentimentAnalyser.polarity_scores(tweet)
        if score['compound'] > 0:
            return 'positive'
        elif score['compound'] < 0:
            return 'negative'
        else:
            return 'neutral'

    topic_name = sys.argv[1]
    spark = SparkSession.builder.appName("StreamingTest")\
        .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.11:2.2.0')\
            .getOrCreate()
            
    df = spark.readStream \
        .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
                .option("subscribe", topic_name) \
                    .option("failOnDataLoss", "false")\
                        .option("startingOffsets", "earliest")\
                            .load()

    values = df.selectExpr("CAST(value AS STRING)")
    sentimentClassification = F.udf(classify, StringType())

    sentiment_tweets = values.withColumn("sentiment", sentimentClassification(values.value))

    sentiment_tweets.writeStream \
    .outputMode("append") \
    .queryName("writing_to_es") \
    .format("org.elasticsearch.spark.sql") \
    .option("checkpointLocation", "/tmp/") \
    .option("es.resource", "tweet_sentiment") \
    .start().awaitTermination()

if __name__=='__main__':
    length = len(sys.argv)

    if length != 2:
        print()
        print("Usuage : <script.py> <topic>")
        print()
        exit(1)
    
    main()
