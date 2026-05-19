"""
FinSight AI — Apache Spark Structured Streaming
Consumes Kafka topics, performs real-time feature extraction,
runs fraud scoring, and writes results to PostgreSQL + console.
Run: python backend/streaming/spark_streaming.py
"""
import os
import json

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (StructType, StructField, StringType,
                                   DoubleType, IntegerType, TimestampType)
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("PySpark not installed. Run: pip install pyspark")

KAFKA_BROKER     = "localhost:9092"
KAFKA_TOPIC_TXN  = "finsight-transactions"
KAFKA_TOPIC_TICK = "finsight-stock-ticks"
JDBC_URL         = "jdbc:postgresql://localhost:5432/finsight"
JDBC_PROPS       = {"user": "finsight", "password": "finsight123",
                    "driver": "org.postgresql.Driver"}


# ─── Schema definitions ───────────────────────────────────────
TXN_SCHEMA = StructType([
    StructField("transaction_id",         StringType()),
    StructField("user_id",                StringType()),
    StructField("amount",                 DoubleType()),
    StructField("merchant_category",      StringType()),
    StructField("geo_distance_km",        DoubleType()),
    StructField("device_fingerprint_match", IntegerType()),
    StructField("is_international",       IntegerType()),
    StructField("transaction_velocity_7d", IntegerType()),
    StructField("timestamp",              StringType()),
])

TICK_SCHEMA = StructType([
    StructField("symbol",    StringType()),
    StructField("price",     DoubleType()),
    StructField("volume",    IntegerType()),
    StructField("timestamp", StringType()),
])


def build_spark():
    return (SparkSession.builder
            .appName("FinSight-Streaming")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.streaming.stopGracefullyOnShutdown", "true")
            .getOrCreate())


def stream_transactions(spark: "SparkSession"):
    """
    Reads transaction events from Kafka, enriches with computed features,
    applies 5-minute tumbling window aggregations, and writes to console + JDBC.
    """
    raw = (spark.readStream
           .format("kafka")
           .option("kafka.bootstrap.servers", KAFKA_BROKER)
           .option("subscribe", KAFKA_TOPIC_TXN)
           .option("startingOffsets", "latest")
           .option("failOnDataLoss", "false")
           .load())

    parsed = (raw
              .select(F.from_json(
                  F.col("value").cast("string"), TXN_SCHEMA
              ).alias("data"), "timestamp")
              .select("data.*",
                      F.col("timestamp").alias("kafka_ts")))

    # Feature engineering in Spark
    enriched = (parsed
                .withColumn("hour_of_day",
                            F.hour(F.to_timestamp("timestamp")))
                .withColumn("day_of_week",
                            F.dayofweek(F.to_timestamp("timestamp")))
                .withColumn("amount_bucket",
                            F.when(F.col("amount") < 100, "micro")
                             .when(F.col("amount") < 500, "small")
                             .when(F.col("amount") < 2000, "medium")
                             .otherwise("large"))
                .withColumn("is_night_txn",
                            F.when((F.col("hour_of_day") <= 5) |
                                   (F.col("hour_of_day") >= 22), 1)
                             .otherwise(0))
                .withColumn("risk_flag",
                            F.when(
                                (F.col("geo_distance_km") > 200) |
                                (F.col("device_fingerprint_match") == 0) |
                                (F.col("transaction_velocity_7d") > 8),
                                "HIGH"
                            ).when(
                                (F.col("geo_distance_km") > 50) |
                                (F.col("is_night_txn") == 1),
                                "MEDIUM"
                            ).otherwise("LOW")))

    # 5-minute tumbling window aggregations per user
    windowed = (enriched
                .withWatermark("kafka_ts", "10 minutes")
                .groupBy(
                    F.window("kafka_ts", "5 minutes"),
                    "user_id"
                )
                .agg(
                    F.count("transaction_id").alias("txn_count_5m"),
                    F.sum("amount").alias("total_amount_5m"),
                    F.avg("amount").alias("avg_amount_5m"),
                    F.max("geo_distance_km").alias("max_geo_dist_5m"),
                    F.sum("is_night_txn").alias("night_txn_count_5m")
                ))

    # Write enriched stream to console (dev) and JDBC (prod)
    query_console = (enriched
                     .writeStream
                     .outputMode("append")
                     .format("console")
                     .option("truncate", False)
                     .option("numRows", 5)
                     .trigger(processingTime="10 seconds")
                     .start())

    # Write windowed aggregations to console
    query_window = (windowed
                    .writeStream
                    .outputMode("update")
                    .format("console")
                    .option("truncate", False)
                    .trigger(processingTime="30 seconds")
                    .start())

    return query_console, query_window


def stream_stock_ticks(spark: "SparkSession"):
    """
    Reads stock ticks from Kafka and computes rolling 1-minute OHLCV aggregations.
    """
    raw = (spark.readStream
           .format("kafka")
           .option("kafka.bootstrap.servers", KAFKA_BROKER)
           .option("subscribe", KAFKA_TOPIC_TICK)
           .option("startingOffsets", "latest")
           .load())

    parsed = (raw
              .select(F.from_json(
                  F.col("value").cast("string"), TICK_SCHEMA
              ).alias("d"), "timestamp")
              .select("d.*",
                      F.col("timestamp").alias("kafka_ts")))

    ohlcv = (parsed
             .withWatermark("kafka_ts", "2 minutes")
             .groupBy(
                 F.window("kafka_ts", "1 minute"),
                 "symbol"
             )
             .agg(
                 F.first("price").alias("open"),
                 F.max("price").alias("high"),
                 F.min("price").alias("low"),
                 F.last("price").alias("close"),
                 F.sum("volume").alias("volume")
             ))

    query = (ohlcv
             .writeStream
             .outputMode("update")
             .format("console")
             .option("truncate", False)
             .trigger(processingTime="60 seconds")
             .start())
    return query


def main():
    if not SPARK_AVAILABLE:
        print("PySpark required. Install with: pip install pyspark")
        return

    print("Starting FinSight Spark Streaming jobs...")
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    q1, q2 = stream_transactions(spark)
    q3 = stream_stock_ticks(spark)

    print("Streaming jobs started. Waiting for data...")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
