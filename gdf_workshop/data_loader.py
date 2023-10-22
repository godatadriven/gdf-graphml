import pyspark.sql.functions as f
from pyspark.sql.types import StructType, IntegerType, StringType
from graphframes import *


def read_dummy_edges(spark):
    edges_df = spark.createDataFrame(
        data=[
            (1, 2, 1, 2005),
            (2, 3, 1, 2005),
            (2, 4, 1, 2005),
            (0, 3, 1, 2005),
            (1, 4, 1, 2005),
            (3, 4, 1, 2005),
            (1, 2, 1, 2010),
            (2, 3, 1, 2010),
            (2, 4, 1, 2010),
            (0, 3, 1, 2010),
            (1, 4, 1, 2010),
            (3, 4, 1, 2010),
            (1, 5, 1, 2010),
        ],
        schema=["id1", "id2", "coauth", "year"],
    )
    return edges_df


def read_dummy_nodes(spark):
    nodes_df = spark.createDataFrame(
        data=[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
            (4, "e"),
            (5, "f"),
        ],
        schema=["id", "author"],
    )

    return nodes_df


def read_csv_edges(spark, filepath):
    schema = schema = (
        StructType()
        .add("year", IntegerType(), True)
        .add("id1", IntegerType(), True)
        .add("id2", IntegerType(), True)
    )
    edges_df = (
        spark.read.option("header", True)
        .schema(schema)
        .csv(filepath)
    )

    return edges_df


def read_csv_nodes(spark, filepath):
    schema = schema = (
        StructType()
        .add("author", StringType(), True)
        .add("number_titles", IntegerType(), True)
        .add("id", IntegerType(), True)
    )
    nodes_df = (
        spark.read.option("header", True).schema(schema).csv(filepath)
    )

    return nodes_df


def friends(spark):
    """A GraphFrame of friends in a (fake) social network."""
    # Vertex DataFrame
    v = spark.createDataFrame(
        [
            ("a", "Alice", 34),
            ("b", "Bob", 36),
            ("c", "Charlie", 30),
            ("d", "David", 29),
            ("e", "Esther", 32),
            ("f", "Fanny", 36),
        ],
        ["id", "name", "age"],
    )
    # Edge DataFrame
    e = spark.createDataFrame(
        [
            ("a", "b", "friend"),
            ("b", "c", "follow"),
            ("c", "b", "follow"),
            ("f", "c", "follow"),
            ("e", "f", "follow"),
            ("e", "d", "friend"),
            ("d", "a", "friend"),
        ],
        ["src", "dst", "relationship"],
    )
    # Create a GraphFrame
    return GraphFrame(v, e)
