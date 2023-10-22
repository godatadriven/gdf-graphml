# system packages
import argparse

# installed packages
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from graphframes import *

# local modules
from data_loader import (
    read_json_data,
    read_dummy_edges,
    read_dummy_nodes,
    read_csv_edges,
    read_csv_nodes,
)
from graphs import (
    create_node_df,
    create_edge_df,
    get_positive_edges,
    add_negative_edges,
    get_relevant_nodes,
)
from features import (
    extract_first_author_collaboration,
    find_common_authors,
    add_preferential_attachment,
    combine_features,
)
from ml import train_model, evaluate_model


spark = (
    SparkSession.builder.config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "6g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("FATAL")


@f.udf("string")
def to_undir(id1, id2):
    if id1 >= id2:
        return "Delete"
    else:
        return "Keep"


def convert_to_undirected(df):
    # assumes every edge between i,j has an edge between j,i
    undirected_df = (
        df.withColumn("undir", to_undir(df.id1, df.id2))
        .filter('undir == "Keep"')
        .drop("undir")
    )
    return undirected_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--use_csv_data", action="store_true")
    parser.add_argument("--use_json_data", action="store_true")
    args = parser.parse_args()

    #################### READ DATA ####################
    if args.use_csv_data:
        edges_df = read_csv_edges(spark)
        nodes_df = read_csv_nodes(spark)

    elif args.use_json_data:
        data = read_json_data(spark)

        nodes_df = create_node_df(data)

        edges_df = create_edge_df(data, nodes_df)
        edges_df = edges_df.transform(extract_first_author_collaboration).transform(
            convert_to_undirected
        )
    elif args.use_dummy_data:
        edges_df = read_dummy_edges(spark)
        nodes_df = read_dummy_nodes(spark)

    #################### CREATE GRAPH ####################

    # create training and test postive edges
    training_positive_edges_df = get_positive_edges(
        edges_df, cutoff=2006, training=True
    )
    test_positive_edges_df = get_positive_edges(edges_df, cutoff=2006, training=False)

    # create training and test negative edges (using PyTorch sampling)
    training_edges_df = add_negative_edges(spark, training_positive_edges_df)
    test_edges_df = add_negative_edges(spark, test_positive_edges_df)

    # create training and test nodes df
    training_nodes_df = get_relevant_nodes(nodes_df, training_edges_df)
    test_nodes_df = get_relevant_nodes(nodes_df, test_edges_df)

    # create training/test graph
    training_graph = GraphFrame(training_nodes_df, training_edges_df)
    test_graph = GraphFrame(test_nodes_df, test_edges_df)

    #################### FEATURES ####################

    # calculate common authors for pairs of nodes
    training_common_authors = find_common_authors(training_graph).alias("train_ca")
    test_common_authors = find_common_authors(test_graph).alias("test_ca")

    # calculate preferntial attachment for pairs of nodes
    training_preferential_attachment = add_preferential_attachment(
        training_graph
    ).alias("train_pa")
    test_preferential_attachment = add_preferential_attachment(test_graph).alias(
        "test_pa"
    )

    # combine features into a single dataframe (for train/test)
    features_cols = ["common_authors", "preferential_attachment"]

    training_features = combine_features(
        training_common_authors,
        training_preferential_attachment,
        output_cols=tuple(features_cols),
        source_table_alias="train_ca",
    )

    test_features = combine_features(
        test_common_authors,
        test_preferential_attachment,
        output_cols=tuple(features_cols),
        source_table_alias="test_ca",
    )

    #################### TRAIN MODEL ####################

    basic_model = train_model(features_cols, training_features)
    evaluate_model(basic_model, test_features)
