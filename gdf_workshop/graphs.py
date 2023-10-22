import torch
from torch_geometric.utils.negative_sampling import negative_sampling
import pyspark.sql.functions as f
from pyspark.sql.window import Window


def create_node_df(data):
    nodes_df = data.withColumn("author", f.explode("authors"))
    nodes_df = nodes_df.select("id", "author", "title", "year")
    nodes_df = nodes_df.distinct().select("author", "title")
    nodes_df = (
        nodes_df.groupBy("author")
        .agg(f.count("title").alias("number_titles"))
        .select("author", "number_titles")
    )
    w = Window().orderBy(f.lit("A"))
    nodes_df = nodes_df.withColumn("id", f.row_number().over(w))
    nodes_df = nodes_df.sort("number_titles", ascending=False)

    return nodes_df


def create_edge_df(data, nodes_df):
    new_df = data.withColumn("author", f.explode("authors"))
    new_df = new_df.select("id", "author", "title", "year")
    joined_df = new_df.alias("a1").join(new_df.alias("a2"), "title")
    edges_df = joined_df.filter(f.col("a1.author") != f.col("a2.author"))
    edges_df = edges_df.select(
        f.col("a1.author").alias("author1"),
        f.col("a2.author").alias("author2"),
        "title",
        f.col("a1.year").alias("year"),
    )
    edges_df = (
        edges_df.join(
            nodes_df.select("author", "id"),
            edges_df.author1 == nodes_df.author,
            how="left",
        )
        .withColumnRenamed("id", "id1")
        .drop("author")
    )
    edges_df = (
        edges_df.join(
            nodes_df.select("author", "id"),
            edges_df.author2 == nodes_df.author,
            how="left",
        )
        .withColumnRenamed("id", "id2")
        .drop("author", "author1", "author2", "title")
    )

    return edges_df


def get_positive_edges(edges_df, cutoff=2006, training=True):
    positive_edges_df = (
        edges_df.filter(edges_df.year < cutoff if training else edges_df.year >= cutoff)
        .select("id1", "id2")
        .withColumn("coauth", f.lit(1))
        .withColumnRenamed("id1", "src")
        .withColumnRenamed("id2", "dst")
    )
    return positive_edges_df


def get_negative_edge_indexes(positive_edges_df):
    torch.manual_seed(2)
    src_edges_list = [row.src for row in positive_edges_df.select("src").collect()]
    dst_edges_list = [row.dst for row in positive_edges_df.select("dst").collect()]
    pos_edge_index = torch.as_tensor([src_edges_list, dst_edges_list])
    negative_edge_index = negative_sampling(pos_edge_index, force_undirected=False)

    return negative_edge_index.T.tolist()


def add_negative_edges(spark, positive_edges_df):
    negative_edges_indexes = get_negative_edge_indexes(positive_edges_df)
    negative_edges_df = spark.createDataFrame(
        data=negative_edges_indexes, schema=["src", "dst"]
    ).withColumn("coauth", f.lit(0))

    edges_df = positive_edges_df.union(negative_edges_df)

    return edges_df


def get_relevant_nodes(nodes_df, edges_df):
    # node dataframe should only contain nodes for which there are edges
    relevant_nodes_df = (
        nodes_df.join(
            edges_df,
            (nodes_df.id == edges_df.src) | (nodes_df.id == edges_df.dst),
            how="inner",
        )
        .select("id", "author")
        .distinct()
    )
    return relevant_nodes_df
