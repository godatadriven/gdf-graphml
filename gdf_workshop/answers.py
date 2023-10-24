from graphframes import *

from features import to_undirected

def add_preferential_attachment(g):
    positive_edges = g.edges.filter(f.col("coauth") == 1)
    undirected_edges = to_undirected(positive_edges)
    directed_graph = GraphFrame(g.vertices, undirected_edges)

    node_degrees = directed_graph.inDegrees.withColumnRenamed(
        "inDegree", "node_degrees"
    )
    edges = g.edges
    preferential_attachment = (
        edges.join(node_degrees, edges.src == node_degrees.id, how="left")
        .withColumnRenamed("node_degrees", "src_node_degrees")
        .drop("id")
        .join(node_degrees, edges.dst == node_degrees.id, how="left")
        .withColumnRenamed("node_degrees", "dst_node_degrees")
        .withColumn(
            "preferential_attachment",
            f.col("src_node_degrees") * f.col("dst_node_degrees"),
        )
        .drop("id", "src_node_degrees", "dst_node_degrees")
    )

    return preferential_attachment

    
def find_common_authors(g, return_directed=False):
    positive_edges = g.edges.filter(f.col("coauth") == 1)
    undirected_edges = to_undirected(positive_edges)
    directed_graph = GraphFrame(g.vertices, undirected_edges)

    all_common_authors = (
        directed_graph.find("(n1)-[]->(neighbor); (n2)-[]->(neighbor)")
        .selectExpr("n1.id as n1_id", "n2.id as n2_id", "neighbor.id as neighbour_id")
        .groupBy("n1_id", "n2_id")
        .agg(f.countDistinct("neighbour_id").alias("common_authors"))
    )

    edges = g.edges
    common_authors = (
        edges.join(
            all_common_authors,
            (edges.src == all_common_authors.n1_id)
            & (edges.dst == all_common_authors.n2_id),
            how="left",
        )
        .na.fill(0)
        .drop("n1_id", "n2_id")
    )

    return common_authors

