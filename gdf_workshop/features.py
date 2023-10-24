import pyspark.sql.functions as f
from pyspark.sql.window import Window
from graphframes import *


# def add_preferential_attachment(g):
#     ### EXERCISE 6A ###
#     ...


# def find_common_authors(g):
#     ### EXERCISE 6B ###
#     ...


def to_undirected(directed_edges_df):
    # order edges so that src < dst
    ordered_directed_edged_df = (directed_edges_df
          .withColumn("src_new", f.least("src", "dst"))
          .withColumn("dst_new", f.greatest("src", "dst"))
          .drop("src", "dst")
          .withColumnRenamed("src_new", "src")
          .withColumnRenamed("dst_new", "dst")
    )

    # get edges in reverse order so that src > dst
    reverse_edges_df = (
        ordered_directed_edged_df.withColumnRenamed("dst", "src_")
        .withColumnRenamed("src", "dst")
        .withColumnRenamed("src_", "src")
    )
    undircted_edges_df = ordered_directed_edged_df.unionByName(reverse_edges_df)
    return undircted_edges_df

def combine_features(*features, output_cols, source_table_alias="ca"):
    combined_features = features[0]
    for next_feature in features[1:]:
        combined_features = (
            combined_features.join(
                next_feature,
                (combined_features.src == next_feature.src)
                & (combined_features.dst == next_feature.dst),
                how="left",
            )
            .selectExpr(
                f"{source_table_alias}.src",
                f"{source_table_alias}.dst",
                f"{source_table_alias}.coauth as label",
                *output_cols,
            )
            .fillna(0, subset=list(output_cols))
        )
    return combined_features

def extract_first_author_collaboration(df):
    w = Window.partitionBy("id1", "id2").orderBy(f.col("year"))
    df = (
        df.withColumn("row", f.row_number().over(w))
        .filter(f.col("row") == 1)
        .drop("row")
    )

    return df
