# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-10


import os
import pandas as pd
from utils import *

# query_files_dir = "/Users/zhangkai/JupyterNotebookProjects/projects/search engine/spite_v4/spite_result_v4/"
query_files_dir = "./spite_v4/spite_result_v4/"
query_file_name_list = os.listdir(query_files_dir)


def get_one_query_df(one_query_results_file_path: str) -> pd.DataFrame:
    columns = ["query", "twitter_name", "twitter_id", "time", "reply", "retweet", "like", "content"]
    pd_query_results = []
    one_query_results = open(one_query_results_file_path).readlines()

    for one_query_result in one_query_results:
        one_row = [file_name]
        one_row.extend(one_query_result.split("\x01"))
        pd_query_results.append(one_row)

    return pd.DataFrame(pd_query_results, columns=columns)


all_df = None
for file_name in query_file_name_list:
    one_query_df = get_one_query_df(query_files_dir + file_name)
    all_df = one_query_df if all_df is None else all_df.append(one_query_df, ignore_index=True)

all_df["cosine_sim"] = all_df.apply(lambda row: cosine_similarity(row["query"], row["content"]), axis=1)
all_df["levenshtein_sim"] = all_df.apply(lambda row: compute_levenshtein_similarity(row["query"], row["content"]),
                                         axis=1)
all_df["simhash_sim"] = all_df.apply(lambda row: compute_simhash_hamming_similarity(row["query"], row["content"]),
                                     axis=1)
all_df["jaccard_sim"] = all_df.apply(lambda row: compute_jaccard_similarity(row["query"], row["content"]), axis=1)

all_df["tf"] = all_df.apply(lambda row: compute_tf_idf_similarity(row["query"], row["content"], "tf"), axis=1)
all_df["tf_idf"] = all_df.apply(lambda row: compute_tf_idf_similarity(row["query"], row["content"], "tf_idf"), axis=1)

# print(all_df)
