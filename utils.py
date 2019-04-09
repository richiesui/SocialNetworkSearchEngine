# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-04-10

import numpy as np
import difflib

import re
from simhash import Simhash

from nltk.text import TextCollection
from nltk.tokenize import word_tokenize


def cosine_similarity(sentence1: str, sentence2: str) -> float:
    """
    compute normalized COSINE similarity.
    :param sentence1: English sentence.
    :param sentence2: English sentence.
    :return: normalized similarity of two input sentences.
    """
    seg1 = sentence1.strip(" ").split(" ")
    seg2 = sentence2.strip(" ").split(" ")
    word_list = list(set([word for word in seg1 + seg2]))
    word_count_vec_1 = []
    word_count_vec_2 = []
    for word in word_list:
        word_count_vec_1.append(seg1.count(word))
        word_count_vec_2.append(seg2.count(word))

    vec_1 = np.array(word_count_vec_1)
    vec_2 = np.array(word_count_vec_2)

    num = vec_1.dot(vec_2.T)
    denom = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos

    return sim


def compute_levenshtein_distance(sentence1: str, sentence2: str) -> int:
    """
    compute levenshtein distance.

    """
    leven_cost = 0
    s = difflib.SequenceMatcher(None, sentence1, sentence2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():

        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)

    return leven_cost


def compute_levenshtein_similarity(sentence1: str, sentence2: str) -> float:
    """Compute the hamming similarity."""
    leven_cost = compute_levenshtein_distance(sentence1, sentence2)
    return leven_cost / len(sentence2)


def compute_simhash_hamming_similarity(sentence1: str, sentence2: str) -> float:
    """need to normalize after compute!"""

    def get_features(s):
        width = 3
        s = s.lower()
        s = re.sub(r'[^\w]+', '', s)
        return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

    hash_value1 = Simhash(get_features(sentence1)).value
    hash_value2 = Simhash(get_features(sentence2)).value

    return compute_levenshtein_similarity(str(hash_value1), str(hash_value2))


def compute_jaccard_similarity(sentence1: str, sentence2: str) -> float:
    word_set1 = set(sentence1.strip(" ").split(" "))
    word_set2 = set(sentence2.strip(" ").split(" "))

    return len(word_set1 & word_set2) / len(word_set1 | word_set2)


def compute_bm25_similarity(sentence1: str, sentence2: str) -> float:
    # todo
    return 1.0


def compute_tf_idf_similarity(query: str, content: str, type: str) -> float:
    """
    Compute the mean tf-idf or tf
     similarity for one sentence with multi query words.
    :param query: a string contain all key word split by one space
    :param content: string list with every content relevent to this query.
    :return: average tf-idf or tf similarity.
    """
    sents = [word_tokenize(content), word_tokenize("")]  # add one empty file to smooth.
    corpus = TextCollection(sents)  # 构建语料库

    result_list = []
    for key_word in query.strip(" ").split(" "):
        if type == "tf_idf":
            result_list.append(corpus.tf_idf(key_word, corpus))
        elif type == "tf":
            result_list.append(corpus.tf(key_word, corpus))
        else:
            raise KeyError

    return sum(result_list) / len(result_list)


if __name__ == '__main__':
    # print(compute_tf_idf_similarity("one sentence",
    #                                 ["this is sentence one", "this is sentence two", "this is sentence three"]))
    # print(compute_levenshtein_similarity("combat readiness", "ab v c"))
    compute_tf_idf_similarity('combat readiness',
                              'Al Qassam: Our combat readiness comes after Israel determination to launch aggressive attack prior Israeli elections\n')
#     # print(cosine_similarity("a v c ", "ab v c"))
#     # print(Simhash('aa').distance(Simhash('bb')))
