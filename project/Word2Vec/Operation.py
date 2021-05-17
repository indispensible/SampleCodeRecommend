#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------
@Author: Lv Gang
@Email: 1547554745@qq.com
@Created: 2021/03/29
------------------------------------------
@Modify: 2021/03/29
------------------------------------------
@Description:
"""
import json
from time import time

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from nltk.corpus import stopwords
from wikipedia2vec import Wikipedia2Vec

from util.path_util import PathUtil


class W2VOperation:

    def __init__(self, word2vec_path=str(PathUtil.wiki_emb_path() / "300" / "new_enwiki_with_title.bin")):
        wv = KeyedVectors.load(word2vec_path)
        self.embedding = {k: wv[k] for k in wv.vocab.keys()}
        self.stop_words = stopwords.words('english')
        print("w2v模型导入完毕")

    def text_semantic_sim(self, text1, text2):
        vec_des = self.text2vec([text1])
        vec_term = self.text2vec([text2])
        norm_des = np.linalg.norm(vec_des)
        norm_term = np.linalg.norm(vec_term)
        if norm_des == 0 or norm_term == 0:
            return 0
        else:
            return 0.5 + ((vec_des.dot(vec_term) / (norm_des * norm_term)) / 2)

    def text2vec(self, topic_words):
        topic_text = " ".join(topic_words).lower()

        if len(topic_text) == 0:
            return np.zeros([300])
        words = [w for w in topic_text.split() if w and w not in self.stop_words]
        if len(words) == 0:
            return np.zeros([300])
        vec_des = sum([self.embedding.get(w, np.zeros([300])) for w in words]) / len(words)

        return vec_des


if __name__ == "__main__":
    word2vec_path = str(PathUtil.wiki_emb_path() / "300" / "new_enwiki_with_title.bin")
    operation = W2VOperation(word2vec_path)

    print(operation.text_semantic_sim("named entity recognition", "nature language processing"))
    print(operation.text_semantic_sim("named entity recognition", "computer vision"))
