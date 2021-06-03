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
import os
from pathlib import Path
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

    # 算两个文本的相似度
    def text_semantic_sim(self, text1, text2):
        vec_des = self.text2vec([text1])
        vec_term = self.text2vec([text2])
        norm_des = np.linalg.norm(vec_des)
        norm_term = np.linalg.norm(vec_term)
        if norm_des == 0 or norm_term == 0:
            return 0
        else:
            return 0.5 + ((vec_des.dot(vec_term) / (norm_des * norm_term)) / 2)

    # 将一个文本转换成word2vec向量表示
    def text2vec(self, topic_words):
        topic_text = " ".join(topic_words).lower()

        if len(topic_text) == 0:
            return np.zeros([300])
        words = [w for w in topic_text.split() if w and w.lower() not in self.stop_words]
        if len(words) == 0:
            return np.zeros([300])
        vec_des = sum([self.embedding.get(w, np.zeros([300])) for w in words]) / len(words)

        return vec_des

    def get_word2vec_score_for_sentences(self, sentence, target_sentence_list: list):
        score_list = []
        for item in target_sentence_list:
            score_list.append(self.text_semantic_sim(sentence, item))
        return score_list

    """
        wikipedia2vec的pkl文件不能直接用KeyedVectors.load_word2vec_format导入，所以需要先下载txt版本，然后转换成二进制bin文件，再用KeyedVectors.load_word2vec_format导入
        即需要下载的是 https://wikipedia2vec.github.io/wikipedia2vec/pretrained/ 网站下的English类别下的enwiki_20180420 (window=5, iteration=10, negative=15)的300d(txt)文件，并将txt文件
        保存在data/wiki_word2vec/300/目录下
    """
    @staticmethod
    def txt_2_bin(txt_path, bin_path):
        if os.path.isfile(Path(txt_path)):
            if not os.path.isfile(Path(bin_path)):
                print("txt开始转换成bin文件")
                wv = KeyedVectors.load_word2vec_format(txt_path, binary=False)
                print("txt读取完毕")
                wv.save_word2vec_format(bin_path, binary=True)
                print("txt文件转换成bin文件完毕")
        else:
            raise Exception("txt文件没有下载，详情请看project/Word2Vec/Operation.py下的第66-68行的注释")


if __name__ == "__main__":

    # 得到word2vec模型的二进制文件
    W2VOperation.txt_2_bin(str(PathUtil.wiki_emb_path() / "300" / "enwiki_20180420_300d.txt"),
                           str(PathUtil.wiki_emb_path() / "300" / "enwiki_20180420_300d.bin"))

    # word2vec_path = str(PathUtil.wiki_emb_path() / "300" / "new_enwiki_with_title.bin")
    # operation = W2VOperation(word2vec_path)
    #
    # # print(operation.text_semantic_sim("named entity recognition", "nature language processing"))
    # # print(operation.text_semantic_sim("named entity recognition", "computer vision"))
    #
    # tmp_score_list = operation.get_word2vec_score_for_sentences('how to always round up to the next integer', [
    #     'How to round up the result of integer division?',
    #     'how to make Math.random round to a number',
    #     'How to get ImageView with round edge and round vertices in android?',
    #     'How do I convert the value of a TextView to integer',
    #     'how to round of 3 numbers to 1?',
    #     'How to round decimal numbers in Android'
    # ])
    # print(tmp_score_list)
