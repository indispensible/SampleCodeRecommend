#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------
@Author: Lv Gang
@Email: 1547554745@qq.com
@Created: 2021/04/08
------------------------------------------
@Modify: 2021/04/08
------------------------------------------
@Description:
"""
import pickle

from gensim import similarities
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from nltk.corpus import stopwords
import numpy as np

from util.path_util import PathUtil


class TFIDFOperation:

    def __init__(self):
        version = "v1"
        trained_tfIdf_model_path = str(PathUtil.output_trained_tfIdf_model_dir() / "tfidf.{v}.model".format(v=version))
        dict_path = str(PathUtil.output_trained_tfIdf_model_dir() / "{v}.dict".format(v=version))
        self.tfidf_model = TfidfModel.load(trained_tfIdf_model_path)
        self.dct: Dictionary = Dictionary.load(dict_path)
        self.sentence_list = pickle.load(open(str(PathUtil.output_document_tfIdf_model_dir() / "tfidf.sentence_vote_list"), "rb"))
        self.similarities_index = similarities.SparseMatrixSimilarity.load(str(PathUtil.output_index_model_dir() / 'sparse_matrix_similarity.index'))
        self.stop_words = stopwords.words('english')

    def get_sims(self, sentence):
        sentence_word_list = [item for item in sentence.split(" ") if item not in self.stop_words]
        bow = self.dct.doc2bow(sentence_word_list)
        vector = self.tfidf_model[bow]
        sims = self.similarities_index[vector]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims

    def predict_n_rank(self, sentence, n=100):
        sims = self.get_sims(sentence)
        sentence_rank_n = []
        rank_n = []
        for item in sims:
            if len(sentence_rank_n) >= n:
                break
            if self.sentence_list[item[0]][0] not in sentence_rank_n and self.sentence_list[item[0]][0] != sentence:
                rank_n.append([self.sentence_list[item[0]], item[1]])
                sentence_rank_n.append(self.sentence_list[item[0]][0])
        return rank_n

    def get_tf_idf_score_for_sentences(self, sentence, target_sentence_list: list):
        sims = self.get_sims(sentence)
        score_list = []
        for i in range(0, len(target_sentence_list)):
            for item in sims:
                if self.sentence_list[item[0]][0] == target_sentence_list[i]:
                    score_list.append(np.float(item[1]))
                    break
        return score_list


if __name__ == "__main__":
    op = TFIDFOperation()

    op.predict_n_rank("Debug exceptions in AWT queue thread")

    # tmp_score_list = op.get_tf_idf_score_for_sentences('how to always round up to the next integer', [
    #     'How to round up the result of integer division?',
    #     'how to make Math.random round to a number',
    #     'How to get ImageView with round edge and round vertices in android?',
    #     'How do I convert the value of a TextView to integer',
    #     'how to round of 3 numbers to 1?',
    #     'How to round decimal numbers in Android'
    # ])
    # print(tmp_score_list)
