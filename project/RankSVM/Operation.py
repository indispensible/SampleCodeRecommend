#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------
@Author: Lv Gang
@Email: 1547554745@qq.com
@Created: 2021/04/06
------------------------------------------
@Modify: 2021/04/06
------------------------------------------
@Description:
"""
import json
import os
import pickle
from random import Random
from time import time

from sekg.ir.doc.wrapper import MultiFieldDocumentCollection
from tqdm import tqdm

from project.FuncVerbClassifier import FuncVerbClassifier
from project.TFIDF.Operation import TFIDFOperation
from project.Word2Vec.Operation import W2VOperation

from project.RankSVM.RankSVM import transform_pairwise, RankSVM
from util.path_util import PathUtil
import numpy as np

"""
    特征是：点赞数/100、tfidf or es分数、w2v分数、funcverb所属的类别、funcverb所属类别的分数
"""


class SVMOperation:

    def __init__(self, sample_code_doc_path=str(PathUtil.doc("sample", "v1.1"))):
        self.rank_svm_model = None
        self.tf_idf_operation = TFIDFOperation()
        sample_code_doc: MultiFieldDocumentCollection = MultiFieldDocumentCollection.load(sample_code_doc_path)
        self.sentence_2_code = {}
        self.w2v_operation = W2VOperation()
        self.func_verb_net = FuncVerbClassifier()
        for item in sample_code_doc.get_document_list():
            self.sentence_2_code[item.get_doc_text_by_field("title_ori")] = item.get_doc_text_by_field("code")
        if os.path.exists(str(PathUtil.output_RankSVM_model_dir() / "rank_svm.model")):
            self.load()

    def train_rank_svm(self, sentence_list=None, model_path=str(PathUtil.output_RankSVM_model_dir() / "rank_svm.model")):
        # sentence_list的格式可以通过origin.sentence文件来查看
        # x的特征是：点赞数/100、tfidf分数、w2v分数、funcverb所属的类别、funcverb所属类别的分数
        if sentence_list is None:
            sentence_list = pickle.load(open(str(PathUtil.output_RankSVM_model_dir() / "origin.sentence"), "rb+"))
        X_list = []
        Y_list = []
        for item in tqdm(sentence_list):
            sentence = item['sentence']
            target_sentence_list = []
            vote_list = []
            func_label_list = []
            func_score_list = []
            for key, value_list in item.items():
                if "sentence_with_vote" in key:
                    target_sentence_list.append(value_list[0])
                    vote_list.append(value_list[1])
                    func_res = self.func_verb_net.predict(value_list[0])
                    func_label_list.append(func_res[0])
                    func_score_list.append(func_res[1])
            tf_idf_score_list = self.tf_idf_operation.get_tf_idf_score_for_sentences(sentence, target_sentence_list)
            word2vec_list = self.w2v_operation.get_word2vec_score_for_sentences(sentence, target_sentence_list)
            new_x_list = []
            new_y_list = []

            for i in range(0, len(target_sentence_list)):
                new_x_list.append([
                    vote_list[i] / 100,
                    tf_idf_score_list[i],
                    word2vec_list[i],
                    func_label_list[i],
                    func_score_list[i]
                ])
                y_score = tf_idf_score_list[i]
                if i == 0:
                    y_score += 10
                new_y_list.append(y_score)

            transform_x_list, transform_y_list = transform_pairwise(np.array(new_x_list), np.array(new_y_list), True)

            X_list.extend(transform_x_list)
            Y_list.extend(transform_y_list)
        X = np.array(X_list)
        Y = np.array(Y_list)
        rank_svm = RankSVM(max_iter=2000000).fit_with_transform_pairwise(X, Y)
        with open(model_path, "wb") as fw:
            pickle.dump(rank_svm, fw)
        print("模型保存完毕")

    def load(self, model_path=str(PathUtil.output_RankSVM_model_dir() / "rank_svm.model")):
        self.rank_svm_model = pickle.load(open(model_path, "rb"))

    def predict(self, sentence, target_sentence="", n=30, print_code=False):
        predict_dict = {
            "question": sentence,
            "target": target_sentence,
            "rank_10_result": [],
            "code_10_result": []
        }
        rank_n_sentence = self.tf_idf_operation.predict_n_rank(sentence, n)
        x = []
        # vote数和TFIDF分数差距太大了，所以vote要除以100，降低参数的影响
        for item in rank_n_sentence:
            item_sentence = item[0][0]
            w2v_score = self.w2v_operation.text_semantic_sim(sentence, item_sentence)
            func_label, func_score = self.func_verb_net.predict(item_sentence)
            x.append([item[0][1] / 100, item[1], w2v_score, func_label, func_score])
        x = np.array(x)
        sort_index = self.rank_svm_model.predict(x)
        sort_dict = dict(zip(sort_index, rank_n_sentence))
        print("Question: " + sentence)
        if target_sentence != "":
            print("Target: " + target_sentence)
        print("-" * 50)
        print("前十名结果:")
        for i in range(n - 1, n - 11, -1):
            predict_dict["rank_10_result"].append(sort_dict[i][0][0])
            predict_dict["code_10_result"].append(self.sentence_2_code[sort_dict[i][0][0]])
            print("No." + str(n - i) + " Related Task: " + sort_dict[i][0][0])
            if print_code:
                print("Code:")
                print(self.sentence_2_code[sort_dict[i][0][0]])
                print("-" * 40)
        print("=" * 60)
        return predict_dict

    def score(self, sentence_list: list, rank_n=30):
        score_list = []
        rank_before_3_score_list = []
        rank_before_5_score_list = []
        rank_before_10_score_list = []

        rank_before_3 = 0
        rank_before_5 = 0
        rank_before_10 = 0

        for sentence, target_sentence in tqdm(sentence_list):
            rank_n_sentence = self.tf_idf_operation.predict_n_rank(sentence, rank_n)
            x = []
            for item in rank_n_sentence:
                w2v_score = self.w2v_operation.text_semantic_sim(sentence, item[0][0])
                func_label, func_score = self.func_verb_net.predict(item[0][0])
                x.append([item[0][1] / 100, item[1], w2v_score, func_label, func_score])
            x = np.array(x)
            sort_index = self.rank_svm_model.predict(x)
            sort_dict = dict(zip(sort_index, rank_n_sentence))
            rank = 0
            for key, item in sort_dict.items():
                if target_sentence == item[0][0] or target_sentence in item[0][0]:
                    rank = rank_n - key
                    break
            if rank == 0:
                score_list.append(0)
                rank_before_3_score_list.append(0)
                rank_before_5_score_list.append(0)
                rank_before_10_score_list.append(0)
            else:
                if rank <= 1:
                    score_list.append(1 / rank)
                else:
                    score_list.append(0)

                if rank <= 3:
                    rank_before_3_score_list.append(1 / rank)
                    rank_before_3 += 1
                else:
                    rank_before_3_score_list.append(0)

                if rank <= 5:
                    rank_before_5_score_list.append(1 / rank)
                    rank_before_5 += 1
                else:
                    rank_before_5_score_list.append(0)

                if rank <= 10:
                    rank_before_10_score_list.append(1 / rank)
                    rank_before_10 += 1
                else:
                    rank_before_10_score_list.append(0)

        print("MRR_1: " + str(sum(score_list) / len(score_list)))
        print("MRR_3: " + str(sum(rank_before_3_score_list) / len(rank_before_3_score_list)))
        print("MRR_5: " + str(sum(rank_before_5_score_list) / len(rank_before_5_score_list)))
        print("MRR_10: " + str(sum(rank_before_10_score_list) / len(rank_before_10_score_list)))

        print("前三包含答案百分比: " + str(rank_before_3 / len(sentence_list)))
        print("前五包含答案百分比: " + str(rank_before_5 / len(sentence_list)))
        print("前十包含答案百分比: " + str(rank_before_10 / len(sentence_list)))

    def get_score_for_random_8_2(self):
        random = Random()
        sentence_list = pickle.load(open(str(PathUtil.output_RankSVM_model_dir() / "origin.sentence"), "rb+"))
        all_id_set = {i for i in range(0, len(sentence_list))}
        train_num = int(len(sentence_list) * 0.8)
        train_id_set = set()
        while len(train_id_set) != train_num:
            num = random.randint(0, len(sentence_list) - 1)
            if num not in train_id_set:
                train_id_set.add(num)
        test_id_set = all_id_set.difference(train_id_set)
        train_data_list = [sentence_list[index] for index in train_id_set]
        test_data_list = [[sentence_list[index]['sentence'], sentence_list[index]['positive_sentence_with_vote'][0]] for index in test_id_set]
        self.train_rank_svm(train_data_list)
        self.load()
        self.score(test_data_list)


if __name__ == "__main__":
    op = SVMOperation()

    # 221到242行一起运行和245行分开运行（分开注释）
    # # 训练RankSVM模型
    op.train_rank_svm(model_path=str(PathUtil.output_RankSVM_model_dir() / "rank_svm.new.final.model"))
    # 测试用例
    op.load(str(PathUtil.output_RankSVM_model_dir() / "rank_svm.new.final.model"))
    op.load()
    start_time = time()
    res_list = [
        op.predict("how to always round up to the next integer", "How to round up the result of integer division?", 30),
        op.predict("How to play a sound (alert) in a java application?", "How can I play sound in Java?", 30),
        op.predict("Debug exceptions in AWT queue thread", "How can I catch AWT thread exceptions in Java?", 30),
        op.predict("JUnit 4 Expected Exception type",
                   "How do you assert that a certain exception is thrown in JUnit 4 tests?", 30),
        op.predict("Measure execution time for a Java method", "How do I time a method's execution in Java?", 30),
        op.predict("Value from last inserted row in DB", "How to get a value from the last inserted row?", 30),
        op.predict("Jaxb ignore the namespace on unmarshalling",
                   "How to ignore namespace during unmarshalling XML document?", 30),
        op.predict("How do you split a list into evenly sized chunks?",
                   "How do I divide an ordered list of integers into evenly sized sublists?", 30)
    ]
    end_time = time()
    print(end_time - start_time)
    with open(str(PathUtil.output_RankSVM_model_dir() / "new_result_example.json"), "w") as f:
        json.dump(res_list, f)

    # 根据8-2随机原则计算MRR分数，运行下面的代码的时候请将main函数下别的代码注释掉
    # op.get_score_for_random_8_2()
