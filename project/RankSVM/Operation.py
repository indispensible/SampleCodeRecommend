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
        self.TFIDF_operation = TFIDFOperation()
        sample_code_doc: MultiFieldDocumentCollection = MultiFieldDocumentCollection.load(sample_code_doc_path)
        self.sentence_2_code = {}
        self.w2v = W2VOperation()
        self.func_verb_net = FuncVerbClassifier()
        for item in sample_code_doc.get_document_list():
            self.sentence_2_code[item.get_doc_text_by_field("title_ori")] = item.get_doc_text_by_field("code")
        if os.path.exists(str(PathUtil.output_RankSVM_model_dir() / "rank_svm.model")):
            self.load()

    def train_rank_svm(self, sentence_list, model_path=str(PathUtil.output_RankSVM_model_dir() / "rank_svm.model")):
        # TODO: 改成从原始句子得到所有特征分数的版本
        # x的特征是：点赞数/100、tfidf分数、w2v分数、funcverb所属的类别、funcverb所属类别的分数
        if sentence_list is None:
            sentence_list = pickle.load(open(str(PathUtil.output_RankSVM_model_dir() / "origin.sentence"), "rb+"))

        X_list = []
        Y_list = []
        for item in sentence_list:
            new_x_list = []
            new_y_list = []
            new_x_list.append([
                item["positive_vote"] / 100,
                np.float(item['positive_score']),
                item['positive_w2v_score'],
                item['positive_func_label'],
                item['positive_func_score']
            ])
            new_y_list.append(np.float(item['positive_score']) + 10)
            for negative_item in item['negative_sentence_list']:
                new_x_list.append([
                    negative_item["vote"] / 100,
                    np.float(negative_item['score']),
                    negative_item['w2v_score'],
                    negative_item['negative_func_label'],
                    negative_item['negative_func_score']
                ])
                new_y_list.append(np.float(negative_item['score']))

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

    def predict(self, sentence, target_sentence="", n=30):
        predict_dict = {
            "question": sentence,
            "target": target_sentence,
            "rank_10_result": [],
            "code_10_result": []
        }
        rank_n_sentence = self.TFIDF_operation.predict_n_rank(sentence, n)
        x = []
        # vote数和TFIDF分数差距太大了，所以vote要除以100，降低参数的影响
        for item in rank_n_sentence:
            item_sentence = item[0][0]
            w2v_score = self.w2v.text_semantic_sim(sentence, item_sentence)
            func_label, func_score = self.func_verb_net.predict(item_sentence)
            x.append([item[0][2] / 100, item[1], w2v_score, func_label, func_score])
        x = np.array(x)
        sort_index = self.rank_svm_model.predict(x)
        sort_dict = dict(zip(sort_index, rank_n_sentence))
        print("Question: " + sentence)
        print("-" * 40)
        print("前十名结果:")
        for i in range(n - 1, n - 11, -1):
            predict_dict["rank_10_result"].append(sort_dict[i][0][0])
            predict_dict["code_10_result"].append(self.sentence_2_code[sort_dict[i][0][0]])
            print(sort_dict[i][0][0])
        print("=" * 60)
        return predict_dict

    def score(self, sentence_list=None, rank_n=30):
        if sentence_list is None:
            sentence_list = []
        score_list = []
        rank_before_3_score_list = []
        rank_before_5_score_list = []
        rank_before_10_score_list = []
        for sentence, target_sentence in tqdm(sentence_list):
            rank_n_sentence = self.TFIDF_operation.predict_n_rank(sentence, rank_n)
            x = []
            for item in rank_n_sentence:
                w2v_score = self.w2v.text_semantic_sim(sentence, item[0][0])
                func_label, func_score = self.func_verb_net.predict(item[0][0])
                x.append([item[0][2] / 100, item[1], w2v_score, func_label, func_score])
            x = np.array(x)
            sort_index = self.rank_svm_model.predict(x)
            sort_dict = dict(zip(sort_index, rank_n_sentence))
            rank = 0
            for key, item in sort_dict.items():
                if target_sentence == item[0][0]:
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
                else:
                    rank_before_3_score_list.append(0)

                if rank <= 5:
                    rank_before_5_score_list.append(1 / rank)
                else:
                    rank_before_5_score_list.append(0)

                if rank <= 10:
                    rank_before_10_score_list.append(1 / rank)
                else:
                    rank_before_10_score_list.append(0)

        print(sum(score_list) / len(score_list))
        print(sum(rank_before_3_score_list) / len(rank_before_3_score_list))
        print(sum(rank_before_5_score_list) / len(rank_before_5_score_list))
        print(sum(rank_before_10_score_list) / len(rank_before_10_score_list))


if __name__ == "__main__":
    op = SVMOperation()

    # 训练RankSVM模型
    op.train_rank_svm()

    # op.load()

    # op.get_example(100, use_es_operation=True)

    # res_list = [
    #     op.predict("how to always round up to the next integer", "How to round up the result of integer division?", 50, True, True),
    #     op.predict("How to play a sound (alert) in a java application?", "How can I play sound in Java?", 50, True, True),
    #     op.predict("Debug exceptions in AWT queue thread", "How can I catch AWT thread exceptions in Java?", 50, True, True),
    #     op.predict("JUnit 4 Expected Exception type",
    #                "How do you assert that a certain exception is thrown in JUnit 4 tests?", 50, True, True),
    #     op.predict("Measure execution time for a Java method", "How do I time a method's execution in Java?", 50, True, True),
    #     op.predict("Value from last inserted row in DB", "How to get a value from the last inserted row?", 50, True, True),
    #     op.predict("Jaxb ignore the namespace on unmarshalling",
    #                "How to ignore namespace during unmarshalling XML document?", 50, True, True),
    #     op.predict("How do you split a list into evenly sized chunks?",
    #                "How do I divide an ordered list of integers into evenly sized sublists?", 50, True, True)
    # ]
    # with open(str(PathUtil.output_RankSVM_model_dir() / "result_example.json"), "w") as f:
    #     json.dump(res_list, f)
