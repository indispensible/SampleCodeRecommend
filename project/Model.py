#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------
@Author: Lv Gang
@Email: 1547554745@qq.com
@Created: 2021/05/17
------------------------------------------
@Modify: 2021/05/17
------------------------------------------
@Description:
"""
import pickle

from gensim import similarities
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk.corpus import stopwords
from sekg.ir.doc.wrapper import MultiFieldDocumentCollection

from project.Word2Vec.Operation import W2VOperation
from util.path_util import PathUtil


class Model:

    def __init__(self, doc_path):
        self.sample_code_doc: MultiFieldDocumentCollection = MultiFieldDocumentCollection.load(doc_path)
        self.sentence_list = []

    # 根据tiidf的预料训练一个稀疏矩阵相似度模型
    def train_TFIDF_model(self, version):
        dataset = []
        tmp_sentence_list = []
        stop_words = stopwords.words('english')
        for sentence, vote in self.sentence_list:
            sentence_2_word_list = [item for item in sentence.split(" ") if item.lower() not in stop_words]
            if sentence_2_word_list:
                tmp_sentence_list.append([sentence, vote])
                dataset.append(sentence_2_word_list)
        dct = Dictionary(dataset)
        corpus = [dct.doc2bow(line) for line in dataset]
        tfidf_model = TfidfModel(corpus, dictionary=dct)
        trained_tfIdf_model_path = PathUtil.output_trained_tfIdf_model_dir() / "tfidf.{v}.model".format(v=version)
        dict_path = PathUtil.output_trained_tfIdf_model_dir() / "{v}.dict".format(v=version)
        dct.save(str(dict_path))
        tfidf_model.save(str(trained_tfIdf_model_path))
        pickle.dump(corpus, open(str(PathUtil.output_corpus_tfIdf_model_dir() / "tfidf.corpus"), "wb+"))
        pickle.dump(tmp_sentence_list,
                    open(str(PathUtil.output_document_tfIdf_model_dir() / "tfidf.sentence_vote_list"), "wb+"))
        index = similarities.SparseMatrixSimilarity(tfidf_model[corpus], num_features=len(dct.items()))
        index.save(str(PathUtil.output_index_model_dir() / 'sparse_matrix_similarity.index'))
        print("TFIDF训练完毕")

    # corpus的格式[["a", "is", "apple"],["b", "like", "apple"]]
    def trained_word2vec_model(self, pre_trained_w2v_path=None, tuned_word_embedding_save_path=None):
        stop_words = stopwords.words('english')
        self.get_sentence_list()
        corpus = [[item.lower() for item in sentence.split(" ") if item.lower() not in stop_words] for sentence, post_id in
                  self.sentence_list]
        print("开始训练word2vec")
        pre_trained_word2vec_model = Word2VecKeyedVectors.load_word2vec_format(pre_trained_w2v_path, binary=True)

        w2v = Word2Vec(window=5, min_count=1, size=300, iter=3, alpha=0.001)
        w2v.build_vocab(corpus)
        training_examples_count = w2v.corpus_count
        w2v.build_vocab([list(pre_trained_word2vec_model.vocab.keys())], update=True)

        w2v.intersect_word2vec_format(pre_trained_w2v_path, binary=True, lockf=1.0)
        w2v.train(corpus, total_examples=training_examples_count, epochs=w2v.epochs)
        w2v.wv.save(tuned_word_embedding_save_path)
        print("训练完毕")

    def get_sentence_list(self):
        multi_document_list = self.sample_code_doc.get_document_list()
        for item in multi_document_list:
            if isinstance(item.field_doc["title_ori"], str) and item.field_doc["url"].split("/")[-1]:
                self.sentence_list.append([item.field_doc["title_ori"], item.field_doc["vote"]])


if __name__ == "__main__":
    sample_code_doc_path = PathUtil.doc("sample", "v1.1")
    model = Model(sample_code_doc_path)
    model.get_sentence_list()
    model.train_TFIDF_model("v1")
    # W2VOperation.txt_2_bin(str(PathUtil.wiki_emb_path() / "300" / "enwiki_20180420_300d.txt"),
    #                        str(PathUtil.wiki_emb_path() / "300" / "enwiki_20180420_300d.bin"))
    pre_w2v_path = str(PathUtil.wiki_emb_path() / "300" / "enwiki_20180420_300d.bin")
    new_w2v_path = str(PathUtil.wiki_emb_path() / "300" / "new_enwiki_with_title.bin")
    model.trained_word2vec_model(pre_w2v_path, new_w2v_path)
