#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------
@Author: Lv Gang
@Email: 1547554745@qq.com
@Created: 2021/04/15
------------------------------------------
@Modify: 2021/04/15
------------------------------------------
@Description:
"""
from funcverbnet.sentence_classifier import FuncSentenceClassifier


class FuncVerbClassifier(FuncSentenceClassifier):

    def predict(self, sentence):
        try:
            label = self.classifier.predict(sentence)
            probability = label[1][0]
            if len(str(label[0][0])) > 10:
                label = str(label[0][0][9]) + str(label[0][0][10])
            else:
                label = str(label[0][0][9])
            return int(label), probability
        except Exception as e:
            print(e, sentence)
            return 0, 0
