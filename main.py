#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------
@Author: Lv Gang
@Email: 1547554745@qq.com
@Created: 2021/05/20
------------------------------------------
@Modify: 2021/05/20
------------------------------------------
@Description:
"""
from project.RankSVM.Operation import SVMOperation
from util.path_util import PathUtil

print("开始导入模型！")
op = SVMOperation()
op.load(str(PathUtil.output_RankSVM_model_dir() / "rank_svm.final.model"))
print("模型导入完毕，可以开始输入了！")

while True:
    print("Question: ")
    question = input()
    op.predict(question, print_code=True)
