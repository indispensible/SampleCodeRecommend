# SampleCodeRecommend
## 项目结构
结构说明
```
project_root                           整个代码项目的根目录
│   README.md                          对于整个项目的介绍
│   .gitignore                         对于某些文件和目录让Git忽略管理
│   requirements.txt                   声明整个项目依赖的Python库
│   definitions.py                     定义一个ROOT_DIR的常量作为项目根目录。
│                      
└───data                               存放训练数据集、测试数据集、样例代码数据集以及模型
│
└───project                            样例代码推荐核心代码目录
│   │                   
│   └───RankSVM
|   |   |
|   |   └───Operation.py               实现样例代码推荐的核心代码，包含模型训练和随机8-2原则选取训练数据集与测试数据集来测试模型效果
|   |   └───RankSVM.py                 RanKSVM模型的实现代码
│   │                   
│   └───TFIDF   
|   |   |
|   |   └───Operation.py               提供根据输入的句子来得到相关问题等功能
|   | 
│   └───Word2Vec   
|   |   |
|   |   └───Operation.py               提供根据两个句子得到他们的相似度等功能
│   |
|   └───FuncVerbClassifier.py          提供根据输入的句子得到其所属功能类别和分数的功能
|   |
|   └───Model.py                       训练TFIDF模型以及Word2Vec模型
|
└───util                               工具文件目录
│   │                   
│   └───path_util.py                   各个文件的导入导出路径
|

```

## 样例代码推荐程序运行

1. 安装依赖
```
  >>> pip install -r requirements.txt
```

2. 训练TFIDF模型以及增量训练Word2Vec模型
```
  >>> python3 -m project.Model
```

3. 推荐效果测试，会将测试用例结果保存到data/RankSVM_model/result_example.json文件下(注：project/RankSVM/Operation.py的215到233行一起运行和236行分开运行（分开注释）)
```
  >>> python3 -m project.RankSVM.Operation
```

注：所有的命令在项目根目录下运行