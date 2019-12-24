# few_shot_learning

## Environment

下面是实验环境，包括系统版本以及python相关的包及其版本

```
System:
    ubuntu: 16.04

Python libs:
    python: 3.5
    pytorch: 1.3.1
    sklearn: 0.20.3
    numpy: 1.16.3
```

## dataset

包含115个类别，但是该数据集中存在一些问题，比如类别对应样本数量少于10的类别数目有50个，不适合用来做小样本学习的数据集。

http://disi.unitn.it/moschitti/corpora.htm


包含12294多个类别。但是给出的是instance的向量，非文本

http://lshtc.iit.demokritos.gr/

百度百科问答数据集 有类别

点此下载：https://pan.baidu.com/s/12TCEwC_Q3He65HtPKN17cA 密码:fu45



## Result

|model|finetune|Optim|5-way 1-shot| 5-way 5-shot| 5-way 10-shot|
|--|--|--|--|--|--|
|SimNet      | | | | | |
|MatchingNet | | | | | |
|ProtoNet    | | | | | |
|Proto-HATT  | | | | | |
|RelationNet | N | Adam| | | |
|InductionNet| | | | | |


# Reference Paper

[1]

[2]

[3]

[4]

[5]

[6] Few-Shot Text Classification with Induction Network

# Reference Code