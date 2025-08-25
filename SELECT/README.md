# AmazaonQA Project
DSTC11的项目工程过于混乱，为了便于管理，新的数据集在这里进行管理。
## bin Folder
存放脚本文件

## checkpoint Folder
存放训练好的模型

## data Folder
数据集
* ReDial
    * ReDial_Orignal 原始数据集
    * ReDial_Select 划分后的数据集，划分规则：保留test set，val set len与test set一致，剩下的train按4:1划分为train,candidate
    * ReDial_Postprocess 处理后的ReDial_Select
## result Folder
模型推理输出文件

## score Folder
模型推理得分文件

## script Folder
预处理代码，模型训练代码，模型推理代码，得分计算代码
* dataProcess
    * dataAnalyse.ipynb 
    * dataDivide.ipynb 数据划分代码
    * preProcess.py 将划分好的数据做进一步处理，符合之后训练模型要求。
        * 处理规则
            * 合并连续发言人
            * 去掉ending，即最后一个turn（sn,sn-1）
            * 去掉开场白 s1
            * reference: sn-3
            * sn-2 :sn-2是提问者对reference的评价 (不需要它)
            * sn-2的发言人A就是提问人，那么相应的另一个对话角色B就是表示系统
            * 往前倒推，A的第一句话就是他的意图，就是QUERY
    * buildVectorLib.ipynb 为SELECT构建的向量库

* modelTrain
* modelInference
* scoreCal

## log Folder
各类日志文件
