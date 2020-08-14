# TCPapers
Worth-reading papers and related resources on text classification.

Suggestions about fixing errors or adding papers, repositories and other resources are welcomed!

文本分类领域值得一读的论文与相关资源集合。

欢迎修正错误以及新增论文、代码仓库与其他资源等建议！

## Papers
- **Convolutional Neural Networks for Sentence Classification**. *Yoon Kim*. (EMNLP 2014) [[paper]](https://arxiv.org/abs/1408.5882) - ***TextCNN***
- **Recurrent Neural Network for Text Classification with Multi-Task Learning**. *Pengfei Liu, Xipeng Qiu, Xuanjing Huang*. (IJCAI 2016) [[paper]](https://arxiv.org/abs/1605.05101) - ***TextRNN***
- **Recurrent Convolutional Neural Networks for Text Classification**. *Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao*. (AAAI 2015) [[paper]](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf) - ***TextRCNN***
- **Bag of Tricks for Efficient Text Classification**. *Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov*. (EACL 2016) [[paper]](https://arxiv.org/abs/1607.01759) - ***FastText***
- **Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification**. *Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, Bo Xu*. (ACL 2016) [[paper]](https://www.aclweb.org/anthology/P16-2034/) - ***Attn-BiLSTM***
- **Hierarchical Attention Networks for Document Classification**. *Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy*. (NAACL 2016) [[paper]](https://www.aclweb.org/anthology/N16-1174/) - ***HAN***
- **Enhancing Local Feature Extraction with Global Representation for Neural Text Classification**. *Guocheng Niu, Hengru Xu, Bolei He, Xinyan Xiao, Hua Wu, Sheng Gao*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1047/) [[code]](https://github.com/cdbgogo/Encoder1-Encoder2) - ***GELE***
- **How to Fine-Tune BERT for Text Classification?**. *Chi Sun, Xipeng Qiu, Yige Xu, Xuanjing Huang*. (CCL 2019) [[paper]](https://arxiv.org/abs/1905.05583)[[code]](https://github.com/xuyige/BERT4doc-Classification)

### Label Embedding
- **Joint Embedding of Words and Labels for Text Classification**. *Guoyin Wang, Chunyuan Li, Wenlin Wang, Yizhe Zhang, Dinghan Shen, Xinyuan Zhang, Ricardo Henao, Lawrence Carin*. (ACL 2018) [[paper]](https://arxiv.org/abs/1805.04174)[[code]](https://github.com/guoyinwang/LEAM) - ***LEAM***
- **Multi-Task Label Embedding for Text Classification**. *Honglun Zhang, Liqiang Xiao, Wenqing Chen, Yongkun Wang, Yaohui Jin*. (EMNLP 2018) [[paper]](https://arxiv.org/abs/1710.07210) - ***MTLE***
- **Explicit Interaction Model towards Text Classification**. *Cunxiao Du, Zhaozheng Chin, Fuli Feng, Lei Zhu, Tian Gan, Liqiang Nie*. (AAAI 2019) [[paper]](https://arxiv.org/abs/1811.09386)[[code]](https://github.com/NonvolatileMemory/AAAI_2019_EXAM) - ***EXAM***
- **GILE: A Generalized Input-Label Embedding for Text Classification**. *Nikolaos Pappas, James Henderson* (TACL Volumn 7 2019) [[paper]](https://transacl.org/ojs/index.php/tacl/article/view/1550)[[code]](https://github.com/idiap/gile)

## Survey & Tutorial
- **Deep Learning Based Text Classification: A Comprehensive Review**. *Shervin Minaee, Nal Kalchbrenner, Erik Cambria, Narjes Nikzad, Meysam Chenaghlu, Jianfeng Gao*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.03705)

## Repository & Tool
- [649453932 / Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch) - 开箱即用的基于PyTorch实现的中文文本分类
- [AnubhavGupta3377 / Text-Classification-Models-Pytorch](https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch) - Implementation of State-of-the-art Text Classification Models in Pytorch
- [brightmart / bert_language_understanding](https://github.com/brightmart/bert_language_understanding) - Pre-training of Deep Bidirectional Transformers for Language Understanding: pre-train TextCNN
- [brightmart / text_classification](https://github.com/brightmart/text_classification) - All kinds of text classification models and more with deep learning
- [chenyuntc / PyTorchText](https://github.com/chenyuntc/PyTorchText) - 1st Place Solution for Zhihu Machine Learning Challenge
- [dennybritz / cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf) - Convolutional Neural Network for Text Classification in Tensorflow
- [linhaow / TextClassify](https://github.com/linhaow/TextClassify) - 基于预训练模型的文本分类模板，CCF BDCI新闻情感分析初赛A榜4/2735，复赛1%
- [luopeixiang / textclf](https://github.com/luopeixiang/textclf) - 基于Pytorch/Sklearn的文本分类框架
- [songyingxin / TextClassification](https://github.com/songyingxin/TextClassification) - Pytorch + NLP, 一份友好的项目实践仓库
- [songyingxin / Bert-TextClassification](https://github.com/songyingxin/Bert-TextClassification) - Implemention some Baseline Model upon Bert for Text Classification
- [Tencent / NeuralNLP-NeuralClassifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier) - An Open-source Neural Hierarchical Multi-label Text Classification Toolkit
- [Vincent131499 / TextClassifier_Transformer](https://github.com/Vincent131499/TextClassifier_Transformer) - 个人基于谷歌开源的BERT编写的文本分类器
- [ZhengZixiang / OpenTC](https://github.com/ZhengZixiang/OpenTC) - Exploring various text classification models based on PyTorch

## Posts
- [Paperweekly / label-embedding在文本分类中的应用](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/107872873)
- [JayLou娄杰 / 打破BERT天花板：11种花式炼丹术刷爆NLP分类SOTA！](https://mp.weixin.qq.com/s/kQ-PiiLlDSyimixy_Pbe-g)
