# **Text Classification for High School Exam Questions**

## Highlights:
1. This is a **multi-class text classification (document classification)** problem.
2. The purpose of this project is to classify High School Exam Questions into some classes and **the number of classes is related to the data set**.

## demands:
1. You can solve this problem with a variety of **machine learning** algorithms.
2. The evaluation method is mainly based on **precision and recall**.


## Data:
Chinese exam questions of high school.
### Example:
* 1354263077  　  21   　 字音　    下列词语中加点的字,读音全都正确的一组是( ) A.尴 尬(ɡà) 口 讷(nà) 髭须(xī) 朔风(shuò) B.拾 掇(duo) 央 浼(měi) 规 矩(jù) 祈祷(qí) C.妥 当(dànɡ) 憎恶(zēnɡ) 滑 稽(jī) 吼 啸(xiào) D. 赍发(jī) 盘 缠(chan) 玷辱(diàn) 胭 脂(zhǐ)
* 1354307841  　  21 　   古诗词阅读　    阅读下面这首词,然后回答问题. 望江怨 送别 [清]万树 春江渺,断送扁舟过林杪①.愁云清未了,布帆遥比沙鸥小.恨残照,犹有一竿红.怪人催去早. [注]①杪:树梢. (1)这首词的前四句描写了怎样的送别场景? (2)怎样理解“怪人催去早“?请结合全词分析..


123是是是是是是是是 是是是是是

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;    * *`1354263077` is ID of the question.* </br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;    * *`21` is the ID of the dataset, so you can ignore it if you don't want to try different datasets.* </br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;    * *`字音` is one of the classes.*


## Train-test split:
In order to unify the standard, we use the questions whose ID end with 9 as the test set and the rest as the train set.

## Evaluation:
```ruby
def count_precision_recall_at_k(y_pred, y_true, k):
    """
    y_pred: [[ 1.3315865   0.71527897 -1.54540029 -0.00838385  0.62133597 -0.72008556]]
    y_true: [[0 0 1 1 0 0]
    """
    y_indices = y_pred.argsort()[:, -k:][:, ::-1]
    pre = 0.0
    rec = 0.0
    for i in range(len(y_true)):
        intersec_true = 0
        for j in y_indices[i]:
            intersec_true += y_true[i][j]
        true_total_count = np.count_nonzero(y_true[i] == 1)
        pred_total_count = len(y_indices[i])
        pre += intersec_true*1.0/pred_total_count
        rec += intersec_true*1.0/true_total_count
    return pre/len(y_true), rec/len(y_true)
```

## Baseline:

baseline   | pre_1 | rec_1 | pre_2 | rec_2 | pre_3 | rec_3
-----------| ------|-------|-------|-------|-------|------
baseline_1 | 81.05 | 76.59 | 48.63 | 88.69 | 33.77 | 92.35
baseline_2 | 85.27 | 80.84 | 49.31 | 90.40 | 33.73 | 92.67

*These baselines are the results of two different algorithms.*

## Reference:
[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)</br>
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)</br>
[A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820)</br>
[Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781)</br>
[Hierarchical Attention Networks for Document Classification](http://www.aclweb.org/anthology/N16-1174)

