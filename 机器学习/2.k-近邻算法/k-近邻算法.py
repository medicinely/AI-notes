#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 实例流程
#    [数据来源](https://www.kaggle.com/c/facebook-v-predicting-check-ins/data)
# 1. 数据集的处理
# 2. 分割数据集
# 3. 对数据集进行标准化
# 4. estimator流程进行分类预测


# In[1]:


from sklearn.model_selection import train_test_split #分割训练集和测试集
from sklearn.neighbors import KNeighborsClassifier   #k-近邻算法
from sklearn.preprocessing import StandardScaler    #特征标准化
import pandas as pd


# In[3]:


def knncls():
    """
    k-近邻预测用户签到位置
    """
    #1.读取数据
    data = pd.read_csv("./train.csv")

    #2.处理数据

        #2.1 缩小数据范围（少量的数据容易操作）
    data = data.query("x > 1.0 & x < 1.25 & y >2.5 & y < 2.75")
        #2.2 处理时间数据
    time_value = pd.to_datetime(data['time'],unit='s')
        #2.3 把日期格式转换成字典格式
    time_value = pd.DatetimeIndex(time_value)

    #3.构造一些特征
#     data.loc[:,['day']] = time.value.day
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday
        #3.1 把时间戳特征删除
    data = data.drop(['time'],axis=1)
        #3.2 把签到数量少于n个目标位置删除
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()  #reset_index()把索引设置为列
    data = data[data['place_id'].isin(tf.place_id)]
        #3.3 删除无关特征
    data = data.drop(['row_id'],axis=1)
        #3.3 取出数据中的特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id'],axis=1)
        #3.4 进行数据的分割训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

    #4.特征工程（标准化）
    std = StandardScaler()
        #4.1 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)


    #5. 进行算法流程（超参数）
    knn = KNeighborsClassifier(n_neighbors=5)  # n_neighbors 为 k值

    #5.1 写入训练集数据
    knn.fit(x_train,y_train)

    #5.2 得到预测结果
    y_predict = knn.predict(x_test)
    print("预测的目标签到位置为：",y_predict)

    #得出准确率
    print("预测的准确率：",knn.score(x_test,y_test))
    return None

knncls()


# # 处理后的问题
# 
# 1. k值取多大？有什么影响
#     * k值取很小：容易受异常点影响
#     * k值取很大：容易受k值数量（类别）波动
#     
# 2. 性能问题？
#     * 性能较低
#     
# # k-近邻算法优缺点
# 
# * 优点
#     * 简单，易于理解，易于实现，`无需估计参数，无需训练`
#     
# * 缺点
#     * 懒惰算法，对测试样本分类时的计算量大，内存开销大
#     * 必须指定k值，k值选择不当则分类精度不能保证
#    
# * 使用场景：小数据场景，几千～几万样本，具体场景具体业务去测试
# 
# # 特征工程的相关知识点
#    * 数值型数据（标准缩放）：
#        1. 标准化：通过对原始数据进行变换，把数据变换到均值为0，方差为1范围内
#        2. 归一化：多个特征同等重要的时候需要进行归一处理
#            * 目的：使得某一个特征对最终结果不会造成更大影响
#            * 特点：通过对原始数据进行变换把数据映射到（默认为【0,1】）之间
#            * 公式：x1 = （x-min）/(max-min)  x2 = x1*(mx -mi) + mi
#        3. 缺失值
#    * 类别型数据：one-hot编码
#    * 时间类型：时间的切分
