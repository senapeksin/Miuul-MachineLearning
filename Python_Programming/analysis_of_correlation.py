import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]  #id ve sonuncu anlamsız değişkenden kurtulmak için yapıyoruz.
df.head()
df.info()

# Amacımız; elimize bir veri seti geldiğinde bunun ısı haritası aracılıgıyla korelasyonlarına bakmak.
# Daha sonra yüksek korelasyonlu bir değişkenlerden bazılarını dışarda bırakmaktır. (her çalışmada yapılacak diye bir kural yoktur.)

# İlk olarak bu veri seti içerisindeki sayısal değişkenleri öncelikle seçmemiz lazım

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

corr = df[num_cols].corr()  #korelasyon hesaplama

# korelasyon: değişkenlerin birbirleriyle ilişkisini ifade eden bir istatistiksel ölcümdür. -1 ile +1 arasında değerler alır.
# -1 ve +1 ' e yaklaştıkca ilişkinin siddeti kuvvetlenir.

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

#######################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
#######################

cor_matrix = df.corr().abs() #korelasyonun negatif yada pozitif olması ile ilgilenmiyorum bundan dolayı mutlak değerini alarak pozitif hale getiriyorum


#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000

#olması gereken cıktı
#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN


upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)

#fonksiyon oluşturakım.
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()  #korelaayon olustur
    cor_matrix = corr.abs()  #mutlak değerini al
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

# Yaklaşık 600 mb'lık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/fraud_train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df, plot=True)

len(df.drop(drop_list, axis=1).columns)














