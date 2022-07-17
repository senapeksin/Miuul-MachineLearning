##################################### VERİ GÖRSELLEŞTİRME #####################################
##################################### Matplotlib & Seaborn #####################################

# Matplotlib : Veri görselleştirmenin babasıdır.
# Seaborn : High leveldir. Daha az caba daha cok iş yapar.

# Kategorik değişken varsa : sütun grafik. countplot veya bar kullanılır.
# Sayısal değişken varsa : hist, boxplot kullanılır. (ikisi de dağılım gösterir)

############## KATEGORİK DEĞİŞKEN GÖRSELLEŞTİRME  #######################

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# Kategorik değişkenler ile işlem yapacağımızda aklımıza gelecek ilk fonk = value_counts()'dır.

# veya su sekilde de yazabiliriz.
df["sex"].value_counts().plot(kind='bar')
plt.show()

############## SAYISAL DEĞİŞKEN GÖRSELLEŞTİRME  #######################

# hist : elimizdeki sayısal değişkenin belli aralıklara göre dağılım bilgisini verir.
plt.hist(df["age"])
plt.show()

# boxplot : elimizdeki sayısal değişkenin belli aralıklara göre dağılım bilgisini verir.
# Veri setindeki aykırı değerleri çeyreklik değerler üzerinden gösterebilir.

plt.boxplot(df["fare"])
plt.show()

############## MATPLOTLIB ÖZELLİKLERİ  #######################

# Matplotlib yapısı gereği katmanlı şekilde veri görselleştirme imkanı sağlar.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()


# plot özelliği : veriyi görselleştirmek için kullandıgımız fonksiyon
x = np.array([1, 8])
y = np.array([8, 150])
plt.plot(x, y, 'o')
plt.show()
# daha fazla nokta olursa;
x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])
plt.plot(x, y, 'o')
plt.show()
# marker özelliği :işaretçi

y = np.array([13, 28, 11, 100])
plt.plot(y, marker='o')
plt.show()

plt.plot(y, marker='*')
plt.show()

# line özelliği : çizgi oluşturmaya olanak sağlar.

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dotted", color= 'red')
plt.show()

# multiple lines
x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dotted", color='red')
plt.plot(x)
plt.show()

# labels : başlıklar

plt.title("Bu ana başlık")
plt.xlabel("x ekseni isimlendirmesi")
plt.ylabel("y ekseni isimlendirmemiz")

plt.grid()
plt.show()

# SubPlots

#plot 1
x = np.array([23, 18, 31, 10, 200, 250, 400, 13])
y = np.array([13, 28, 11, 100, 45, 249, 350, 77])
plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)

#plot 2
x = np.array([23, 18, 31, 10, 200, 250, 400, 13])
y = np.array([13, 28, 11, 100, 45, 249, 350, 77])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)

############## SEABORN İLE VERİ GÖRSELLEŞTİRME   #######################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")
df.head()

#kategorik : kadın-erkek , günler => sutun grafik kullanıyoruz.

df["sex"].value_counts()
sns.countplot(x= df["sex"], data= df)
plt.show()

#matplotlib 'de bu grafiği şu sekilde çizdiririz.
df["sex"].value_counts().plot(kind='bar')
plt.show()


#SEARBORN İLE SAYISAL DEĞİŞKENLERİ GÖRSELLEŞTİRMEK

sns.boxplot(x= df["total_bill"])
plt.show()

df["total_bill"].hist() #pandas fonksiyonudur
plt.show()

# Sonuc olarak elimizde 3 temel yaklasım var
# 1 - value_count ve countplot
# 2 - hist
# 3 - boxplot  bu 3 'ünü kullanarak birçok görselleştirme yapılabilir.
