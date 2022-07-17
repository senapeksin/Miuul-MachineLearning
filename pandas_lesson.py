# Pandas Series ve Pandas DataFrame
# Pandas serileri tek boyutlu ve index bilgisi bulunduran bir veri yapısıdır.
# Pandas dataframe 'i ise çok boyutlu ve index bilgisi bulundan veri yapısıdır.

############################## PANDAS SERIES ##############################################
import pandas as pd
s = pd.Series([10, 77, 12, 4, 5])   # Series metodu bir liste ister ki onu pandas serisine cevirir.
# 0    10
# 1    77
# 2    12
# 3     4
# 4     5
# dtype: int64
type(s)     # pandas.core.series.Series

s.index     # index bilgilerine ulaşabiliriz.
s.dtype     # içindeki elemanları tipini verir
s.size      # pandas serisinin eleman sayısını verir
s.ndim      # pandas serileri tek boyutludur. Serinin boyut bilgisini verir
s.values    # içindeki değerlere erişmiş oluruz (numpy array döndü)
s.head(3)   # içerisindeki ilk  3 degeri getirme
s.tail(3)   # sondan 3 değeri getirme

############################## VERİ OKUMA  ##############################################

import pandas as pd
# csv dosyasını okumak
df = pd.read_csv("datasets/car_crashes.csv")
df.head()

############################## VERİYE HIZLI BAKIŞ  ##############################################

import seaborn as sns
df = sns.load_dataset("titanic")   #seaborn içerisinde bazı yaygın verisetleri var bunlara erişebiliyoruz.
df.head()
df.tail()
df.shape    # boyut bilgisine
df.info()   # değişkenler ve değişkenlerin tiplerine erişiriz
df.columns  #  eğişkenlerin isimlerine erişiriz
df.index    # index bilgisine erişiriz
df.describe().T  # özet istatistik sayısal bilgiyi verir. T dersek transpoz daha iyi okuma yapmamızı sağlar.
df.isnull().values.any()  #veri setinde eksik değer var mı kontrolü sağlar
df.isnull().sum()   # değişkenlerde kaç tane eksik değer var bilgisini hesaplar
df["sex"]  #dataframe üzerinden değişken seçmek istediğimizde bu sekilde yaparız.
df["sex"].value_counts() #kategorik değişkende kaç tane sınıf var bilgisine erişelim mesela kadın erkek kacı kadın kacı erkek?



############################## PANDAS SEÇİM İŞLEMLERİ ##############################################

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

#
df.index  #indexlerine gitmek istersek bu kodu kullanırız
df[0:13]  #0 dahil 13 dahil olmayan slice işlemi
df.drop(0, axis=0).head()  #index silme işlemi (satırlardan silmek)
delete_indexs= [1, 3, 5, 7]
df.drop(delete_indexs, axis=0).head()  #birden fazla index silmek

#inplace = True dersek değişiklilikleri kalıcı yap demiş oluruz.
# df.drop(delete_indexs, axis=0,inplace = True)

####### Değişkeni indexe çevirmek ######

df["weight"].head()
df.weight.head()   # ikiside seçim işlemidir

df.index = df["weight"]   # değişkeni index'e çevirdik

#indexe çevirdik, değişkenlerden kaldırmak istersek
df.drop("weight", axis=1).head() #axis=1 diyoruz cünkü satır değil sutun işlemi  yapıyoruz.
df.drop("weight", axis=1, inplace=True)  #inplace=True diyerek bu işlemi kalıcı hale getiriyoruz.

####### Index'i Değişkene  çevirmek ######
# 1. yöntem
df.index
# bir dataframe'e yeni değişken eklemek için o dataframe içerisinde olmayan bir isimlendirme girersek yeni bir değişken eklemiş oluruz
df["weight"] = df.index
df.head()

# 2. yol : reset_index metodu hem indexi silecektir ve sildiği indexi sutun olarak ekleyecektir
df = df.reset_index()
df.head()

############################## DEĞİŞKENLER(Sütunlar) ÜZERİNDE  İŞLEMLERİ ############################################

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
# verisetindeki ... noktadan kurtulmak istersek;
pd.set_option('display.max_columns', None)
df.head()

# bir dataframe'de herhangi bir değişikliğin varlıgını sorgulamak için;
"age" in df
df["age"].head()
df.age.head()

type(df["age"].head())  #çıktısı pandas.core.series.Series , bir pandas serisi geldi
# Bu tarz değişken seçimi yaptıgınızda pandas serisi gelecek. Tek bir değişken seçerken seçimin sonucunun
# dataframe olarak kalmasını istiyorsanız iki tane köşeli parantez kullanmak gerekir!!!!

df[["age"]].head()
type(df[["age"]].head())  # pandas.core.frame.DataFrame

# Birden fazla değişken seçmek istersek;
df[["age","alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]


# Dataframe 'e değişken ekleme (verisetinize değişken eklemek)
df["age2"] = df["age"] ** 2
df["age3"] = df["age"] / df["age2"]
# sütun silmek istiyorsak; aşağıdaki işlem kalıcı olmayacaktır.
df.drop("age3", axis=1).head()
# Birden fazla değişken silmek için;
df.drop(col_names, axis=1).head()

# Verisetinde belirli bir string ifadeye göre değişkenleri silmek için
# nasıl yakalayacağız?
df.loc[:, df.columns.str.contains("age")]  # içerisinde age olanları getirdi
# bunları silmek istersek;
df.loc[:,  ~df.columns.str.contains("age")]


############################## LOC & ILOC ############################################

# Loc ve Iloc : Datafamelerde seçim işlemleri için kullanılır
# İloc : integer based selection : İndex bilgisi vererek seçim yapma işlemini sağlar.
# Loc : label based selection  : indexte etiket nasılsa onu seçer . İsimlendirmenin kendisini seçiyor.
# İndexlerdeki labellara göre işlem yapar

df.iloc[0:3]
df.iloc[0, 0]

df.loc[0:3]  # label base selection yaptı ve 3. indexi de getirdi farkı budur.!!!! isimlendirmenin kendisini seçiyor

df.iloc[0:3, "who"]  # hata verir. iloc integer base oldugu için string ifadeyi kabul etmez
df.iloc[0:3, 0:3]   # doğrusu bu sekildedir


df.loc[0:3, "who"]  # doğrusu budur(çünkü string deger ister)

# birden fazla değişkeni de isimlerini söyleyerek seçebiliriz.
col_names = ["age2", "age3"]
df.loc[0:3, col_names]


################################### KOŞULLU İŞLEMLER #########################################

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
pd.set_option('display.max_columns', None)
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50].count()   # yaşı 50'den büyük kaç kişi var ?
df[df["age"] > 50]["age"].count()   # çıktıda herhangi bir değişken girerek sayıya ulaşırız

# koşula göre değişken seçmek
df.loc[df["age"] > 50, "class"].head()   #koşul ve koşulun değişken degelerini görmek istersek. koşul ve değişken seçmek
df.loc[df["age"] > 50, ["class", "age"]].head()   #koşul ve 2 değişken seçmek

#yaşı 50'den büyük erkekleri seçmek istiyoruz.
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()
#birden fazla koşul gireilecekse parantez içine almak gerekir!!!!

df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & (df["embark_town"] == "Cherbourg"),
       ["age", "class", "embark_town"]].head()

# yada kullanmak
df_new = df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]].head()

df_new["embark_town"].value_counts()

###################################  TOPLULAŞTIRMA VE GRUPLAMA (AGGREGATION § GROUPING) #########################################

# Toplulaştırma : Bir veri yapısının içinde bulunan degerleri toplu bi şekilde temsil etmek (Özet istatiktikler veren fonksiyonlardır)
# Toplulaştırma fonksiyonları:
#  -count()
#  -first()
#  -last()
#  -mean()
#  -median()
#  -min()
#  -max()
#  -std()
#  -var()

# Gruplama: Bu toplulaştırma fonksiyonlarını group_by fonksiyonu ile gruplayabiliriz.

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
pd.set_option('display.max_columns', None)
df.head()

# kadın ve erkeklerin yaş ortalamasına erişmek istiyorum

df["age"].mean() # dersek yaşın ortalamasını almış oluruz.
# cinsiyete göre gruplayıp yaş ortalamasını aldık.
df.groupby("sex")["age"].mean()   #hesaplanmak istenilen değişkene göre(age) aggregation fonk. uygulanır.
# sex
# female    27.915709
# male      30.726645

# cinsiyete göre gruplayıp yaş ortalaması ve toplamını alalım

df.groupby("sex").agg({"age": "mean"})  # bu kullanım yukarıdakine göre daha iyidir.

df.groupby("sex").agg({"age": ["mean", "sum"]})  # daha fazla fonksiyon kullanabiliriz.

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

# sadece cinsiyete göre değil , diğer değişkenlere göre de yapalım


df.groupby(["sex", "embark_town"]).agg({"age": ["mean", "sum"],
                                        "survived": "mean"})

# 3 kırımlı gruba göre yaş ve hayatta kalma olasılıkları:
df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean", "sum"],
                                                  "survived": "mean"})


###################################  PIVOT TABLE #########################################

# Group by işlemlerine benzer şekilde verisetini kırılımlar açısından değerlendirmek ve
# ilgilendiğimiz özet istatiği bu kırılımlar açısından görmemizi sağlar.

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
pd.set_option('display.max_columns', None)
df.head()

pd.set_option('display.width', 500)

# pivot table default olarak ortalamasını(mean) alır. bunu değiştirebiliriz.

# yaş ve embarked açısından survived bilgisine erişmek isteyelim
# pivot table fonksiyonu aşağıdaki gibi kullanılır;
# pivot_table(kesişimlerde neyi görmek istiyorsun?, satırda hangi değişkeni görmek istiyorsun?, sütunda hangi değişkeni görmek istiyorsun?)

df.pivot_table("survived", "sex", "embarked")
# embarked         C         Q         S
# sex
# female    0.876712  0.750000  0.689655
# male      0.305263  0.073171  0.174603

# bu kesişimde değişkenlerin standart sapmasını hesapladı
df.pivot_table("survived", "sex", "embarked", aggfunc="std")


df.pivot_table("survived", "sex", ["embarked", "class"])

# sayısal degişkenleri ayırma mesela age = kadın age / erkek age şeklinde ayırmak
# yaş sayısal değişkenini kategorik değişkene çevirmek;

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

# elinizdeki sayısal değişkenleri kategorik değişkene çevirebilmek için cut ve qcut kullanılır.
# Eğer sayısal değişkeni neye göre kategorize edeceğinizi biliyorsanız cut, bilmiyorsanız qcut kullanılır.
# cut(neyi bölücem? , nerelerden bölücem?) der.

df.pivot_table("survived", "sex", "new_age")  # yaş ve cinsiyet kırılımında hayatta kalma olasılıkları

###################################  APPLY &  LAMBDA #########################################

# Apply : Satır ve sütunlarda otomotik olarak fonksiyonları çalıştırmayı sağlar
# Bir döngü yazmadan değişkenler içerisinde dolanmak
# Lambda : Kullan at fonksiyondur. Fonksiyon tanımlamadan kullan at fonksiyon yapılarıdır..

# 2 yeni değişken oluşturalım
df["age2"] = df["age"] ** 2
df["age3"] = df["age"] ** 5

# Amacımız bu veeri seti içerisindeki yaş değişkenlerinin 10'a bölünmesini istiyoruz. Normalde bunu tek tek yaparız veya döngü uygularız

# değişkenlere fonksiyon uygulamak istiyorum. O zaman bunu tek tek yaparsak aşağıdaki gibi olur
df["age"] / 10
df["age2"] / 10
(df["age3"] / 10).head()   # head uygulamak istersek paranteze almamız lazım!

# Değişkenlere fonksiyon uygulamak istiyorum ama çok değişken var , döngü yazmalıyız.
# Bu işlemi döngü ile yaparsak;
for col in df.columns:
       if "age" in col:
              print(col)

# bütün değerleri 10' a bölelim

for col in df.columns:
       if "age" in col:
              print((df[col]/10).head())

# Bu durumda biz bu 10 ile bölünmüş değerleri kaydetmedik. Kaydetmemiz lazım bunun için;
for col in df.columns:
       if "age" in col:
              df[col] = df[col] / 10
df.head()

df[["age", "age2", "age3"]].apply(lambda x:x / 10).head()

df.loc[:, df.columns.str.contains("age3")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age3")].apply(lambda x: (x- x.mean()) / x.std()).head()

# dışardan fonksiyonda verebiliriz.

def standart_scaler(col_name):
       return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age3")].apply(standart_scaler).head()

###################################  BİRLEŞTİRME İŞLEMLERİ  #########################################

# iki şekilde birleştirme işlemi yapılabilir.Concat ve merge olarak yapabiliriz.
# Concat: Elimizde iki tane dataframe var diyelim. Bunları birleştirebiliriz.

import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])  # 3 sütunlu olsun dediğimiz için 3 kolon ismi ile dataframe oluşturuyoruz.
df2 = df1 + 99

pd.concat([df1, df2])  # alt alta birleşmiş oldu , fakat index sayısı düzgün artmıyor. Bunun için;

pd.concat([df1, df2], ignore_index=True)  # diyerek index sayısını düzzgün arttırdık.

# Sütunlara göre (yan yana )de birleştirme yapabiliriz. DEfault olarak satırlara göre yapar(alt alta)

#MERGE ile birleştirme işlemleri;

df1 = pd.DataFrame({'employees': ['John', 'dennis', 'mark', 'maris'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']
                    })

df2 = pd.DataFrame({'employees': ['John', 'dennis', 'mark', 'maris'],
                    'start_date': [2020, 2019, 2018, 2022]
                    })
# Her çalışanın başlangıc tarihine erişmek istiyoruz.

pd.merge(df1, df2)  #yan yana birleştirme işlemi yapar.

# hangi değişkene göre yap demedik ama employess 'a göre yaptı. Bunu söyleyedebiliriz

pd.merge(df1, df2, on="employees")

# Amacımız: her çalışanın müdür bilgisine ulasalım.

df3 = pd.merge(df1, df2, on="employees")

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Sena', 'Mustafa']
                    })

pd.merge(df3, df4)