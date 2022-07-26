# Amac: Verilerin ölçeklenebilir fonksiyonel tarzda işleyebilmeli amaçlar
# Hızlı bir şekilde genel fonksiyonlar ile veriyi analiz etmek.

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


df = sns.load_dataset("titanic")
df.head()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


df.shape  # satır sütun bilgisi
df.info()
df.isnull().values.any()  #eksik değer var mı sorgusu
df.columns
df.index
df.isnull().sum()  #kacar tane eksik deger var?

def check_df(dataframe, head=5):
    print("###### shape ####")
    print(dataframe.shape)
    print("###### types ####")
    print(dataframe.dtypes)
    print("###### head ####")
    print(dataframe.head(head))
    print("###### tail ####")
    print(dataframe.tail(head))
    print("###### eksik değerler ####")
    print(dataframe.isnull().sum())
    print("###### quantiles  ####")
    #print(dataframe.describe([0, 0.5, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

############################ KATEGORİK DEĞİŞKEN ANALİZİ ############################

#tek bir değişkeni value_ocunts kullanarak analiz ediyorduk. Fakat tüm değişkenlere tek tek yapamayız
df["embarked"].value_counts()   # sınıf sayısına ulaşabiliriz
df["sex"].unique()  # unique değerlerine ulaşabiliriz.
df["class"].nunique()  # kaç tane toplamda eşssiz değer var ?

# Birden fazla değişkene bu tarz genel fonksiyonları uygulayacağız?
# Veriseti içerisinden otomotik sekilde olası tüm kategorik değişkenleri seçen bir sey yapalım.
df.head()

# tip bilgisi bulalım
df.info()
# aslında bool , category , object veri tipleri kategoril değişkenlerdir.
# tip bilgisinden hareketle kategorik değişkenler 3 tanedir : bool , category, object
# birde sinsirella vardır örneğin survived, pclass değişkenleri. int tipinde gözüküyorlar fakat kategorik değişkenlerdir
# bunları tip bilgisi üzerinden yakalayamayız.


# Kategorik değişken yakalayacağız evet , birkaç problem var heö tip bilgisine göre seçim yapmamız lazım , hem de tip bilgisi farklı
# olmasına rağmen numeric ama kategorik olan değişkenleri yakalamamız lazım.

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
# ['sex',
#  'embarked',
#  'class',
#  'who',
#  'adult_male',
#  'deck',
#  'embark_town',
#  'alive',
#  'alone']

# sinsirella yakalama mesela survived, pclass (normalde kategorik ama int tipinde duruyor)
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
# ['survived', 'pclass', 'sibsp', 'parch']

# ölçülenemeyecek kadar sınıfı olanları ayırt edelim mesela isim soyisim unique 'dir.
# Öyle bir şey yapalım ki; progmatik olarak bu verisetindeki kategorik tipte oldugu halde kategorik olmayan değişkenleri yakalıyor olmamız lazım.
# Zaten int ve float değişkenlerin cok fazla unique değere sahip ise zaten sayısal değişkendir.


cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["int", "float"]]
# boş

cat_cols = cat_cols + num_but_cat  # tüm kategorileri birleştirelim

df[cat_cols]

# doğrulama
df[cat_cols].nunique()

# sayısal değişkenler nerede
[col for col in df.columns if col not in cat_cols]
# ['age', 'fare'] toplam 2 tane sayısal değikenimiz varmış

# Fonksiyon yazalım

# Bu fonksiyon şunları yapsın:
# değerlerin value_count() alsın , yani hangi sınıftan kaçar tane var
df["survived"].value_counts()
# sınıfların yüzdelik bilgisini yazdıralım
100 * df["survived"].value_counts() / len(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################")

cat_summary(df,"survived")
cat_summary(df, "pclass")
cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

# Bu derste yaptıklarımızı generic hale getiricez.

############################ KATEGORİK DEĞİŞKEN ANALİZİ 2 ############################

# cat_summary fonksiyonuna grafik bilgisi ekleyelim

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)  #birden fazla grafiği yönetmek

cat_summary(df, "sex", plot=True)

# tüm değişkenlere for ile dönelim
for col in cat_cols:
    if df[col].dtypes =="bool":
        print("nsdndnndwnnenwenen bboool value sjdjsjkdjksdjdk")
    else:
        cat_summary(df, col, plot=True)

# tüm değişkenlere for ile döndüğümüzde bazı tipi bool olan değişkenlerden hata alıyoruz. Bunu çözebilmek için;
# adult_male değişkeni bool tip değişkendir.
# ne yapıcaz ?
# yukarıda bool tipli olanları pas geç dedik ama bunu çözelim der isek ;
# bool tipi değiştirmek istiyoruz varsayalım

df["adult_male"]
# öyle bi sey yapmalıyız ki kullandıgımız fonksyionun kabul edeceği bir değişkene çevirmem lazım
df["adult_male"].astype(int)  # dediğimizde true olan yerler 1 , false degerler ise 0 oldu


for col in cat_cols:
    if df[col].dtypes =="bool":
        df[col] = df[col].astype(int)   # eğer değişken tipi bool ise tipi değiştir. Değişen tip ile özet bilgisi fonksiyonunu kullan
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

# burdaki dönüşümü en başta da yapabilirsiniz.

# Tüm işlemleri tek bir fonksiyon içinde yapsak ? kod okuması kotu olur

############################ SAYISAL DEĞİŞKEN ANALİZİ  ############################

# Yaş değişkenini incelemek isteyelim

df[["age", "fare"]].describe().T


num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
# ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']

#bazı değişkenler sayısal gözükse de kategoriktir mesela pclass.

num_cols = [col for col in num_cols if col not in cat_cols]
#  ['age', 'fare'] sayısal değişkenlerimiz geldi

def num_summary(dataframe, numeric_col):
      quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
      print(dataframe[[numeric_col]].describe(quantiles).T)

num_summary(df, "age")

#bütün sayısal değişkenler için yapalım.

for col in num_cols:
    num_summary(df, col)


#grafik özelliği ekleyelim
def num_summary(dataframe, numeric_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(dataframe[[numeric_col]].describe(quantiles).T)

    if plot:
        dataframe[numeric_col].hist()
        plt.xlabel(numeric_col)
        plt.title(numeric_col)
        plt.show(block=True)

num_summary(df, "age", plot=True)


for col in num_cols:
    num_summary(df, col, plot=True)


############################  DEĞİŞKENLERIN YAKALANMASI  ve İşlemlerin Genelleştirilmesi ############################

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# veri seti içerisindeki kategorik ve numeric değişkenleri ayrı ayrı getirsin

def grab_col_names(dataframe, cat_th=10, car_th=20):
   """
   Veri setindeki kategorik, numerik ve kategorik fakar kardinal değişkenlerin isimlerini verir.
   Parameters
   ----------
   dataframe : dataframe
        değişken isimleri alınmak istenen dataframe'dir.
   cat_th : int,float
        numeric fakat kategorik olan değişkenler sınıf eşik değeri
   car_th : int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

   Returns
   -------
    cat_cols : list
        Kategorik değişken listesi
    num_cols : list
        Numeric Değişken listesi
    cat_but_car : List
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols 'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir:
   """

   #cat_cols , cat_but_car
   cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
   num_but_cat = [col for col in df.columns if df[col].nunique() <10 and df[col].dtypes in ["int", "float"]]
   cat_but_car = [col for col in df.columns if
                  df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
   cat_cols = cat_cols + num_but_cat
   cat_cols = [col for col in cat_cols if col not in cat_but_car]


   num_cols = [col for col in df.columns if str(df[col].dtypes) in ["int", "float"]]
   num_cols = [col for col in num_cols if col not in cat_cols]

   print(f"Observations: {dataframe.shape[0]}")
   print(f"Variables: {dataframe.shape[1]}")
   print(f'cat_cols: {len(cat_cols)}')
   print(f'num_cols: {len(num_cols)}')
   print(f"cat_but_car: {len(cat_but_car)}")
   print(f"num_but_car: {len(num_but_cat)}")


   return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


############

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################")

cat_summary(df,"sex")

for col in cat_cols:
    cat_summary(df, col)

#numeric fonksşyonumuz
def num_summary(dataframe, numeric_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(dataframe[[numeric_col]].describe(quantiles).T)

    if plot:
        dataframe[numeric_col].hist()
        plt.xlabel(numeric_col)
        plt.title(numeric_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)



# bonus :  cat_summary fonksiyonu görsel özellikli olarak daha iyi kullanmak (bool değişkenini ilk başta değiştirmek)
df = sns.load_dataset("titanic")
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

df.info()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)  #birden fazla grafiği yönetmek

for col in cat_cols:
    cat_summary(df, col, plot=True)

#bool değişkenlerde cat_summary ile görselleşti
# bundan dolayı veri setini okuduktan sonra tip dönüşümü yapmak cok daha mantıklı

for col in num_cols:
    num_summary(df, col, plot=True)
############################  HEDEF DEĞİŞKENIN ANALİZİ  ############################
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()


# bool değişkenleri integer tipine dönüştürdük.
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

# grab_col_names fonksiyonunu kullanarak veri seti içerisindeki numeric, kategoril ve kardinal değişkenleri otomatik olarak yakaladık
def grab_col_names(dataframe, cat_th=10, car_th=20):
   """
   Veri setindeki kategorik, numerik ve kategorik fakar kardinal değişkenlerin isimlerini verir.
   Parameters
   ----------
   dataframe : dataframe
        değişken isimleri alınmak istenen dataframe'dir.
   cat_th : int,float
        numeric fakat kategorik olan değişkenler sınıf eşik değeri
   car_th : int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

   Returns
   -------
    cat_cols : list
        Kategorik değişken listesi
    num_cols : list
        Numeric Değişken listesi
    cat_but_car : List
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols 'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir:
   """

   #cat_cols , cat_but_car
   cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
   num_but_cat = [col for col in df.columns if df[col].nunique() <10 and df[col].dtypes in ["int", "float"]]
   cat_but_car = [col for col in df.columns if
                  df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
   cat_cols = cat_cols + num_but_cat
   cat_cols = [col for col in cat_cols if col not in cat_but_car]


   num_cols = [col for col in df.columns if str(df[col].dtypes) in ["int", "float"]]
   num_cols = [col for col in num_cols if col not in cat_cols]

   print(f"Observations: {dataframe.shape[0]}")
   print(f"Variables: {dataframe.shape[1]}")
   print(f'cat_cols: {len(cat_cols)}')
   print(f'num_cols: {len(num_cols)}')
   print(f"cat_but_car: {len(cat_but_car)}")
   print(f"num_but_car: {len(num_but_cat)}")


   return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#sırada ne var ? Sırada hedef değişkeni (yani survived değişkenini) kategorik ve sayısal değişkenler üzerinden analiz etmektir.
# survived değişkenini analiz etmemiz gerekiyor

df["survived"].value_counts()
cat_summary(df, "survived")

# insanların hayatta kalmasını etkileyen şey nedir? bunu anlamamız lazım
# Bağımlı değişkene göre diğer değişkenleri göz önünde bulundurarak analizler yapmamız lazım.

########## Hedef Değişkenin Kategorik Değişkenler ile Analizi ##########
##########                                                    ##########

df.groupby("sex")["survived"].mean()
# bu işlemi fonksiyon ile yapalım

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

#dediğimizde target hedef değişkenini bir kategorik değişken ile otomatik olarak nasıl inceleriz tanımlamıs olduk.
target_summary_with_cat(df, "survived", "sex")
target_summary_with_cat(df, "survived", "pclass")

# Elimizde kategorik değişken listesi vardı otomatik olarak değişkenleri geziyordu hepsini bi gezelim target 'ı seri bi şekilde inceleyelim

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)


# Böylece bütün kategorik değişkenler ile hedef değişkenimiz analize sokulmuş oldu.

########## Hedef Değişkenin Sayısal Değişkenler ile Analizi ##########
##########                                                    ##########

df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(df.groupby("survived").agg({"age":"mean"}), end="\n\n\n")

target_summary_with_num(df, "survived", "age")

# tüm num_collar'da fonksiyonu uygularsak;
for col in num_cols:
    print(target_summary_with_num(df, "survived", col))


# KORELASYON ANALİZİ

