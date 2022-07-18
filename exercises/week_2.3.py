###################### Miuul WEEK-2.3 ######################

###################### # Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama   ######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Görev 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("persona.csv")
print(df.head())

# Görev 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()

# Görev 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()

# Görev 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

 # Görev 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()


# Görev 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY").agg({"PRICE":"sum"})


# Görev 7: SOURCE türlerine göre satış sayıları nedir?
df.groupby("SOURCE").agg({"PRICE":"count"})

# Görev 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE":"mean"})


# Görev 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE").agg({"PRICE":"mean"})


# Görev 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})


# Görev 11: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazanç nedir?
df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"})


# Görev 12: Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a göre uygulayınız.
#Çıktıyı agg_df olarak kaydediniz.
agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).sort_values("PRICE",ascending=False)
agg_df


# Görev 13: Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir. Bu isimleri değişken isimlerine çeviriniz.
agg_df = agg_df.reset_index()
agg_df


#Görev 14: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
# • Aralıkları ikna edici şekilde oluşturunuz.
# • Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'
agg_df["AGE"]=agg_df["AGE"].astype("category")
df["AGE"].astype("category")
agg_df["AGE"].dtype
agg_df["AGE_CAT"]=pd.cut(agg_df["AGE"], [0,18,23,30,40,70], labels=["0_18","19_23","24_30","31_40","41_70"])
agg_df.head()

#Görev 15: Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
#• Yeni eklenecek değişkenin adı: customers_level_based
#• Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.

# Hint: Dikkat! List comprehension ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18. Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.
agg_df["customer_level_based"] = agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]].apply("_".join,axis=1)
agg_df["customer_level_based"]

agg_df.customer_level_based.value_counts()
new_persona = agg_df.groupby("customer_level_based").agg({"PRICE": "mean"})
new_persona
new_persona=new_persona.reset_index()
new_persona


#Görev 15: Yeni müşterileri (personaları) segmentlere ayırınız.
#• Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).
pd.qcut(new_persona["PRICE"], 4 ,labels=["D", "C", "B", "A"])

agg_df["SEGMENT"] =  pd.qcut(new_persona["PRICE"], 4 ,labels=["D", "C", "B", "A"])
agg_df["SEGMENT"]


agg_df.groupby("SEGMENT").agg({"PRICE":["mean","sum","max"]})


# Görev 16: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
#33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_user = "TUR_ANDR0ID_FEMALE_31_40"
agg_df[agg_df["customer_level_based"]== new_user]
