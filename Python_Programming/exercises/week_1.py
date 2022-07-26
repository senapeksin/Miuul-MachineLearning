###################### Miuul WEEK-1 ######################

###################### GÖREV 1: ######################
# Verilen değerlerin veri yapılarını inceleyiniz.
x = 8
type(x)
#int

y = 3.2
type(y)
#float

z = 8j + 18
type(z)
#complex

a = "Hello Word!"
type(a)
#string

b = True
type(b)
#bool

c = 23 < 22
type(c)
#bool

l = [1, 2, 3, 4]
type(l)
#list

d = {
    "Name": "Jake",
    "Age": 27,
    "Address": "Downtown"
    }
type(d)
#dictionary

t = ("Machine Learning", "Data Science")
type(t)
#tuple

s = {"Python", "MAchine Learning", "Data Science"}
type(s)
#set

###################### GÖREV 2: ######################
# Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.

text = "The goal is  to turn data into information, and information into insight."
# 'THE GOAL IS  TO TURN DATA INTO INFORMATION, AND INFORMATION INTO INSIGHT.'
new_text = text.replace(",", " ")
# 'THE GOAL IS  TO TURN DATA INTO INFORMATION  AND INFORMATION INTO INSIGHT.'
replace_text = new_text.replace(".", " ")
# 'THE GOAL IS  TO TURN DATA INTO INFORMATION  AND INFORMATION INTO INSIGHT '
result_text = replace_text.upper().split()
# ['THE',
#  'GOAL',
#  'IS',
#  '',
#  'TO',
#  'TURN',
#  'DATA',
#  'INTO',
#  'INFORMATION',
#  '',
#  'AND',
#  'INFORMATION',
#  'INTO',
#  'INSIGHT',
#  '']

###################### GÖREV 3: ######################
# Verilen listeye aşağıdaki adımları uygulayınız.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakınız.
len(lst)
# 11

# Adım 2: Sıfırıncı ve onuncu indesteki elemanları çağırınız.
lst[0]
lst[10]
# D , E

# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz
lst[0:4]
# ['D', 'A', 'T', 'A']

# Adım 4: Sekizinci indeksteki elemanı siliniz.
lst.remove(lst[8])
# ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'C', 'E']

# Adım 5: Yeni bir eleman ekleyiniz.
lst.append("S")
# ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'C', 'E', 'S']

# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")
# ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'N', 'C', 'E', 'S']

###################### GÖREV 4: ######################
# Verilen sözlük yapısına aşağıdaki adımları uygulayınız.

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

# Adım 1: Key değerlerine erişiniz.
dict.keys()
# dict_keys(['Christian', 'Daisy', 'Antonio', 'Dante'])

# Adım 2: Value'lara erişiniz.
dict.values()
#dict_values([['America', 18], ['England', 12], ['Spain', 22], ['Italy', 25]])

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak günceleyiniz.
dict['Daisy'] = ["England", 13]
# dict['Daisy'] = ["England", 13]

# Adım 4: Key değeri Ahmet, value değeri ["Turkey", 24] olan yeni bir değer ekleyiniz.
dict['Ahmet' ] = ["Turkey", 24]
# {'Christian': ['America', 18],
#  'Daisy': ['England', 13],
#  'Antonio': ['Spain', 22],
#  'Dante': ['Italy', 25],
#  'Ahmet': ['Turkey', 24]}

# Adım 6: Antonio'yu dictionary'den siliniz.
dict.pop('Antonio')
# {'Christian': ['America', 18],
#  'Daisy': ['England', 13],
#  'Dante': ['Italy', 25],
#  'Ahmet': ['Turkey', 24]}

###################### GÖREV 5: ######################
# Argüman olarak bir liste alan, listenin içerisindeki
# # tek ve çift sayıları ayrı listelere atayan ve bu listeleri
# # return eden fonksiyonu yazınız.

def my_function(list):
    even_list = []
    odd_list =[]
    [even_list.append(i) if i%2 == 0  else odd_list.append(i) for i in list]
    return  even_list,odd_list

l = [2, 13, 18, 93, 22]
deger = my_function(l)
# ([2, 18, 22], [13, 93])


###################### GÖREV 6 : ######################
# Görev 6: List Comprehension yapısı kullanarak car_crashes
# verisindeki numeric değişkenlerin isimlerini büyük harfe
# çeviriniz ve başına NUM ekleyiniz

# İpucu: Numeric olmayan değişkenlerin de isimleri büyümeli.
# İpucu: Tek bir list comprehension yapısı kullanılmalı.
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

["NUM"+i.upper() if df[i].dtype != "O" else i.upper() for i in df.columns]
# ['NUMTOTAL',
#  'NUMSPEEDING',
#  'NUMALCOHOL',
#  'NUMNOT_DISTRACTED',
#  'NUMNO_PREVIOUS',
#  'NUMINS_PREMIUM',
#  'NUMINS_LOSSES',
#  'ABBREV']

###################### GÖREV 7 : ######################
# List Comprehension yapısı kullanarak car_crashes
# verisindeki isminde "no" BARINDIRMAYAN değişkenlerin sonuna
# "FLAG" yazınız.

# İpucu: Tüm değişkenlerin isimleri büyük harf olmalı.
# İpucu: Tek bir list comprehension yapısı kullanılmalı.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

[i.upper() if "no" in i else i.upper()+"FLAG" for i in df.columns]
# ['TOTALFLAG',
#  'SPEEDINGFLAG',
#  'ALCOHOLFLAG',
#  'NOT_DISTRACTED',
#  'NO_PREVIOUS',
#  'INS_PREMIUMFLAG',
#  'INS_LOSSESFLAG',


###################### GÖREV 8 : ######################
#List Comprehension yapısı kullanarak aşağıda verilen
# değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.

# İpucu: Önce verilen listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
# İpucu: Sonra df[new_cols] ile bu değişkenleri seçerek yeni birdf oluşturunuz ve adını new_df olarak isimlendiriniz.

og_list = ["abbrev", "no_previous"]

new_cols = [i for i in df.columns if i not in og_list]
new_df = df[new_cols]
new_df.head()

