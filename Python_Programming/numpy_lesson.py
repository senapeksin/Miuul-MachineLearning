# Neden Numpy ?
# Hız : Verimli veri saklamadır. Sabit tipte veri saklar bundan dolayı hızlıdır.
# Fonknsiyonel düzeyde bize kolaylıklar sağlar.
# Verimli veri saklama özelliği vardır.
# Daha az çaba ile daha fazla iş yapar.
# Bu yönüyle python listelerden farklılaşır.


############################## NUMPY ARRAY OLUŞTURMA ##############################################
# Numpy arrrayleri pythonda kullanılan veri yapıları gibi bir veri yapısıdır.
# Numpy 'ın da kendisine ait bir veri yapısı vardır. Buna ndarray adı verilir.
# Numpy işlemlerini yapabilmek için ndarray ' e ihtiyacımız vardır.
# Sıfırdan numpy array'i  oluşturabilmek için ;
import numpy as np
# liste üzerinde array oluşturduk.
np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))  # numpy.ndarray

# Belirli düzen ve sıralamada sıfırdan array oluşturabiliriz

np.zeros(10, dtype=int) # Bu metod girdiğiniz sayı adetince 0 oluşturuyor. Tip bilgisi istersek girebiliriz
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])   Tipleri integer olan sıfırlardan oluşan 10 elemanlı  arrray oluşturdu.

np.random.randint(0, 10, size=10)  #0 ile 10 arasında  10 tane rastgele int üret
#  array([6, 1, 8, 6, 9, 6, 2, 3, 0, 0])

np.random.normal(10, 4, (3, 4))   # ortalaması 10 , standart sapması 4 olan 3'e 4'lük bir array oluştur. ( 3 satırlı 4 sütunlu)
# array([[ 6.92646916, 13.45969212,  7.27030178, 14.77291221],
#        [ 9.98496093,  6.72400139, 10.33195471,  8.59555834],
#        [ 6.92433644,  7.4151785 ,  1.33232663,  9.28101824]])

############################## NUMPY ARRAY ÖZELLİKLERİ ##############################################

# ndim : boyut sayısı
# shape : boyut billgisi (satır ve sütun sayı bilgisini almak istersek)
# size : toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10 , size = 5 ) #  5 tane , 0 ile 10 arasında olan bir array oluştur.
# array([3, 6, 0, 8, 8])
a.ndim
a.shape
a.size
a.dtype

############################## RESHAPING (YENİDEN ŞEKİLLENDİRME)) ##############################################
# Elimizdeki bir arrayın boyutunu değiştirmek istediğimizde kullanırız.
np.random.randint(1, 10, size= 9)
#  array([5, 3, 6, 8, 4, 6, 3, 7, 2]) tek boyutlu array

# iki boyuta cevirelim
np.random.randint(1, 10, size= 9).reshape(3, 3)
# array([[5, 9, 8],
#        [5, 2, 3],
#        [9, 6, 7]])

# NOT: array eleman sayısı ile dönüşülecek satır sutun sayısına dikkat edilmeli.

############################## INDEX İŞLEMLERİ ##############################################
# İndex seçimi işlemleri: Bazı veri yapıları ile çalışırken, bu veri yapıları içerisindeki verilere ulaşmak isteyebilriz
# Bu verilere erişmek için kullandıgımız yöntemlerdir.

a = np.random.randint(10, size=10)
# array([4, 6, 3, 1, 4, 4, 7, 9, 6, 1])
a[0] # 0.indexteki elemana gidebiliriz

#slicing: dilimleme ile yapar
a[0:5]  # sol taraftaki dahil, sağ taraftaki hariç şekilde işlem yapar
# array([4, 6, 3, 1, 4])

a[0] = 999

m = np.random.randint(10, size=(3, 5))  # 0'dan 10!a kadar olan sayılar olsun.Boyut bilgisi 3 satırlı 5 sütunlu array olsun
# array([[3, 0, 2, 2, 5],
#        [0, 7, 9, 6, 4],
#        [3, 6, 9, 3, 0]])

# İlk elemana erişmek istersek; virgülden öncesi satırları, virgülden sonrası sütunları temsil eder.
m[0, 0]  #3
m[1, 1]  #7
m[2, 3] = 90
#array([[ 3,  0,  2,  2,  5],
#       [ 0,  7,  9,  6,  4],
#       [ 3,  6,  9, 90,  0]])

# Bütün satırları seç ,  0. sütunu seç

m[: , 0]           # array([3, 0, 3])

# Birinci satır, bütün sutunları seçelim

m[1, :]            # array([0, 7, 9, 6, 4])

# Hem satır hem sütun bilgisi verelim

m[0:2, 0:3]
#array([[3, 0, 2],
#     [0, 7, 9]])

############################## FANCY INDEX ##############################################
# Elimizde birden fazla binlerce index bilgisi olabilir. Bu durumda sanki tek seçim yapıyormus gibi ilgli array'e
# liste gönderdiğimizde o listeye karşılık gelen index bilgilerini bize verir.
#########################################################################################
import numpy as np
v = np.arange(0, 30, 3)
v[3]
check = [1, 2, 3]
v[check]

############################## NUMPY İLE KOŞULLU İŞLEMLER  ##############################
# Elimizde numpy arrray oldugunu düşünelim. Amacımız 3'den kücük olan değerlere erişmek olsun.
#########################################################################################

# Klasik döngü ile

v = np.array([1, 2, 3, 4, 5])
ab = []
for i in v:
    if i < 3 :
        ab.append(i)

# numpy ile
import numpy as np
v = np.array([1, 2, 3, 4, 5])
v < 3
v[v < 3]  # bir koşulun sağlanıp sağlanmadıgını otomotik olarak numpy yapıyor ve bize true false değerlerini döndürüyor.
          # Bu değerleri array'in içine koydugumuzda koşulu sağlayanlar gelmiş oldu.

# Özetle : Eğer bir arrayın içerisinden koşullu elaman seçme işlemi gerçekleştirmek istiyorsanız bunu yapmanız gerekmektedir.

############################## MATEMATİKSEL İŞLEMLER ##############################################
import numpy as np
np.array([1, 2, 3, 4, 5])
v / 5
v * 5 / 10
v ** 2
v - 1

#metodlar aracılıgyla gercekleştirelim
import numpy as np
np.subtract(v, 1)  #cıkarma işlemi için
np.add(v, 1)       #toplama işlemi için
np.mean(v)         #ortalama  için
np.sum(v)          #toplam alma işlemi için
np.min(v)
np.max(v)
np.var(v)          #varyans işlemi için


############# NUMPY İLE İKİ BİLİNMEYENLİ DENKLEM ÇÖZÜMÜ ###########

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]]) #katsayıları gönderdik
b = np.array([12,10])  # sonucları gonderdik

np.linalg.solve(a, b)