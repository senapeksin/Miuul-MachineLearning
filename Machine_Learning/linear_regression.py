######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("Machine_Learning/datasets/advertising.csv")
df.shape

X = df[["TV"]]
y = df[["sales"]]


##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*x
# y_hat = b + w*TV

# sabit (b - bias)
reg_model.intercept_[0]

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]


##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik tv harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T


# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


##########################
# Tahmin Başarısı
##########################

# MSE
# MSE metodu der ki ; bana gerçek ve tahmin edilen değerleri ver.Bunların farklarını alırım, karelerini alırım. Toplayıp
# ortalamasını alarak sana ortalama hatanı veririm der.

# y_pred = reg_model.predict(y, y_pred)

# Fakat elimizde tahmin edilen değerler yok. Ne yapmamız lazım ?
# reg_model i kullanarak predict metodunu çağırıyorum. Ve bu metoda bağımsız değişkenlerimi yani X 'i veriyorum ve diyorum ki
# bu X değerlerini sorsam bu değerlere göre bana bu modelin bağımlı değişkenlerini tahmin etsen.
# Dolayısıyla bağımsız değişkenleri modele sordum ve bağımlı değişkenleri tahmin etmesini isteyerek y_pred değişkenine atadım.

y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 10.51
y.mean()  #satışların ortalaması
y.std()  # satışların standart sapması

# ortalaması 14. Standart sapması 5 çıktı. Yani ortalama 19 ve 9 arasında değerler değişiyor gibi gözüküyor. Bu durumda  elde ettiğim
# 10 değeri büyük mü kücük mü diye düşünecek olursak, büyük kaldı gibi .
# Ne kadar kücük o kadar iyi.  Fakat büyüklüğü de neye göre değerlendirmem gerektiğini bilmediğim durumda bağımlı değişkenin
# ortalamasına ve standart sapmasına bakıyoruz.

# RMSE
# RMSE = MSE'nin kareköküdür
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
# MAE = mutlak hata
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE
# R-KARE : score metodu ile gerçekleştiriyoruz. Regresyon modeline bir score hesapla talebini gönderiyoruz
# R-KARE : Doğrusal regresyon modellerinde modelin başarısına ilişkin cok önemli olan bir metriktir.
# Şunu ifade eder: Verisetindeki bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir.
# Bu verisetinde bir tane bağımsız değişken var televizyon değişkeni. Televizyon değişkeninin satış değişkenindeki değişikliği açıklama yüzdesidir.
# Bu modelde bağımsız değişkenler , bağımlı değişkenin yüzde 61'ini açıklayabilmektedir şeklinde yorumlanır.
reg_model.score(X, y)
#0.61

# !!!! Değişken sayısı arrtıkca R2 şişmeye meyillidir. Burada düzeltilmiş R2 değerinin de göz önünde bulundurulması gerekir.
# bir değil birden fazla bağımsız değişken ile ele alalım


######################################################
# Multiple Linear Regression
######################################################
# bir değil birden fazla bağımsız değişkenimiz olacak.
df = pd.read_csv("Machine_Learning/datasets/advertising.csv")

X = df.drop('sales', axis=1) # bağımsız değişkenler

y = df[["sales"]]


##########################
# Model
##########################
#Veri seti yüzde 20 si test ve yüzde 80'ini train olarak iki parçaya bölünür.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_test.shape
X_train.shape
y_test.shape
y_train.shape

# şimdi train seti ile model kuracaktık, test seti ile test edeceğiz.

reg_model = LinearRegression().fit(X_train, y_train)  #train setinin bağımlı ve bağımsız değişkenlerini verdik.

# sabit (b - bias)
reg_model.intercept_
reg_model.intercept_[0]

# coefficients (w - weights)
reg_model.coef_


##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90 (sabitimiz)
# 0.0468431 , 0.17854434, 0.00258619 (w)

#Ne yapılması lazım ? doğrusal bir formda katsayılarla ilgili gözlem değerlerinin çarpılıp buradaki sabit ile toplanması lazım

#model denklemi:
# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619


# Diyelim ki modeli kurduk . Bu modeli canlı sistemle basit bir excel sistemiyle bir yerlerle entegre edeceğimizi düşünelim
# Diyelim ki bir departman ilgili TV,radyo ve gazete harcamalarını girerek bir tahmin sonucu alacak.

# girilecek değerleri bir listeye çeviriyoruz
# ardından bu değerleri dataframe ' e ceviriyoruz.ve transpozunu alıyoruz.
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

# model nesneme tahmin et diyorum. Neyi tahmin edeyim diyor? Bana bağımsız değişkenleri ver, ben gideyim bu regresyon
# modeline onları sorayım diyor.Buna göre de bağımlı değişkenin ne olabileceği bilgisini sana vereyim diyor.
reg_model.predict(yeni_veri)


# Modelimizi kullanarak  manuel tahminlerde bulunduk. Bunlar modelin görmediği değerlerdi . Dısardan gelen herhangi yeni gözlemlerdi.
# Şimdi tahmin başarısını değerlendirmemiz lazım.

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Şimdi elimizde bir train seti var test seti var . Hangisinin hatasına bakmalıyız. Birlikte mi bakmalıyız, ayrı ayrı mı bakmalıyız.
# Train test yaptık ama cross validation vardı onu da mı değerlendirseydik gibi sorular olabillir. Bunların cevaplarına bakalım.

# Öncelikle train hatamızı inceleyebiliriz. Yani bir model kurduk bunu train seti üzerine kurduk.İyi de daha öncede o x ve y'yi
# komple train seti gibi düşünrsek orada da hata hesaplamıştık.Dolayısıyla regresyon modelini train setinde kurduk ya
# train setinin bağımlı değişkenini de bir kenarda saklayabiliriz ve onun hata karaler ortalamasını karekökü (RMSE) değerine erişebiliriz.


# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73   bu bizim train hatamızdı.

# Evet modeli train seti üzerinden kurduk. Bunun hatası değerlendirmek istiyorsak ne yapacağız?
# Zate o tain seti gerçek değerlerine ve ve train seti için tahmin ettiğimiz değerlere bakmamız lazım.

# TRAIN RKARE
# RKARE NEYDİ : Doğrusal regresyon problemindeyiz. Bağımsız değişkenlerin bağımlı değişkenleri etkileme açıklama oranıdır.
reg_model.score(X_train, y_train)
# daha önce  yüzde 60'lar civarında olan değer yüzde 90'lara geldi. 3 tane yeni değişken eklediğimizde  yüzde 90 civarına geldi
# Buradan anladığımız sey yeni değişken eklendiğinde başarının arttıgı, hatanın düştüğüdür.

# Test RMSE
# İlk defa modele  test setini soruyoruz. predict metodu tahmin etmek için kullanılır .
# Reg modele diyoruz ki, simdi sana bir set göndericem bunu bi değerlendir bakalım diyor
y_pred = reg_model.predict(X_test)  # test setini gönder
# (Test setinin x'leri yani bağımsız değişkenlerini soruyoruz modele) o da test setinin bağımlı değişkeninini tahmin ediyor.
np.sqrt(mean_squared_error(y_test, y_pred))  # bağımlı deeğişkenin bizde gerçek değeri var (y_test). Bağımlı değişkenin birde tahmin edilen değerleri var y_pred ' dir.
# 1.41
# normalde test hatası train hatasından daha yüksek cıkar. Burda düşük cıktı cok güzel bir senaryo, yani iyi bir durum

# Test RKARE
reg_model.score(X_test, y_test)

# Evet holdout yöntemi ile train,test olarak ayırdık.Train setinde model kurduk.Test setinde hatamızı değerlendirdik.
# bunun yerine 10 katlı cv da yapabilirdik.

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69

#  neg_mean_squared_error metodunun cıktısı eksi değerlerdir.Bu nedenle - ile çarpıyoruz.
# Test hatamız 1.41, train hatamız 1.73 dü. Çapraz doğrulama ile elde ettiğimiz hata ise 1.69 cıktı.
# Hangi hataya güvenmemiz lazım ?
# Veri setimiz bol olsaydı, bu fark etmeyebilirdi gibi bir yorum çok yanlış
# Veri setimizin boyutu az oldugundan dolayı bu durumda 10 katlı capraz doğrulama yöntemine daha fazla güvenmek doğru olabilir.
# Diğer yandan  veri setinin boyutu az 10 parça yerşne 5 parça mı yapsak ? yorumu da geçerli olabilecektir.


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,  # tüm veri setini veriyoruz.
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71

# Böylece tek değişkenli ve çok değişkenli olacak sekilde regresyon modellerimizi kurduk.
# Bu regresyon modellerine ilişkin tahmin etme işlemi gerçekleştirdik.
# Daha sonra tahmin başarısını değrlendirme işlemleri gerçekleştirdik.
# bundan sonra bir regresyon kullanma ihtiyacımız oldugunda direkt olması gereken şey;
# - veri setini okuma
# - ilgili ön işleme, özellik mühendisliği işlemlerini yapma
# - model kurma basamağına gelince de modeli kurmadan önce verisetini 80' e  20 ayırmak
#  yada komple bütün veriye hata bakması gibi aşağıdaki gibi cross validation yapmak


######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

# Bu bölümde basit doğrusal regresyonu sıfırdan kendi metotlarımız ile gradient descent yöntemi kullanarak uygulamaya calışıcaz

# Ana amacımız, bir hata fonksiyonumuz vardı. cost fonksiyonumuz vardı (MSE ifadesi) bunu minimuma getirmeye çalısıyorduk.
# Nasıl yapıyorduk bunu w değerlerini yada bias sabit değerlerini değiştirerek minimum oldugu noktaya gitmeye çalışıyorduk
# Nasıl bi işlem gerçekleşiyordu? Parametrelere göre kısmi türev alındıktan sonra bu kısmi türevler neticesinde ortaya cıkan
# değerler belirli bir öğrenme katsayısı ile çarptıktan sonra buradan gelicek olan eksi yada artı ifadesine göre parametrenin
# eski değerine ekleniyor veya cıkarılıyordu. Böylece bu fonksiyonda minumum noktaya doğru gitmeye çalısıyorduk.

# Şimdi bu işlemleri kod seviyesinde görelim.
# 1. Fonksiyonumuz cost fonksiyonu yani MSE fonksiyonu:
# Cost fonksiyonumuzun görevi sudur : MSE değerini hesaplamak. Bunun için bütün gözlem birimlerini gezmemiz ve bir sabit ve
# w çiftine göre tahmin edilen değerleri hesaplamamız lazım.

# Cost function MSE
def cost_function(Y, b, w, X): # y bağımlı , b sabit, w ağırlık ve bağımsız değişken
    m = len(Y)  # gözlem sayısını tutuyoruz
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m  # toplam hatayı m ' e bölünce ortalama hatayı bulmus olcaz.
    return mse

# 2. fonksiyonumuz update fonksiyonu. W leri update etme görevi.

# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("Machine_Learning/datasets/advertising.csv")

# değişkenlerimizi oluşturalım
X = df["radio"]
Y = df["sales"]

# hyperparameters : Veri setinde bulunamayan ve kullanıcı tarafından ayarlanması gereken parametrelerdir.
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)

# Gradient descent i sıfırdan kendimiz kodladık.
# After 100000 iterations b = 9.311638095155203, w = 0.2024957833925339, mse = 18.09239774512544







