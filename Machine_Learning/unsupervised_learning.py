################################
# Unsupervised Learning
################################

# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

################################
# K-Means
################################

df = pd.read_csv("Machine_Learning/datasets/USArrests.csv", index_col=0)

df.head()
df.isnull().sum()
df.info()
df.describe().T

# uzaklık temelli bir yöntem kullanıcaz. Uzaklık temelli ve gradient descent temelli yöntemlerin
# kullanımındaki süreçlerde değişkenlerin standartlaştırılması önem arz ediyordu.
# dolayısıyla buradaki değişkenleri de standartlaştırmamız gerekemdktedir.


sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)  # fit_transformdan çıktıktan sonra bu numpy array' e dönüştü.
df[0:5]

kmeans = KMeans(n_clusters=4, random_state=17).fit(df)  # x imizi giricez yani df değerlerimizi giriyoruz.
kmeans.get_params()

kmeans.n_clusters  # küme sayısı
kmeans.cluster_centers_  # cluster'ların merkezi
kmeans.labels_  # küme etiketleri gelmiş olacak 0 ilk küme 3 son kümeyi ifade ediyor.
kmeans.inertia_  # sse veya ssr değerinin karşılığı (gözlem birimlerinin en yakın cluster' a olan uzaklığı inertia_ değeridir.)

# Kümeler belirlendi, labellara baktık, merkezler var. Ne yapıcaz simdi?
# Çalışmanın sonucunda buradaki kümeleri ve gözlem birimlerini isimlendirmeleriyle yanyana getiriyor olucaz.
# Dolayısıyla simdilik bu bölümü askıda bekletelim.
# Öncelikle belirlemmeiz gereken max küme sayısının ne olması gerektiği konusuna gelelim
# sonrasında final clusterları oluşturduktan sonra bu labelları gözlem birimlerinin yanına yazıyor olucaz.

################################
# Optimum Küme Sayısının Belirlenmesi
################################

# Öyle bir işlem yapmalıyım ki farklı  K parametre değerlerine göre bir seyleri inceleyip o şeylere göre karar vermeliyim
# Neyi incelemeliyim? SSE 'yi incelemeliyim.
# Dolayısıyla şöyle yapalım boş bir kmeans nesnesi ve boş bir list olustuyorum ve 1 den 30 a kadar K'lar oluşturuyorum

kmeans = KMeans()
ssd = []
K = range(1, 30)

# döngü aracılıgı ile bunun içerisinde gezerek bu işlemleri yapabiliriz.

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

ssd

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# gözlem birimi kadar cluster olursa ssd 0 olur. Çünkü bütün gözlem birimleri bir cluster olur.
# Herbir gözlem birimi merkez olur .Dolayısıyla sse 0 olur. Bundan dolayı zaten küme sayısı arttıkça hatanın düşmesini bekleriz.
# o zaman napıcaz?

# Kmeans yöntemi hiyeraşik kümeleme yöntemi gibi kümeleme yöntemleri kullanılırken,
# algoritmanın bize verdiği matematiksel referanslara göre, sse 'ye göre olan küme sayılarınına direkt  bakılarak iş yapılmaz.

# normalde optimum noktaya karar verirken dirseklenmenin yani eğimin en şiddetli oldugu noktalar seçilir.
# Ama bu grafikte bunu seçmek zordu. Bunun daha otomotik bir yolu var. Elbow yöntemi.
# elbow yöntemi seçilmesi gereken optimum  küme sayısı olarak belirler.
# Yani bu verisetini kümelere ayırmak istiyorsan ayırman gereken optimum nokta burasıdır demiştir yellowbrick kütüphanesinin ilgili fonksiyonu.


kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.read_csv("Machine_Learning/datasets/USArrests.csv", index_col=0)

df["cluster"] = clusters_kmeans

df.head()

df["cluster"] = df["cluster"] + 1
df[df["cluster"]==2]

df.groupby("cluster").agg(["count","mean","median"])

df.to_csv("clusters.csv")


################################
# Hierarchical Clustering
################################

df = pd.read_csv("Machine_Learning/datasets/USArrests.csv", index_col=0)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")  # birleştirici bir clustering yöntemi. Öklid uzaklıgına göre gözlem birimlerini kümelere ayırıyor.


# Hiyerarşik kümeleme yöntemlerinin temel noktası Dendogram adı verilen küümeleme yapısını gösteren şemadır.

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)  # leaf_font_size : en aşığıda gördüğümü isimlendirmelerin boyutu
plt.show()


plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

################################
# Kume Sayısını Belirlemek
################################

# Dendogram üzerinden bir bilgi aldık. Küme sayısını belirlemede çizgi atma işini de işin içine katalım.
# Bunun için axhline metodunu kullanıyoruz.

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering  # birleştirici clustering metodunu import ediyoruz.

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
# cluster yapısını ortaya cıkaracak olan ilgili metodu kullandık.

clusters = cluster.fit_predict(df)

df = pd.read_csv("Machine_Learning/datasets/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = df["kmeans_cluster_no"]  + 1
df["kmeans_cluster_no"] = clusters_kmeans


################################
# Principal Component Analysis
################################

df = pd.read_csv("Machine_Learning/datasets/hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]  # sayısal değişkenleri ve bağımsız değişkenleri aldık.

df[num_cols].head()

df = df[num_cols]  # dataframe'den ucurmak
df.dropna(inplace=True)
df.shape

# şu anda bağımlı değişken yok. Çeşitli değerler var elimde. değişken değerleri.
# Örnek olması acısından kendime bir veri seti oluşturdum.
# Amacım bu çok değişkenli verinin daha az değişken ile ifade edilmesi .
# burada 16 tane değişken var ya ben bu verisetini 2 veya 3  bileşene indirgemeye çalışıcam.

# Bu yöntem yine standartlaştırma ihtiyacı duydugumuz bir yöntem.

df = StandardScaler().fit_transform(df)
df

pca = PCA()
pca_fit = pca.fit_transform(df)

# temel bileşen analizini uyguladık.
# Şimdi bu bileşenler oluştugunda bileşenlerin başarısını yani yeni değişkenler aslında bunlar ön tanımlı değeri ile oluşturduk.
# Bu bileşenlerin başarısı nasıl değerlendirilecek?
# Bu bileşenlerin başarısı bileşenlerin açıkladıgı varyans oranlarına göre  belirlenmektedir.
# DİKKAT ! Temel bileşen analizinin amacı neydi ?
# Verisetindeki değişkenliğin verisetindeki bilginin daha az sayıdaki bileşen ile açıklanması idi.
# Bunu yaparken ne oluyordu? Belirli miktarda bir bilgi kaybı oluyordu. Bilgi = varyanstır.
# Dolayısıyla bu bileşenlerin başarısını değerlendirebileceğimiz metrik  açıkladıkların varyans oranıdır.

pca.explained_variance_ratio_   #bileşenlerin açıkladıkların varyans oranları bilgisi gelecek. Bu ne demek ? Açıkladıkları bilgi oranı demek
np.cumsum(pca.explained_variance_ratio_)  # kümülatif varyansları hesapladık. Yani peş peşe 2 bileşenin açıklayacak oldugu varyans nedir vs şeklinde gözlem yaptıgımızı düşünelim.


# ne yaptık ?
# pca.explained_variance_ratio_ => burada elde ettiklerimiz pca'nın oluşturdugu 16 adet yeni bileşenin hepsinin açıkladıgı varyans oranı
# sonra bunun kümülatifini alıyorum ki bu bileşenler bir araya geldiğinde toplam ne kadar açıklama oranına sahip
# dolayısıyla mesela birinci değişken (0.46037855) verisetinin içindeki bilginin neredeyse yarısını tek basına açıklıyor.
# bir bileşen daha koydugumuzd akümülatif olarak yani 46'nın üstüne bir değer eklediğimizde 72 gelmiş.
# 3. bileşen 4. bileşen  5. bileşene geldiğimizde verinin içinde bulunan bilginin (varyansın) yüzde 91'inin açıklandıgını görüyoruz.
# iyi de hani şey demiştik sanki ? Daha az sayıda bileşendi hani ? Evet öyle demiştik gelicez yavas yavas.
# PCA 'yı ön tanımlı değeri ile kullandık bütün değerlere bir baktık sadece.
# Dolayısıyla aslında buradan da neye karar vermemiz gerektiğine ilişkin bir bilgiyi edindik aslında
# Mesela şöyle bir karar: Ben zaten 3 bileşen kullandıgımda mevcut değişkenliğin %82 'sini açıklayabiliyorum
# o zaman  bu veri setini 3 bileşene indirgeyebilirim şeklinde karar veriyor olacağız.

# Şimdi bu bileşen sayısına nasıl karar vereceğimizi değerlendirelim.

################################
# Optimum Bileşen Sayısı
################################

# Yine elbow yöntemi. Bu yöntem ile neyi inceleyebiliyorduk ?
# En keskin geçisin en kayda değer değişikliğin nerede oldugunu inceliyorduk.

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()

# Bu grafik incelendiğinde 2 3 gibi bir değer tercih edilebilir.

################################
# Final PCA'in Oluşturulması
################################

pca = PCA(n_components=3)  # bileşen sayısını 3 seçerek final modeli oluşturuyorum.
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)  # kümülatif toplamını alırsak, bu 3 bileşenin toplam ne kadar bilgi açıkladıgını buluyoruz.


################################
# BONUS: Principal Component Regression (Temel Bileşen Regresyon Modeli) PCR
################################

# Bileşenleri çıkardık ama bunları şimdi nereye koyacağız. Ne anlama geliyor? soruları olabilir.
# Diyelimki Hitters veri seti doğrusal bir model ile modellenmek istiyor
# ve değişkenler arasında coklu  doğrusal bağlantı problemi var.
# Bu doğrusal regresyon modellerinde sağlanması gereken varsayımlardandır.
# Değişkenler arasında  yüksek korelasyon oldugunda bu çeşitli problemlere sebep olur. Bunu istemiyoruz

# Diyelim ki böyle bir problemimiz var.İfade ettiğimiz gibi PCR başka amaçlarla kullanılan bir araçtır.
# Şimdi böyle bir amacımız oldugunu düşünelim

# Birincisi bu amacımızı yerine getiricez.
# İkincisi yukarda cıkarmıs oldugumuz bileşenlerin neye karsılık  geldiğini anlamaya çalışıcaz.

df = pd.read_csv("Machine_Learning/datasets/hitters.csv")
df.shape
# 322 tane gözlem birimi var, pca_fit içeresinde kaç tane gözlem vardı? bundada 322 tane gözlem birimi var.

len(pca_fit)

# Anlıyorum ki, gözlem birimlerim aynı yerinde şu an. Ama burada ne vardı?  burada 3 tane bileşen vardı.(pca_fit için)
# diğerinde ne var daha fazla değişken var.Anlıyorum ki gözlem birimlerim yerinde.



num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols)  # 16 tane numeric değişken var. Biz ne yapmıstık bu 16 tane değişkeni 3 tane bileşene indirgemiştik

others = [col for col in df.columns if col not in num_cols] # num_cols' dısında kalan değişkenleri getir diyorum.
others #  ['League', 'Division', 'Salary', 'NewLeague']   , Bunlar da diğer değişkenler

pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]).head() # 3 bileşenli olan pca_fit değişkenlerini dataframe e çeviriyorum ve isimlerndirmelerini yapıyorum.
# 3 tane yeni değişken bu. Diğer değişkenlere de bakalım
df[others].head()

# Buradan ne anlıyoruz? Daha öncesinde 16 tane sayısal değişkenim vardı.
# Bunları ben 3 bileşene indirgedim.E bunun yanında bir de işte bu verisetindeki bağımlı değişken var ve kategorik değişkenler var
# ne yapıcam ? problemimiz şu: pcr denilen bir makine öğrenmesi yöntemi var. Bu yöntem şöyle çalışıyor
# Önce bir temel bileşen analizi yöntemi uygulanıp değişkenlerin boyutu  indirgeniyor.
# Daha sonrasında bu bileşenlerin üzerine bir regresyon modeli kuruluyor.

# Direk bunu yapan bir python kodu yok. İstatistik literatüründe var olan bir konu kendim uyarlayacağım.
# MAdem sayısal değişkenleri seçersem ve onları bileşenlerce ifade edebilirsem e bunun üzerine bir regresyon modeli kurarım,
# bu değişken gibi zaten ya dolayısıyla regresyon modelinin sonucunu elde edebilirim.
# bileşenleri değişken olarak kullanabiliyoruz.  Sonuclara baktıgımızda tutarlı olması  durumunda doğru yoldayız diyebilirim.

# Madem bileşenleri indirgeme işi tammalndı. O zaman bu iki veri setini bir araya getirelim.

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                      df[others]], axis=1) # axis =1 yani yan yana koy.
final_df.head()
# Burada 16 tane değişken vardı su anda 3 tane var. Ama ? Aması yok boyut indirgedim.
# Artık o 16 değişkenin taşıdıgı bilginin yüzde 82 si bu değişkenlerce taşınıyor ve burada bir problemimiz oldugunu varsaydık.
# Çoklu doğrusal bağlantı problemi. Bağımsız değişkenlerin birbiri ile yüksek korelasyonlu olması problemi
# buradaki bileşenler korelasyonsuz. ( PC1       PC2       PC3) kırdık korelasyonu.

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# final df var. Model kurmak isitiyorum ama bir problemim var .Buradaki  Kategorik değişkenler.
# bu kategorik değişknelerin hepsinin sınıf değişkeni 2 oldugu için bunları dilersek direkt label encoder dan geçirebiliriz
# yada dummys kullanabiliriz.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:  # 3 tane kategorik değişken var, sınıf sayılarının 2 oldugunu bilyorum. Ondan label encoden kullanıyorum
    label_encoder(final_df, col)
final_df.head()
final_df.dropna(inplace=True) # nan değerleri siliyoruz.

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()  # daha sonra kullanmak derdim olmadıgı için fit etme işlemi yapmıyorum direkt hatasına bakıcam cross_val_score yöntemi ile
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse  #  345.6021106351967
# doğru yolda mıyız yanlış yolda mıyız ? nasıl karar veriyorduk buna
# y bağımlı değişkenin bir ortalamasını alalım.
y.mean() # 535.9258821292775
#iyi mi değil, çok mu kötü değil

# DecisionTreeRegressor da yapalım

cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse # 371.8732139505637  (daha kötü çıktı)

# hyperparametre optimizasyonu yapalım
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse  # 330 a düştü verdiğimiz arama seti ile .

# SORU: Elimde bir veri seti var ama verisetinde label yok, sınıflandırma problemi çözmek istiyorum. Ne yapabilirim?
# önce unsupervised yöntemi ile çeşitli cluster'lar çıkarırım. Daha sonra bu cıkardıgım clusterlar = sınıflar diye düşünürüm.
# Etikletlerim onları sonra bunları sınıflandırıcıya asokarım


################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme  / PCA Görselleştirme
################################

################################
# Breast Cancer
################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("Machine_Learning/datasets/breast_cancer.csv")

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)

# amacımız şu olsun; bu cok değişkenli veriyi 2 eksen üzerinde görselleştirmek isteyelim.
# Bunu cok fazla değişken oldugundan dolayı yapmak mümkün değil
# Öyle bir işlem yapmam lazım ki bunu 2 boyutta görselleştirebiliyor olayım

# Önce veri setini 2 boyuta indirgememiz lazım, sonra da görselleştirmemiz lazım
# birinci işimiz için fonksiyonumuzda, önce bağımsız değişkenleri standartlaştıracak daha sonra PCA  hesabı yapacak
# değişken değerlerini dönüştürmüş olacak yani bileşenleri çıkarmıs olacak
# bu bileşenleri bir dataframe e çevirdikten sonra bağımlı değişken ile yan yana concat ederek dısarıya verecek.

def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y)
pca_df  #2 değişkene indirgendi

df.shape # 32 tane değişen vardı, bir tanesi bağımlı değişken diğeri ıd
pca_df.shape # 3 tane kaldı , biri bağımlı değişken yani 2 tane bileşen olmus oldu

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")

#başka bir veri setinde de yapalım

################################
# Iris
################################

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1) # kategorik değişken olmaması lazım

# pca dataframe olusturma fonksiyonu ve pca görselleştirme fonksiyonunu her ihtiyacımızda kullanabiliriz.
# Sadece bu fonksiyonlara göndereceğimiz X 'in sayısal değişkenlerden oluşması lazımdır.

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")


################################
# Diabetes
################################

df = pd.read_csv("Machine_Learning/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)  #

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")




















