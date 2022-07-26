###################### Miuul WEEK-2.1 ######################

###################### DOCSTRING OLUSTURMAK ######################

# Görev 1 :  cat_summary() fonksiyonuna 1 özellikekleyiniz. Bu özellik argümanla biçimlendirilebilir olsun.
# Var olan özelliğide argümanlakontroledilebilirhale getirebilirsiniz

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

def cat_summary(dataframe, col_name, plot= False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################")
    if plot:
        sns.countplot(x=dataframe[col_name], data= dataframe)
        plt.show()

cat_summary(df,"survived", True)

# Görev: check_df(), cat_summary() fonksiyonlarına 4 bilgi(uygunsa) barındıran numpy tarzı docstring yazınız. (task, params, return, example)

#cat_summary() fonksiyonumuz
def cat_summary(dataframe, col_name, plot= False):
    """

    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    col_name : str
        kolon ismini string tip ile girmeliyiz.
    plot : bool
        Grafik çizdirmek istiyorsak True / istemiyorsak False değer vermeliyiz.
    Returns
    -------

    """
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################")
    if plot:
        sns.countplot(x=dataframe[col_name], data= dataframe)
        plt.show()

cat_summary(df,"survived", True)

# check_df() fonksiyonumuz
def check_df(dataframe, head=5):
    """

    Parameters
    ----------
    dataframe :dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    head : int
        Veri setinin başından başlayarak görülmesi istenen satır sayısıdır. Default değeri 5'dir
    Returns
    -------

    """
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