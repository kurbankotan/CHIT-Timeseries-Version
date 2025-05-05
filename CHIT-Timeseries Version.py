!pip install yfinance
!pip install tensorflow

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler

#Makine öğrenme algoritmaları
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import math
from math import sqrt
from numpy import format_float_positional

# Parametreler
tahminEdilecekGunSayisi = 30
time_step = 4


plt.style.use("fivethirtyeight")

# Veri çekme
start_date = "2021-01-01"
end_date = "2021-12-31" #"2021-11-30" #
df = yf.download(tickers="XU030.IS", start=start_date, end=end_date)
df.head()


# Sadece Kapanış değerlerinin olduğu bir dataframe oluştur
df_Close = df[['Close']]
df_Close.head()

# Grafiği çizme
plt.figure(figsize=(15, 8))
plt.plot(df_Close, label="Kapanış")
plt.title("BIST30 Kapanış Fiyatları")
plt.xlabel("Tarih")
plt.ylabel("Kapanış Fiyatı")
plt.show()

# Grafiği çizme (İngilizce)
plt.figure(figsize=(15, 8))
plt.plot(df_Close, label="Close")
plt.title("BIST30 Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()

"""## Önnişleme"""

# Tüm tarih aralığını içeren bir tarih indeksi oluşturun (end_date'i dahil edin)
full_date_range = pd.date_range(start=start_date, end=end_date)

# Oluşturulan tarih aralığı ile yeni DataFrame oluşturma
full_df = pd.DataFrame(index=full_date_range)

# ve orijinal veri setinden 'Close' değerlerini bu yeni DataFrame'e ekleyin
full_df['Close'] = df_Close['Close'].reindex(full_date_range)

# Sonuçları yazdır
print(full_df)

# Grafiği çizme
plt.figure(figsize=(15, 8))
plt.plot(full_df, label="Kapanış")
plt.title("BIST30 Kapanış Fiyatları")
plt.xlabel("Tarih")
plt.ylabel("Kapanış Fiyatı")
plt.show()

from sklearn.preprocessing import MinMaxScaler

# Min-Max normalizasyonu için scaler oluştur
scaler = MinMaxScaler()

# 'Close' sütununu numpy dizisine dönüştür ve reshape et
close_values = full_df['Close'].values.reshape(-1, 1)

# NaN değerleri içermeyen değerleri normalleştir
# Bu adımda, NaN değerleri olan satırlar işleme dahil edilmeyecek
normalized_values = scaler.fit_transform(close_values)

# Normalleştirilmiş değerleri 'Close_Normalized' olarak full_df'e ekle
full_df['Close_Normalized'] = normalized_values

full_df = pd.DataFrame(full_df)

# Sonucu kontrol et
full_df

# Normalize Edilmiş Grafiği çizme
plt.figure(figsize=(15, 8))
plt.plot(full_df['Close_Normalized'], label="Kapanış")
plt.title("BIST30 Kapanış Fiyatları")
plt.xlabel("Tarih")
plt.ylabel("Kapanış Fiyatı")
plt.show()

def create_test_train_data(df, column_name, num_non_nan):
    """
    Verilen DataFrame'den, belirtilen sütunda NaN olmayan belirli sayıda son değeri içeren
    test veri seti oluşturur ve geriye kalan verileri eğitim seti olarak döndürür.

    Args:
    df : pandas.DataFrame
        İşlem yapılacak DataFrame.
    column_name : str
        İşlemlerin uygulanacağı sütun adı.
    num_non_nan : int
        Seçilecek NaN olmayan değer sayısı.

    Returns:
    pandas.DataFrame, pandas.DataFrame
        Test veri seti, Eğitim veri seti.
    """
    # 'column_name' sütunundaki NaN olmayan değerlerin indekslerini al
    non_nan_indices = df[column_name].dropna().index

    # Son 'num_non_nan' kadar NaN olmayan değerin indeksi
    last_valid_indices = non_nan_indices[-num_non_nan:]

    # Seçilen son geçerli indeksin başlangıcını bul
    start_index = last_valid_indices[0]

    # start_index'in DataFrame içindeki konumunu bul
    start_pos = df.index.get_loc(start_index)

    # Test verisi olarak seçilecek başlangıç noktası
    test_start_index = df.index[start_pos]

    # Başlangıç indeksinden itibaren tüm verileri test verisi olarak seç
    test_data = df.loc[test_start_index:, column_name]

    # Test verisinin başlangıç indeksi öncesinde kalan verileri eğitim verisi olarak seç
    train_data = df.loc[:df.index[start_pos - 1], column_name] if start_pos > 0 else pd.DataFrame()

    return pd.DataFrame(test_data), pd.DataFrame(train_data)

# Fonksiyonu test et
test_data, train_data = create_test_train_data(full_df, 'Close_Normalized', tahminEdilecekGunSayisi)

test_data

train_data

# Grafiği çizme
plt.figure(figsize=(15, 8))

# Orijinal Kapanış Fiyatlarını Çizme (NaN değerlerle)
plt.plot(train_data['Close_Normalized'], label="Eğitim için Ayrılan Gerçek Kapanış (NaN ile)", color='blue', linestyle='-', linewidth=2)

# İmpütasyonla Doldurulmuş Değerleri Çizme
plt.plot(test_data['Close_Normalized'], label="Test için Ayrılan Gerçek Kapanış (NaN ile)", color='red', linestyle='--', linewidth=1)


plt.title("BIST30 Kapanış Fiyatları")
plt.xlabel("Tarih")
plt.ylabel("Kapanış Fiyatı")
plt.show()

# array'i dataset'e çevir
def create_dataset_with_dates(dataset, column_name, time_step=1):
    dataX, dataY, datesX, datesY = [], [], [], []
    for i in range(len(dataset) - time_step):
        a = dataset[column_name].iloc[i:(i + time_step)].values
        dataX.append(a)
        dataY.append(dataset[column_name].iloc[i + time_step])
        datesX.append(dataset.index[i:(i + time_step)])  # X için tarihleri sakla
        datesY.append(dataset.index[i + time_step])  # Y için tarihleri sakla
    return np.array(dataX), np.array(dataY), np.array(datesX), np.array(datesY)

# Fonksiyonu kullanarak veri seti oluştur
X_train, y_train, datesX_train, datesY_train = create_dataset_with_dates(train_data, 'Close_Normalized', time_step)


# Elde edilen verileri göster

X_train= pd.DataFrame(X_train)
y_train= pd.DataFrame(y_train)
datesX_train= pd.DataFrame(datesX_train)
datesY_train= pd.DataFrame(datesY_train)

# X ve y DataFrame'lerini birleştir
combined_train_dates = pd.concat([datesX_train, datesY_train], axis=1)

# Kolon isimlerini güncelle
combined_train_dates.columns = [f'feature_{i}' for i in range(datesX_train.shape[1])] + ['target']

# Birleştirilmiş DataFrame'i göster
combined_train_dates

# X ve y DataFrame'lerini birleştir
combined_train = pd.concat([X_train, y_train], axis=1)

# Kolon isimlerini güncelle
combined_train.columns = [f'feature_{i}' for i in range(X_train.shape[1])] + ['target']

# Birleştirilmiş DataFrame'i göster
combined_train

# NaN olmayan satırları bul train_data için
non_nan_rows = ~combined_train.isnull().any(axis=1)


# Hiç NaN içermeyen veri setini ve tarihlerini seç
datasetComplete = combined_train[non_nan_rows]
datasetComplete_dates = combined_train_dates[non_nan_rows]



# En az bir NaN içeren veri setini ve tarihlerini seç
datasetMissing = combined_train[combined_train.isnull().any(axis=1)]
datasetMissing_dates = combined_train_dates[combined_train.isnull().any(axis=1)]

datasetComplete.style.highlight_null('red') # NaN değerleri varsa görelim

datasetMissing.style.highlight_null('red') # NaN değerleri varsa görelim

len(datasetComplete)

len(datasetMissing)

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Kullanıcıdan sütun bazlı ve satır bazlı imputasyon yöntemlerinin seçimini alma
print("Satır bazlı eksik veri doldurma için bir regresyon modeli seçin:")
print("1: K-Nearest Neighbors")
print("2: Linear Regression")
print("3: Support Vector Machine")
print("4: Decision Tree")
print("5: Random Forest")
print("6: Gradient Boosting")
print("7: Deep Learning (Keras)")

row_model_choice = input("Seçiminiz (1/2/3/4/5/6/7): ")

print("\nSütun bazlı eksik veri doldurma için bir yöntem seçin:")
print("1: Ortalama")
print("2: Medyan")
print("3: Mod")
print("4: Ardışık Değerlerle Doldurma (Forward/Backward Fill)")
print("5: KNN Imputer")
print("6: Interpolate")
column_method_choice = input("Seçiminiz (1/2/3/4/5/6): ")



# Kullanıcının seçimlerini onayla
row_models = {
    "1": "K-Nearest Neighbors",
    "2": "Linear Regression",
    "3": "Support Vector Machine",
    "4": "Decision Tree",
    "5": "Random Forest",
    "6": "Gradient Boosting",
    "7": "Deep Learning (Keras)",

}

column_methods = {
    "1": "Ortalama",
    "2": "Medyan",
    "3": "Mod",
    "4": "Ardışık Değerlerle Doldurma (Forward-Backward Fill)",
    "5": "KNN Imputer",
    "6": "Interpolate"
}


# Kullanıcının seçtiği model ve yöntemleri yazdır
selected_row_model = row_models.get(row_model_choice, "Bilinmeyen model")
selected_column_method = column_methods.get(column_method_choice, "Bilinmeyen yöntem")








def get_regression_model(choice, input_shape):
    models = {
        "1": KNeighborsRegressor(),
        "2": LinearRegression(),
        "3": SVR(),
        "4": DecisionTreeRegressor(),
        "5": RandomForestRegressor(),
        "6": HistGradientBoostingRegressor(),  # Daha önceden eklemiştiniz
        "7": Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # MLP için son aktivasyon değiştirdim
        ])
    }
    model = models.get(choice)

    if choice == "7":
        model.compile(optimizer='adam', loss='mean_squared_error')
    return model

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.impute import KNNImputer




# Sütun bazlı eksik veri doldurma fonksiyonu
def apply_column_imputation(dframe, method):
    datasetComplete_copy = datasetComplete.copy()
    dframe_copy = dframe.copy()

    # Ortalama kullanarak doldurma
    if method == "1":
        mean_value = dframe.mean(axis=1)
        dframe.fillna(mean_value.item(), inplace=True)  # mean_value.item() ile scalar bir değer alınır

    # Medyan kullanarak doldurma
    elif method == "2":
        median_value = dframe.median(axis=1)
        dframe.fillna(median_value.item(), inplace=True)

    # Mod (en sık değer) kullanarak doldurma
    elif method == "3":
      for column in dframe.columns:
        mode_values = dframe[column].mode()
        if not mode_values.empty:
            # Mod değerlerinden ilkini kullan
            dframe[column].fillna(mode_values.iloc[0], inplace=True)
        else:
            # Mod yoksa, alternatif bir değer ile doldur
            # Sütunun medyanıyla doldur
            median_value = dframe[column].median()
            dframe[column].fillna(median_value, inplace=True)

    # Ardışık değerlerle doldurma (ileri veya geri)
    elif method == "4":
        dframe.fillna(method='ffill', axis=1, inplace=True)
        dframe.fillna(method='bfill', axis=1, inplace=True)

    # KNN Imputer kullanarak doldurma
    elif method == "5":
        # Veri setlerini birleştir
        combined_data = pd.concat([datasetComplete, dframe], ignore_index=True)
        imputer = KNNImputer(n_neighbors=3)
        imputed_data = imputer.fit_transform(combined_data)

        # İmpute edilmiş verileri DataFrame'e dönüştür
        imputed_df = pd.DataFrame(imputed_data, columns=combined_data.columns)

        # Eksik değeri içeren orijinal dframe'i güncelle
        updated_dframe = imputed_df.iloc[-1:]  # dframe son satırda varsayılarak
        return updated_dframe

    # Interpolate (İnterpolasyon)
    elif method == "6":
        dframe.interpolate(axis=1, inplace=True)

    # Eksik veri yok
    elif method == "7":
        pass

    return dframe




# Satır bazlı model seçim fonksiyonu
def get_regression_model(choice, input_shape):
    models = {
        '1': KNeighborsRegressor(),
        '2': LinearRegression(),
        '3': SVR(),
        '4': DecisionTreeRegressor(),
        '5': RandomForestRegressor(),
        '6': HistGradientBoostingRegressor(),
        '7': Sequential([Input(shape=(input_shape,)), Dense(128, activation='relu'), Dense(64, activation='relu'), Dense(1)]),
    }
    model = models.get(choice)
    if choice == '7':
        model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    return model






def find_max_nan_column_sum(df, satir_indeksi):
  sutun_isimleri = []
  for sutun_adi in df.columns:
    if np.isnan(df.loc[satir_indeksi, sutun_adi]):
      sutun_isimleri.append(sutun_adi)

  if not sutun_isimleri:
    return None

  column_nan_sums = df[sutun_isimleri].isnull().sum()

  column_nan_sums.sort_values(ascending=False, inplace=True)
  en_fazla_nan_sutun = column_nan_sums.index[0]
  sutun_indeksi = df.columns.get_loc(en_fazla_nan_sutun)
  return sutun_indeksi

def update_with_predictions(datasetMissing, datasetMissing_dates, predicted_value, target_date):

    # datasetMissing_dates içinde target_date ile eşleşen tüm hücrelerin konumlarını bul
    matched_indices = np.where(datasetMissing_dates == target_date)

    # Eşleşen hücrelerin konumları üzerinden döngü
    for row_pos, col_pos in zip(matched_indices[0], matched_indices[1]):
        # Satır sırası yerine satır etiketini kullan
        row_label = datasetMissing_dates.index[row_pos]
        # Sütun adını al
        col_name = datasetMissing_dates.columns[col_pos]

        # Eşleşen hücredeki NaN değeri güncelle
        datasetMissing.at[row_label, col_name] = predicted_value

    return datasetMissing


def remove_filled_rows(datasetMissing, datasetMissing_dates, datasetComplete, datasetComplete_dates):
    # Eksik verisi olmayan satırları datasetComplete'e aktar
    completed_rows_indexes = datasetMissing[~datasetMissing.isnull().any(axis=1)].index
    for index in completed_rows_indexes:
        # Satırı datasetComplete'e ve ilgili tarih bilgisini datasetComplete_dates'e ekle
        datasetComplete = pd.concat([datasetComplete, datasetMissing.loc[[index]]])
        datasetComplete_dates = pd.concat([datasetComplete_dates, datasetMissing_dates.loc[[index]]])

        # İşlenen satırı datasetMissing ve datasetMissing_dates'ten çıkar
        datasetMissing.drop(index, inplace=True)
        datasetMissing_dates.drop(index, inplace=True)

    return datasetComplete, datasetComplete_dates, datasetMissing, datasetMissing_dates





# Eksik verileri doldurma fonksiyonu
def fill_missing_values(datasetComplete, datasetComplete_dates, datasetMissing, datasetMissing_dates, column_method_choice, row_model_choice):
    while not datasetMissing.empty:
        # En az NaN değere sahip satırın indisini bul
        min_missing_row_index = datasetMissing.isnull().sum(axis=1).idxmin()

        # En az NaN değere sahip satırı seç
        row_to_fill = datasetMissing.loc[[min_missing_row_index]].copy()

        #row_to_fill satır dataframe'inin indisini tutalım
        original_index=row_to_fill.index[0]

        # Geçici olarak sütun bazlı imputasyon uygula
        row_to_fill = pd.DataFrame(apply_column_imputation(row_to_fill, column_method_choice))

        #İşlem yapılan satırın tüm hücrelerinin bulunduğu satırı datasetMissing_dates'den getir
        row_to_fill_date = datasetMissing_dates.loc[[min_missing_row_index]].copy()

        #gerçek indisini geri atayalım
        row_to_fill.index = [original_index]

        # Kalıcı değerleri tahmin et ve atama yap
        # min_missing_row_index'e karşılık gelen satırı seç
        selected_row = datasetMissing.loc[[min_missing_row_index]]

        # Bu satırdaki NaN değerlerine sahip sütunları bulun
        nan_columns = selected_row[selected_row.isnull()].index.tolist()
        for col in nan_columns:
            #row_to_fill_imputed üzerinde geçici değerli olanları bul ya da aynı row_to_fill_imputed ile aynı indisli satırı datasetMissing üzerinde bul
            #datasetMissing aynı indisli satırda NaN değerleri vardır. Aynı sütun indisli hücrelerde row_to_fill_imputed üzerinde geçici değerler olur
            #Bu NaN değerlerin datasetMissing içinde aynı indisli satıdaki NaN değerlerin olduğu hücrelerin bulunduğu hücrelerdeki NaN sayılarının maksimumunu bul
            max_NaN_counts_column_index = find_max_nan_column_sum(datasetMissing, min_missing_row_index)


            #İşlem yapılan HÜCRE'nin satırın değil dikkat et, hücrenin tarihini not et
            target_date = row_to_fill_date.iloc[0, max_NaN_counts_column_index]

            # Hedef sütunu ayarla ve modeli eğit
            X_train = datasetComplete.copy().drop(columns=[datasetComplete.copy().columns[max_NaN_counts_column_index]])
            y_train = datasetComplete.copy()[datasetComplete.copy().columns[max_NaN_counts_column_index]]


            #İşelenen satırı tahmin etmek için hazırla
            y_test = row_to_fill[[row_to_fill.columns[max_NaN_counts_column_index]]]
            X_test = row_to_fill.drop(columns=[row_to_fill.columns[max_NaN_counts_column_index]])


            # Tahmin yap ve eksik değeri güncelle
            # Model seçimi ve eğitimi
            model = get_regression_model(row_model_choice, X_train.shape[1])
            if hasattr(model, 'fit'):   # modelin fit özelliği var mı bir bak
              model.fit(X_train, y_train)


            # Tahmin yap ve eksik değeri güncelle
            predicted_value = model.predict(X_test)
            if row_model_choice == '7': # NN ise
              predicted_value = predicted_value[0][0]


            #Kalıcı değeri ata
            datasetMissing.loc[min_missing_row_index, datasetMissing.columns[max_NaN_counts_column_index]]=predicted_value

            #Güncellenen hücrenin tarihinden diğer aynı tarihli hücreleri bul ve aynı değeri ata
            datasetMissing = update_with_predictions(datasetMissing, datasetMissing_dates, predicted_value, target_date)


        #row_to_fill'deki tüm geçici değerler kalıcı değerle dolduruldu.
        #Bundan dolayı row_to_fill'i datasetComplete'e ve datasetComplete_dates'e  ekle (Tamamlanan satırları da datasetMissing ve datasetMissing_dates'den  kaldır)
        datasetComplete, datasetComplete_dates, datasetMissing, datasetMissing_dates =remove_filled_rows(datasetMissing, datasetMissing_dates, datasetComplete, datasetComplete_dates)





    return datasetComplete, datasetComplete_dates

# Özellik isimlerini dizeye dönüştür (TRAIN)
datasetComplete.columns = datasetComplete.columns.astype(str)
datasetMissing.columns = datasetMissing.columns.astype(str)

# Fonksiyonun çağrılması
datasetComplete_filled, datasetComplete_dates_filled = fill_missing_values(datasetComplete, datasetComplete_dates, datasetMissing, datasetMissing_dates, column_method_choice, row_model_choice)



# İşlenmiş veriyi kontrol et
print("Doldurulmuş datasetComplete Önizleme:")
print(datasetComplete_filled.head())

len(datasetMissing)

len(datasetComplete_filled)

def apply_sorting_from_to(source_df, target_df, sort_column):
    """
    source_df DataFrame'inde sort_column sütununa göre sıralama yapar ve bu sıralamayı target_df üzerinde uygular.

    Parametreler:
    - source_df (pd.DataFrame): Sıralama için kaynak DataFrame.
    - target_df (pd.DataFrame): Sıralamanın uygulanacağı hedef DataFrame.
    - sort_column (str): Sıralama için kullanılacak sütun adı.

    Dönüş:
    - pd.DataFrame: Sıralamanın uygulandığı hedef DataFrame.
    """
    # source_df'deki sort_column'a göre sıralama indekslerini elde et
    sort_index = np.argsort(source_df[sort_column].values)

    sorted_source_df = source_df.sort_values(by=sort_column)

    # Bu indeksleri kullanarak target_df'i sırala ve sonucu döndür
    return target_df.iloc[sort_index].reset_index(drop=True), sorted_source_df.reset_index(drop=True)

# 'feature_0' sütununa göre 'datasetComplete_dates_filled' üzerinde sıralama yap ve bu sıralamayı 'datasetComplete_filled' üzerinde uygula (TRAIN)
datasetComplete_filled, datasetComplete_dates_filled = apply_sorting_from_to(datasetComplete_dates_filled, datasetComplete_filled, 'feature_0')

# İşlem sonucunu kontrol et
datasetComplete_filled

# Tarih ve değerleri birleştirmek için boş listeler oluştur (TRAIN)
combined_dates = []
combined_values = []

# Tarih ve değer sütunlarının sayısını al (Her "feature" ve "target" için bir tane olacak şekilde)
num_columns = len(datasetComplete_dates_filled.columns)

# Tarih ve değerleri birleştir
for i in range(num_columns):
    # Tarihleri al ve listeye ekle
    combined_dates.extend(datasetComplete_dates_filled.iloc[:, i])

    # Karşılık gelen değerleri al ve listeye ekle
    combined_values.extend(datasetComplete_filled.iloc[:, i])

# combined_dates ve combined_values listelerini kullanarak yeni bir DataFrame oluştur
final_df = pd.DataFrame({'Date': combined_dates, 'Close_Normalized': combined_values})

# DataFrame'i tarih sırasına göre sırala
final_df.sort_values(by='Date', inplace=True)

# Sonucu kontrol et
print(final_df)

# Tarih sütununa göre gruplayıp, değerlerin en çok tekrar edenini al (TRAIN)
final_df_grouped = final_df.groupby('Date').agg(lambda x: x.mode()[0]).reset_index()

# Sonucu kontrol et
final_df_grouped

# 'Close_Normalized' sütunundaki normalleştirilmiş değerleri orijinal ölçeğe dönüştür (TRAIN)
original_values = scaler.inverse_transform(final_df_grouped[['Close_Normalized']])

# Orijinal ölçekteki değerleri yeni bir sütuna ekleyelim
final_df_grouped['Close'] = original_values

final_df_grouped

test_data

# 'Close_Normalized' sütunundaki normalleştirilmiş değerleri orijinal ölçeğe dönüştür (TRAIN)
original_test_data = scaler.inverse_transform(test_data[['Close_Normalized']])

# Orijinal ölçekteki değerleri yeni bir sütuna ekleyelim
test_data['Close'] = original_test_data

test_data

# Grafiği çizme
plt.figure(figsize=(15, 8))

# Orijinal Kapanış Fiyatlarını Çizme (NaN değerlerle)
plt.plot(full_df.index, full_df['Close'], label="Gerçek Kapanış (NaN ile)", color='blue', linestyle='-', linewidth=2)

# İmpütasyonla Doldurulmuş Değerleri Çizme - TRAIN verileri
plt.plot(final_df_grouped['Date'], final_df_grouped['Close'], label="TRAIN - İmpütasyonla Doldurulmuş Kapanış Değerleri", color='red', linestyle='--', linewidth=1)

# TEST verileri
plt.plot(test_data['Close'], label="TEST - NaN Değer İçeren Kapanış Değerleri", color='yellow', linestyle='--', linewidth=1)

plt.title("BIST30 Kapanış Fiyatları")
plt.xlabel("Tarih")
plt.ylabel("Kapanış Fiyatı")
plt.legend()
plt.show()

# Date sütununu DataFrame'in indeksi olarak ayarlama
final_df_grouped.set_index('Date', inplace=True)

# Şimdi, 'Close' sütununu seçerek eğitim setini oluşturabiliriz.
train = final_df_grouped['Close'].to_frame()

# Sonuçları kontrol et
train.head()

test = test_data[['Close']]
test.index.name = 'Date'
test







from sklearn.model_selection import cross_val_score

# Model listesi ve isimleri aynı kalacak
models = [
    DecisionTreeRegressor(),
    LinearRegression(),
    KNeighborsRegressor(n_neighbors=1),
    RandomForestRegressor(),
    SVR(kernel='rbf', C=100, epsilon=0.1),
]
model_names = [
    "Karar Ağacı",
    "Doğrusal Regresyon",
    "K-En Yakın Komşu",
    "Rastgele Orman",
    "Destek Vektör Makinesi"
]

# K-Fold Cross-Validation uygulama
for i, model in enumerate(models):
    # Veri seti ve hedef değerleri numpy dizisine dönüştür
    X = df_Close.values
    y = df['Close'].values

    # Cross-validation skorlarını hesapla
    cv_scores = cross_val_score(model, X, y, cv=10) # 10-fold CV

    # Sonuçları yazdır
    print(f'{model_names[i]} için Cross-Validation Sonuçları: {cv_scores}')
    print(f'Ortalama RMSE: {-cv_scores.mean()} (Standart sapma: {cv_scores.std()})\n')

# array'i dataset'e çevir
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    dataset = dataset.values  # Pandas serisini numpy dizisine çeviriyoruz
    for i in range(len(dataset)-time_step-1):
        dataX.append(dataset[i:(i+time_step)])
        dataY.append(dataset[i + time_step])
    return np.squeeze(np.array(dataX)), np.squeeze(np.array(dataY))

test_non_nan = test.dropna()
test_non_nan

train

test_non_nan

X_train, y_train = create_dataset(train, time_step)

test_non_nan

X_train

y_train

recursive_predictions = []  # list of numpy arrays olarak saklayacağız

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    preds = []

    last_window = X_train[-1].copy()

    for _ in range(len(test_data)):
        next_val = model.predict(last_window.reshape(1, -1))[0]
        preds.append(next_val)
        last_window = np.append(last_window[1:], next_val)

    recursive_predictions.append(np.array(preds))

# Sonuçları göster
for arr in recursive_predictions:
    print(arr)

test_non_nan

plt.figure(figsize=(15, 8))
plt.title('Kapanış Fiyatları ve Model Tahminleri', fontsize=16)
plt.xlabel('Tarih', fontsize=14)
plt.ylabel('Kapanış Fiyatı', fontsize=14)

# Gerçek tüm kapanışlar
plt.plot(df.index, df['Close'], label='Gerçek Değerler', color='blue')

# Geçerli test indekslerini al
valid_indices = [test.index.get_loc(date) for date in test_non_nan.index]

# Doğru tahminleri çizen bölüm
for i, pred in enumerate(recursive_predictions):
    if i <= 1:  # örnek: ilk 2 model
        filtered_pred = [pred[j] for j in valid_indices]
        plt.plot(test_non_nan.index, filtered_pred, label=f'{model_names[i]} Tahmin')

plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()

# English mapping of model names (if using Turkish names elsewhere)
format_map = {
    "Karar Ağacı": "Decision Tree",
    "Doğrusal Regresyon": "Linear Regression",
    "K-En Yakın Komşu": "K-Nearest Neighbors",
    "Rastgele Orman": "Random Forest",
    "Destek Vektör Makinesi": "Support Vector Machine"
}

# Get valid prediction indices that correspond to non-NaN dates
valid_indices = [test.index.get_loc(date) for date in test_non_nan.index]
prediction_dates = test_non_nan.index

# Plotting
plt.figure(figsize=(15, 8))
plt.title('Closing Prices and Predictions', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Closing Price', fontsize=14)

# Plot full actual data (train + test)
plt.plot(df.index, df['Close'], label='Actual Values', color='blue')

# Plot model predictions aligned to valid non-NaN dates
for i, pred in enumerate(recursive_predictions):
    if i <= 1:  # example: only first 2 models
        filtered_pred = [pred[j] for j in valid_indices]
        model_label = format_map.get(model_names[i], model_names[i])
        plt.plot(prediction_dates, filtered_pred, label=f'{model_label} Predictions')

plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
plt.title('Kapanış Fiyatları ve Model Tahminleri', fontsize=16)
plt.xlabel('Tarih', fontsize=14)
plt.ylabel('Kapanış Fiyatı', fontsize=14)

# Gerçek tüm kapanışlar
plt.plot(df.index, df['Close'], label='Gerçek Değerler', color='blue')

# Geçerli test indekslerini al
valid_indices = [test.index.get_loc(date) for date in test_non_nan.index]

# Doğru tahminleri çizen bölüm
for i, pred in enumerate(recursive_predictions):
    if 1<i:  # örnek: ilk 2 model
        filtered_pred = [pred[j] for j in valid_indices]
        plt.plot(test_non_nan.index, filtered_pred, label=f'{model_names[i]} Tahmin')

plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()

# English mapping of model names (if using Turkish names elsewhere)
format_map = {
    "Karar Ağacı": "Decision Tree",
    "Doğrusal Regresyon": "Linear Regression",
    "K-En Yakın Komşu": "K-Nearest Neighbors",
    "Rastgele Orman": "Random Forest",
    "Destek Vektör Makinesi": "Support Vector Machine"
}

# Get valid prediction indices that correspond to non-NaN dates
valid_indices = [test.index.get_loc(date) for date in test_non_nan.index]
prediction_dates = test_non_nan.index

# Plotting
plt.figure(figsize=(15, 8))
plt.title('Closing Prices and Predictions', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Closing Price', fontsize=14)

# Plot full actual data (train + test)
plt.plot(df.index, df['Close'], label='Actual Values', color='blue')

# Plot model predictions aligned to valid non-NaN dates
for i, pred in enumerate(recursive_predictions):
    if 1< i :  # example: only first 2 models
        filtered_pred = [pred[j] for j in valid_indices]
        model_label = format_map.get(model_names[i], model_names[i])
        plt.plot(prediction_dates, filtered_pred, label=f'{model_label} Predictions')

plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Gerçek test değerleri (NaN olmayanlar)
y_test = test_non_nan['Close'].values

# Performans ölçütlerini hesaplama
performans_olcumleri = []

# NaN olmayan tarihler test içinde hangi indekslerde?
valid_indices = [test.index.get_loc(tarih) for tarih in test_non_nan.index]

for i, model in enumerate(models):
    # recursive_predictions[i] tüm test dönemi kadar uzun
    # sadece geçerli tarihlere denk gelen tahminleri al
    tahmin = np.array([recursive_predictions[i][j] for j in valid_indices])

    mse = mean_squared_error(y_test, tahmin)
    mae = mean_absolute_error(y_test, tahmin)
    mape = mean_absolute_percentage_error(y_test, tahmin)
    r2 = r2_score(y_test, tahmin)
    rmse = sqrt(mse)
    bagil_rmse = rmse / np.mean(y_test)

    performans_olcumleri.append({
        "Model": model_names[i],
        "RMSE": rmse,
        "Bağıl RMSE": bagil_rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    })

# DataFrame ve sıralama
performans_df = pd.DataFrame(performans_olcumleri)
sirali_df = performans_df.sort_values(by='RMSE')

performans_df

sirali_df

from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
import pandas as pd

# Türkçe → İngilizce model adı eşlemesi
format_map = {
    "Karar Ağacı": "Decision Tree",
    "Doğrusal Regresyon": "Linear Regression",
    "K-En Yakın Komşu": "K-Nearest Neighbors",
    "Rastgele Orman": "Random Forest",
    "Destek Vektör Makinesi": "Support Vector Machine"
}

# Metriği hesaplayan yardımcı fonksiyon
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Bootstrap güven aralığı
def bootstrap_ci(data, num_samples=1000, ci=95):
    sample_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_samples)]
    lower = np.percentile(sample_means, (100 - ci) / 2)
    upper = np.percentile(sample_means, 100 - (100 - ci) / 2)
    return (lower, upper)

# Model karşılaştırma fonksiyonu (i indexlerine göre)
def compare_two_models(index1, index2, recursive_predictions, model_names, test_non_nan, test):
    # Gerçek değerler
    y_true = test_non_nan['Close'].values

    # Geçerli indeksler
    valid_indices = [test.index.get_loc(date) for date in test_non_nan.index]

    # Tahminleri al
    y_pred1 = np.array([recursive_predictions[index1][j] for j in valid_indices])
    y_pred2 = np.array([recursive_predictions[index2][j] for j in valid_indices])

    # Model isimlerini al
    name1 = format_map.get(model_names[index1], model_names[index1])
    name2 = format_map.get(model_names[index2], model_names[index2])

    # Metrikler
    m1 = calculate_metrics(y_true, y_pred1)
    m2 = calculate_metrics(y_true, y_pred2)

    errors1 = np.abs(y_true - y_pred1)
    errors2 = np.abs(y_true - y_pred2)

    # İstatistiksel testler
    t_stat, p_ttest = stats.ttest_rel(errors1, errors2)
    w_stat, p_wilcoxon = stats.wilcoxon(errors1, errors2)

    # Güven aralıkları
    ci1 = bootstrap_ci(errors1)
    ci2 = bootstrap_ci(errors2)

    # Sonuçlar
    results = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2", "Paired t-test p-value", "Wilcoxon p-value", "95% CI (mean error)"],
        name1: [m1[0], m1[1], m1[2], p_ttest, p_wilcoxon, ci1],
        name2: [m2[0], m2[1], m2[2], "", "", ci2]
    })

    return results

# Örneğin: Doğrusal Regresyon vs Rastgele Orman
results = compare_two_models(
    index1=1,  # Linear Regression
    index2=3,  # Random Forest
    recursive_predictions=recursive_predictions,
    model_names=model_names,
    test_non_nan=test_non_nan,
    test=test
)
print(results.to_string())

#excel_file_name = f'resuls.xlsx'
#sirali_df.to_excel(excel_file_name, index=False)