import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Memuat Dataset
dataset = pd.read_csv("dataset.csv")

# Hapus attribute yang tidak digunakan
dataset = dataset.drop(columns=['KECAMATAN'])

# Check Missing Value
missing_value = ['Tidak dilakukan', 'Tidak diketahui']
dataset['UMUR'].isna()
dataset['JENIS KELAMIN'].isnull()
dataset['FOTO TORAKS'].replace(missing_value[0], pd.NA, inplace=True)
dataset['STATUS HIV'].replace(missing_value[1], pd.NA, inplace=True)
dataset['RIWAYAT DIABETES'].replace(missing_value[1], pd.NA, inplace=True)
dataset['HASIL TCM'].replace(missing_value[0], pd.NA, inplace=True)
dataset['LOKASI ANATOMI (target/output)'].isnull()

missing = dataset.isnull().sum()

# Normalisasi Data
min_umur = dataset['UMUR'].min()
max_umur = dataset['UMUR'].max()
dataset['UMUR'] = (dataset['UMUR'] - min_umur) / (max_umur - min_umur)
dataset['JENIS KELAMIN'] = dataset['JENIS KELAMIN'].replace({'P': 0, 'L': 1})
dataset['FOTO TORAKS'] = dataset['FOTO TORAKS'].replace({'Negatif': 0, 'Positif': 1})
dataset['STATUS HIV'] = dataset['STATUS HIV'].replace({'Negatif': 0, 'Positif': 1})
dataset['RIWAYAT DIABETES'] = dataset['RIWAYAT DIABETES'].replace({'Tidak': 0, 'Ya': 1})
dataset['HASIL TCM'] = dataset['HASIL TCM'].replace({'Rif resisten': 2, 'Negatif': 0, 'Rif Sensitif': 1})
dataset['LOKASI ANATOMI (target/output)'] = dataset['LOKASI ANATOMI (target/output)'].replace({'Paru': 1, 'Ekstra paru': 0})

# Mengisi missing value dengan modus
dataset['FOTO TORAKS'].fillna(dataset['FOTO TORAKS'].mode()[0], inplace=True)
dataset['STATUS HIV'].fillna(dataset['STATUS HIV'].mode()[0], inplace=True)
dataset['RIWAYAT DIABETES'].fillna(dataset['RIWAYAT DIABETES'].mode()[0], inplace=True)
dataset['HASIL TCM'].fillna(dataset['HASIL TCM'].mode()[0], inplace=True)

# Memisahkan fitur dan target
X = dataset.drop(columns=['LOKASI ANATOMI (target/output)'])
y = dataset['LOKASI ANATOMI (target/output)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(X_test)
# Inisialisasi model
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

y_predict = perceptron.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_predict)
print (f"Iteration: {perceptron.n_iter_}")
print (f"Accuray: {accuracy}")

confusion = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(confusion)

# Dump model
# model_data = {
#     'model': perceptron,
#     'min_umur': min_umur,
#     'max_umur': max_umur
# }

# pickle.dump(model_data, open("model.pkl", "wb"))