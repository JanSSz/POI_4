import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Parsowanie danych wejściowych
data_dict = {}
with open('gres_128_128_features.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(': ')
        if len(parts) == 2:  # Upewniamy się, że linia zawiera dwie części oddzielone dwukropkiem
            feature_name = parts[0]
            values = [float(val) for val in parts[1].strip('[]').split()]
            data_dict[feature_name] = values

# Konwersja na tablicę NumPy
max_len = max(len(data_dict[key]) for key in data_dict)
data_array = np.zeros((len(data_dict), max_len))
for i, key in enumerate(data_dict):
    data_array[i, :len(data_dict[key])] = data_dict[key]

# Wyodrębnienie wektora cech do macierzy X
X = data_array[:, 1:].T  # Pomijamy pierwszą kolumnę, która zawiera etykiety

# Wyodrębnienie etykiet kategorii do tablicy y
y = data_array[:, 0]

# Wstępne przetwarzanie danych
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)

onehot_encoder = OneHotEncoder(sparse=False)
y_int = y_int.reshape(len(y_int), 1)
y_onehot = onehot_encoder.fit_transform(y_int)
print("Liczność X:", X.shape[0])
print("Liczność y:", y.shape[0])

# Usunięcie zbędnych etykiet kategorii
y_onehot = np.delete(y_onehot, np.where(np.sum(X, axis=1) == 0), axis=0)
y_onehot = y_onehot[:-1]  # Usunięcie ostatniej etykiety kategorii

# Podział zbioru na część treningową i testową
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3)

# Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Dense(units=10, activation='sigmoid', input_dim=X_train.shape[1]))
model.add(Dense(units=y_onehot.shape[1], activation='softmax'))  # Liczba jednostek odpowiada liczbie unikalnych kategorii

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Uczenie sieci
model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)

# Testowanie sieci
y_pred = model.predict(X_test)

# Przekonwertowanie wektorów y_test oraz y_pred do kodowania całkowitego
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)

# Wyliczenie macierzy pomyłek
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)
