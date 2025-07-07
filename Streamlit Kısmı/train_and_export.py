# train_and_export.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# Veriyi oku
df = pd.read_csv("train.csv")
X = df.drop("price_range", axis=1)
y = df["price_range"]

# Standardize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Özellik seçimi (RFE)
model_lr = LogisticRegression(max_iter=1000)
selector = RFE(estimator=model_lr, n_features_to_select=5)
X_selected = selector.fit_transform(X_scaled, y)

# Eğitim
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, stratify=y, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Kaydet
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("selector.pkl", "wb") as f:
    pickle.dump(selector, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ Model, scaler ve seçici kaydedildi.")
