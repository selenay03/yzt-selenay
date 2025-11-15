import pandas as pd
import matplotlib.pyplot as plt

# CSV dosyasını doğru ayırıcı ile yükle
df = pd.read_csv("C:\\Users\\you\\OneDrive\\Masaüstü\\munich.csv", sep=";")

# time kolonunu datetime biçimine çevir
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# index olarak time kolonunu kullan
df = df.set_index('time')

# Kolonları daha kolay kullanmak için yeniden isimlendirelim
df.rename(columns={
    "precipitation_sum (mm)": "rain_mm",
    "snowfall_sum (cm)": "snow_cm"
}, inplace=True)

# NaN verileri 0 yap (istersen)
df = df.fillna(0)

print(df.head())
print("\nVeri seti boyutu:", df.shape)
print("\nİstatistiksel Özet:\n", df.describe())

# Toplam değerler
print("\nToplam Yağış (mm):", df['rain_mm'].sum())
print("Toplam Kar (cm):", df['snow_cm'].sum())

# En yağışlı gün
max_rain_day = df['rain_mm'].idxmax()
max_rain_value = df['rain_mm'].max()
print("\nEn yağışlı gün:", max_rain_day, "→", max_rain_value, "mm")

# Günlük yağış grafiği
plt.figure(figsize=(16,4))
plt.plot(df.index, df['rain_mm'])
plt.title("Günlük Yağış Miktarı (mm)")
plt.xlabel("Tarih")
plt.ylabel("mm")
plt.grid(True)
plt.show()

# Aylık yağış
monthly_rain = df['rain_mm'].resample('M').sum()
plt.figure(figsize=(10,5))
monthly_rain.plot(kind='bar')
plt.title("Aylara Göre Toplam Yağış")
plt.ylabel("mm")
plt.show()

# Kar yağışı grafiği
if df['snow_cm'].sum() > 0:
    plt.figure(figsize=(16,4))
    plt.plot(df.index, df['snow_cm'])
    plt.title("Günlük Kar Yağışı (cm)")
    plt.ylabel("cm")
    plt.grid(True)
    plt.show()
else:
    print("→ Bu dönemde neredeyse hiç kar yağışı yok.")
