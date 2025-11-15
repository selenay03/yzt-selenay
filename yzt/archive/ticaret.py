import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

basket = pd.read_csv("C:/Users/you/OneDrive/Masaüstü/archive/basket_details.csv")
customer = pd.read_csv("C:/Users/you/OneDrive/Masaüstü/archive/customer_details.csv")

basket['basket_date'] = pd.to_datetime(basket['basket_date'])

df = basket.merge(customer, on="customer_id", how="left")
df.head()
total_items = df['basket_count'].sum()
unique_customers = df['customer_id'].nunique()
unique_products = df['product_id'].nunique()
top_products = df.groupby("product_id")["basket_count"].sum().sort_values(ascending=False)
sex_sales = df.groupby("sex")["basket_count"].sum()
df["age_group"] = pd.cut(df["customer_age"], bins=[18,25,35,45,60], labels=["18-25","25-35","35-45","45-60"])
age_sales = df.groupby("age_group")["basket_count"].sum()
sex_sales.plot(kind="bar")
plt.title("Cinsiyete Göre Sepet Adedi")
plt.ylabel("Adet")
plt.xlabel("Cinsiyet")
plt.show()
age_sales.plot(kind="bar")
plt.title("Yaş Gruplarına Göre Sepet Adedi")
plt.ylabel("Adet")
plt.xlabel("Yaş Grubu")
plt.show()
top_products.head(10).plot(kind="bar")
plt.title("En Çok Sepete Eklenen 10 Ürün")
plt.ylabel("Adet")
plt.xlabel("Product ID")
plt.show()
