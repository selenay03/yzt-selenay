import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.max_open_warning": 0})

# ---------------------------
# Aşama 1: Veri Yükleme ve Hazırlık
# ---------------------------

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Beklenen sütunlar
    for col in ['title', 'year', 'rating']:
        if col not in df.columns:
            raise KeyError(f"Beklenen sütun yok: {col}")

    # runtime temizleme
    if 'runtime' in df.columns:
        df['runtime'] = (df['runtime'].astype(str)
                         .str.replace(r'[^0-9]', '', regex=True)
                         .replace('', np.nan)
                         .astype(float))

    # numeric dönüşüm
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

    # genre
    if 'genre' in df.columns:
        df['primary_genre'] = (df['genre'].astype(str)
                               .str.split('[|,;/]').str[0].str.strip())

    # director vs
    for col in ['director', 'writer', 'cast']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # votes
    if 'votes' in df.columns:
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

    df = df.dropna(how='all')
    return df

# ---------------------------
# Aşama 2: Keşifsel Veri Analizi (EDA)
# ---------------------------

def eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.describe(include='all')
    missing = df.isna().sum().rename('missing_count')
    return pd.concat([summary, missing], axis=1, sort=False)

def top_directors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if 'director' not in df.columns:
        return pd.DataFrame()
    g = df.groupby('director').agg(
        movie_count=('title', 'count'),
        avg_rating=('rating', 'mean'),
        total_votes=('votes', 'sum') if 'votes' in df.columns else ('rating', 'count')
    ).dropna(subset=['avg_rating'])
    return g.sort_values(['movie_count', 'avg_rating'], ascending=[False, False]).head(n)

def genre_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if 'primary_genre' not in df.columns:
        return pd.DataFrame()
    g = df.groupby('primary_genre').agg(
        movie_count=('title', 'count'),
        avg_rating=('rating', 'mean'),
        avg_runtime=('runtime', 'mean')
    ).sort_values('movie_count', ascending=False)
    return g

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numerics = df.select_dtypes(include=[np.number])
    return numerics.corr()

# ---------------------------
# Aşama 3: Veri Görselleştirme
# ---------------------------

def plot_genre_distribution(df: pd.DataFrame, output_dir: str) -> str:
    plt.figure(figsize=(10,6))
    counts = df['primary_genre'].value_counts().head(15)
    counts.plot(kind='bar')
    plt.title('En Yaygın Türler (ilk 15)')
    plt.ylabel('Film Sayısı')
    plt.tight_layout()
    path = os.path.join(output_dir, 'genre_distribution.png')
    plt.savefig(path)
    plt.close()
    return path

def plot_rating_vs_runtime(df: pd.DataFrame, output_dir: str) -> str:
    plt.figure(figsize=(8,6))
    df_plot = df.dropna(subset=['rating', 'runtime'])
    plt.scatter(df_plot['runtime'], df_plot['rating'], alpha=0.6)
    plt.xlabel('Süre (dakika)')
    plt.ylabel('IMDb Puanı')
    plt.title('Puan vs Süre')
    plt.tight_layout()
    path = os.path.join(output_dir, 'rating_vs_runtime.png')
    plt.savefig(path)
    plt.close()
    return path

def plot_yearly_trends(df: pd.DataFrame, output_dir: str) -> str:
    df_year = df.dropna(subset=['year', 'rating']).groupby('year').agg(
        movie_count=('title','count'),
        avg_rating=('rating','mean')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(df_year['year'], df_year['movie_count'], alpha=0.6)
    ax1.set_xlabel('Yıl')
    ax1.set_ylabel('Film Sayısı', color='black')

    ax2 = ax1.twinx()
    ax2.plot(df_year['year'], df_year['avg_rating'], marker='o', color='orange')
    ax2.set_ylabel('Ortalama Puan', color='black')

    plt.title('Yıllara Göre Film Sayısı ve Ortalama Puan')
    plt.tight_layout()
    path = os.path.join(output_dir, 'yearly_trends.png')
    plt.savefig(path)
    plt.close()
    return path

# ---------------------------
# Aşama 4: Raporlama
# ---------------------------

def save_report(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    eda = eda_summary(df)
    eda.to_csv(os.path.join(output_dir, 'eda_summary.csv'))

    top_dirs = top_directors(df, n=20)
    top_dirs.to_csv(os.path.join(output_dir, 'top_directors.csv'))

    genres = genre_analysis(df)
    genres.to_csv(os.path.join(output_dir, 'genre_analysis.csv'))

    plot_genre_distribution(df, output_dir)
    plot_rating_vs_runtime(df, output_dir)
    plot_yearly_trends(df, output_dir)

    html_parts = [
        '<html><head><meta charset="utf-8"><title>IMDb Veri Seti Analiz Raporu</title></head><body>',
        '<h1>IMDb Veri Seti Analiz Raporu</h1>',
        '<h2>Özet İstatistikler</h2>',
        eda.to_html(classes="table table-striped"),
        '<h2>Tür Dağılımı (Grafik)</h2>',
        '<img src="genre_distribution.png" width="800">',
        '<h2>Puan vs Süre</h2>',
        '<img src="rating_vs_runtime.png" width="600">',
        '<h2>Yıllara Göre Trend</h2>',
        '<img src="yearly_trends.png" width="800">',
        '<h2>En Çok Film Yapan / En Başarılı Yönetmenler</h2>',
        top_dirs.to_html(classes="table table-striped"),
        '</body></html>'
    ]

    with open(os.path.join(output_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))


# ---------------------------
# Program Başlangıcı
# ---------------------------

if __name__ == '__main__':
    DATA_PATH = "movies_initial.csv"    # CSV dosyan
    OUTPUT_DIR = "."                    # Çıktılar aynı klasöre

    print('Veri yükleniyor...')
    df = load_data(DATA_PATH)

    print('Veri temizleniyor...')
    df = clean_data(df)

    print('EDA özet çıktı (ilk bazı satırlar):')
    print(eda_summary(df).head())

    print('En iyi yönetmenler:')
    print(top_directors(df, n=10))

    print('Rapor oluşturuluyor...')
    save_report(df, OUTPUT_DIR)

    print(f'Rapor ve görseller klasöre kaydedildi.')
