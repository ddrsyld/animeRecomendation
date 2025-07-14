from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# --- Memuat Model ---
print("Memuat model dari arsip...")
try:
    model_data = joblib.load('anime_recommender.pkl')
    anime_df = model_data['anime_df']
    cosine_sim = model_data['cosine_sim']
    indices = pd.Series(anime_df.index, index=anime_df['name']).drop_duplicates()
    print("Model berhasil dimuat!")
except FileNotFoundError:
    print("File 'anime_recommender.pkl' tidak ditemukan!")
    exit()

def get_genre_recommendations(anime_name):
    """Fungsi untuk mendapatkan rekomendasi."""
    try:
        idx = indices[anime_name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        anime_indices = [i[0] for i in sim_scores]
        recommendations = anime_df[['name', 'rating']].iloc[anime_indices]
        return list(recommendations.itertuples(index=False, name=None))
    except KeyError:
        return [("Anime tidak ditemukan.", "")]

@app.route('/', methods=['GET', 'POST'])
def home():
    """Rute utama halaman web."""
    # Daftar untuk autocomplete input
    anime_titles_for_datalist = sorted(anime_df['name'].dropna().unique().tolist())
    
    # Daftar LENGKAP untuk galeri interaktif (nama dan rating)
    full_anime_gallery = list(anime_df[['name', 'rating']].dropna().itertuples(index=False, name=None))

    recommendations = None
    anime_input = ""
    if request.method == 'POST':
        anime_input = request.form.get('anime_name')
        if anime_input:
            recommendations = get_genre_recommendations(anime_input)
    
    return render_template(
        'index.html',
        recommendations=recommendations,
        anime=anime_input,
        anime_list=anime_titles_for_datalist,
        gallery_list=full_anime_gallery
    )

if __name__ == '__main__':
    app.run(debug=True)