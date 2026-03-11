"""
🎬 Movie Recommendation System - Streamlit App
Dataset: TMDB 5000 Movie Dataset (Kaggle)

Models:
- Content-Based Filtering (TF-IDF + Cosine Similarity)
- KNN-Based Filtering
- Collaborative Filtering (Simulated SVD)
- Hybrid Approach
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import requests
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main-header { 
        font-size: 3rem; 
        color: #E50914; 
        text-align: center; 
        font-weight: bold; 
        margin-bottom: 0.5rem; 
    }
    .sub-header { 
        font-size: 1.2rem; 
        color: #666; 
        text-align: center; 
        margin-bottom: 2rem; 
    }
    .movie-card { 
        background-color: #f8f9fa; 
        padding: 1rem; 
        border-radius: 10px; 
        margin: 1.5rem 0; 
        border-left: 4px solid #E50914; 
    }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
    }
    .stButton>button { 
        background-color: #E50914; 
        color: white; 
        border-radius: 5px;
        padding: 0.5rem 2rem; 
        font-weight: bold; 
        border: none; 
    }
    .stButton>button:hover { 
        background-color: #b20710; 
    }
    .algo-badge { 
        display: inline-block; 
        padding: 4px 10px; 
        border-radius: 20px;
        font-size: 0.75rem; 
        font-weight: bold; 
        margin-bottom: 8px; 
    }
    .badge-content { 
        background-color: #dbeafe; 
        color: #1d4ed8; 
    }
    .badge-knn { 
        background-color: #dcfce7; 
        color: #15803d; 
    }
    .badge-collab { 
        background-color: #fef9c3; 
        color: #854d0e; 
    }
    .badge-hybrid { 
        background-color: #fce7f3; 
        color: #be185d; 
    }
    .gold-rank { 
        color: #FFD700; 
        font-weight: 700; 
    }
    .silver-rank { 
        color: #C0C0C0; 
        font-weight: 700; 
    }
    .bronze-rank { 
        color: #CD7F32; 
        font-weight: 700; 
    }
    .movie-title-link a {
        color: inherit; 
        text-decoration: none; 
        font-weight: bold; 
        font-size: 1rem;
    }
    .movie-title-link a:hover { 
        color: #E50914; 
        text-decoration: underline; 
    }
    .model-card {
        background: #f8f9fa; 
        border-radius: 10px; 
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem; 
        border-left: 4px solid #E50914;
    }
    .model-card h4 { 
        margin-top: 0; 
        margin-bottom: 0.4rem; 
    }
    .model-card table { 
        width: 100%; 
        font-size: 0.9rem; 
        border-collapse: collapse; 
        margin-top: 0.6rem; 
    }
    .model-card td, .model-card th { 
        padding: 4px 8px; 
        border: 1px solid #dee2e6; 
    }
    .model-card th { 
        background: #e9ecef; 
    }
    </style>
""", unsafe_allow_html=True)

# ── TMDB API ───────────────────────────────────────────────────────────────────
try:
    TMDB_API_KEY = st.secrets["API_KEY"]
except Exception:
    TMDB_API_KEY = ""

TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH_URL     = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_BASE_URL = "https://www.themoviedb.org/movie"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400)
def get_movie_tmdb_data(movie_title):
    """
    Single API call returning (poster_url, overview, tmdb_id).
    Replaces separate get_movie_poster / get_movie_overview calls.
    """
    if not TMDB_API_KEY:
        return None, "No synopsis available.", None
    try:
        params = {'api_key': TMDB_API_KEY, 'query': movie_title, 'language': 'en-US', 'page': 1}
        r = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        if r.status_code == 200:
            results = r.json().get('results', [])
            if results:
                hit      = results[0]
                poster   = f"{TMDB_IMAGE_BASE_URL}{hit['poster_path']}" if hit.get('poster_path') else None
                overview = hit.get('overview', 'No synopsis available.') or 'No synopsis available.'
                tmdb_id  = hit.get('id')
                return poster, overview, tmdb_id
    except Exception:
        pass
    return None, "No synopsis available.", None


def tmdb_url(tmdb_id, fallback_title=""):
    """Return a TMDB movie page URL."""
    if tmdb_id:
        return f"{TMDB_MOVIE_BASE_URL}/{tmdb_id}"
    q = str(fallback_title).replace(' ', '+')
    return f"https://www.themoviedb.org/search?query={q}"


def clickable_title(title, tmdb_id=None):
    """HTML anchor that opens TMDB in a new tab."""
    url = tmdb_url(tmdb_id, title)
    return f'<span class="movie-title-link"><a href="{url}" target="_blank">{title}</a></span>'


def load_image_from_url(url):
    try:
        r = requests.get(url, timeout=5)
        return Image.open(BytesIO(r.content))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    """Load movie DataFrame + cosine-sim matrix from pickle or CSV."""
    try:
        with open('movie_recommendation_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['movies'], data['cosine_sim'], data['indices'], data['m'], data['C']
    except FileNotFoundError:
        pass

    try:
        movies  = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
    except FileNotFoundError:
        st.error("Data files not found. Please add tmdb_5000_movies.csv and tmdb_5000_credits.csv.")
        st.stop()

    movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
    movies = movies[['id', 'title_x', 'overview', 'genres', 'keywords',
                     'cast', 'crew', 'vote_average', 'vote_count', 'popularity']]
    movies.rename(columns={'title_x': 'title'}, inplace=True)
    movies = movies.dropna(subset=['overview'])

    def extract_names(text, key='name', limit=3):
        if pd.isna(text):
            return []
        try:
            return [item[key] for item in ast.literal_eval(text)[:limit]]
        except Exception:
            return []

    def get_director(crew_str):
        if pd.isna(crew_str):
            return []
        try:
            for member in ast.literal_eval(crew_str):
                if member.get('job') == 'Director':
                    return [member.get('name', '')]
        except Exception:
            pass
        return []

    def clean_text(lst):
        if isinstance(lst, list):
            return ' '.join([str(i).lower().replace(' ', '') for i in lst])
        return ''

    movies['genres_list']    = movies['genres'].apply(lambda x: extract_names(x, 'name', 5))
    movies['keywords_list']  = movies['keywords'].apply(lambda x: extract_names(x, 'name', 5))
    movies['cast_list']      = movies['cast'].apply(lambda x: extract_names(x, 'name', 3))
    movies['director']       = movies['crew'].apply(get_director)
    movies['genres_clean']   = movies['genres_list'].apply(clean_text)
    movies['keywords_clean'] = movies['keywords_list'].apply(clean_text)
    movies['cast_clean']     = movies['cast_list'].apply(clean_text)
    movies['director_clean'] = movies['director'].apply(clean_text)
    movies['soup'] = (
        movies['overview'].fillna('') + ' ' +
        movies['genres_clean'] + ' ' +
        movies['keywords_clean'] + ' ' +
        movies['cast_clean'] + ' ' +
        movies['director_clean']
    )

    tfidf      = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_mat  = tfidf.fit_transform(movies['soup'])
    cosine_sim = cosine_similarity(tfidf_mat, tfidf_mat)
    indices    = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(0.90)

    return movies, cosine_sim, indices, m, C


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BUILDERS  (lazy - only built when that algorithm is selected)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_knn_model(_movies_df):
    """KNN on reduced TF-IDF (3 000 features) for a faster first-time build."""
    tfidf     = TfidfVectorizer(max_features=3000, stop_words='english')
    tfidf_mat = tfidf.fit_transform(_movies_df['soup'].fillna(''))
    knn = NearestNeighbors(n_neighbors=21, metric='cosine', algorithm='brute', n_jobs=-1)
    knn.fit(tfidf_mat)
    return knn, tfidf_mat


@st.cache_resource(show_spinner=False)
def build_collab_model(_movies_df):
    """
    Lightweight Collaborative Filtering via SVD.
    Uses 100 virtual users and vectorised numpy ops instead of a Python loop.
    k=15 latent factors.
    """
    np.random.seed(42)
    n_movies = len(_movies_df)
    n_users  = 100

    all_genres = []
    for gl in _movies_df['genres_list'].dropna():
        if isinstance(gl, list):
            all_genres.extend(gl)
    unique_genres = list(set(all_genres)) or ['Action', 'Drama', 'Comedy']

    # Vectorised genre membership matrix (n_movies × n_genres)
    genre_index = {g: i for i, g in enumerate(unique_genres)}
    membership  = np.zeros((n_movies, len(unique_genres)), dtype=np.float32)
    for i, gl in enumerate(_movies_df['genres_list'].reset_index(drop=True)):
        if isinstance(gl, list):
            for g in gl:
                if g in genre_index:
                    membership[i, genre_index[g]] = 1.0

    ratings_arr   = _movies_df['vote_average'].values.astype(np.float32)
    rating_matrix = np.zeros((n_users, n_movies), dtype=np.float32)

    for u in range(n_users):
        n_pref   = np.random.randint(1, 4)
        pref_idx = np.random.choice(len(unique_genres), size=n_pref, replace=False)
        liked    = membership[:, pref_idx].sum(axis=1) > 0
        noise    = np.random.normal(0, 0.5, n_movies).astype(np.float32)
        rating_matrix[u] = np.where(liked, np.clip(ratings_arr + noise, 0, 10), 0)

    sparse_mat = csr_matrix(rating_matrix, dtype=np.float64)
    k = min(15, min(sparse_mat.shape) - 1)
    U, sigma, Vt = svds(sparse_mat, k=k)
    return np.dot(np.dot(U, np.diag(sigma)), Vt)


# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def weighted_rating(row, m, C):
    v, R = row['vote_count'], row['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


def get_content_recommendations(title, movies_df, cosine_sim, indices, top_n=10):
    try:
        idx    = indices[title]
        scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
        mi     = [s[0] for s in scores]
        sims   = [s[1] for s in scores]
        recs = movies_df.iloc[mi][['id', 'title', 'genres_list', 'vote_average', 'vote_count', 'popularity']].copy()
        recs['similarity_score'] = sims
        recs['model_score']      = sims
        recs['rank'] = range(1, len(recs) + 1)
        return recs
    except KeyError:
        return None


def get_hybrid_recommendations(title, movies_df, cosine_sim, indices, m_val, C_val, top_n=10):
    content = get_content_recommendations(title, movies_df, cosine_sim, indices, top_n=50)
    if content is None:
        return None

    rec_idx   = movies_df[movies_df['title'].isin(content['title'])].index
    m_h       = movies_df['vote_count'].quantile(0.70)
    qualified = movies_df.loc[rec_idx][movies_df.loc[rec_idx]['vote_count'] >= m_h].copy()

    if len(qualified) == 0:
        result = content.head(top_n).copy()
        result['model_score'] = result['similarity_score']
        result['rank'] = range(1, len(result) + 1)
        return result

    qualified['weighted_score'] = qualified.apply(lambda r: weighted_rating(r, m_val, C_val), axis=1)
    hybrid = content.merge(qualified[['title', 'weighted_score']], on='title', how='inner')
    hybrid['hybrid_score'] = 0.7 * hybrid['similarity_score'] + 0.3 * hybrid['weighted_score'] / 10
    hybrid['model_score']  = hybrid['hybrid_score']
    hybrid = hybrid.sort_values('hybrid_score', ascending=False).head(top_n)
    hybrid['rank'] = range(1, len(hybrid) + 1)
    return hybrid


def get_knn_recommendations(title, movies_df, knn, tfidf_mat, top_n=10):
    try:
        idx           = movies_df[movies_df['title'] == title].index[0]
        dists, nidxs  = knn.kneighbors(tfidf_mat[idx], n_neighbors=top_n + 1)
        results       = [(ni, 1 - d) for d, ni in zip(dists[0], nidxs[0]) if ni != idx][:top_n]
        mi   = [r[0] for r in results]
        sims = [r[1] for r in results]
        recs = movies_df.iloc[mi][['id', 'title', 'genres_list', 'vote_average', 'vote_count', 'popularity']].copy()
        recs['similarity_score'] = sims
        recs['model_score']      = sims
        recs['rank'] = range(1, len(recs) + 1)
        return recs
    except (IndexError, KeyError):
        return None


def get_collab_recommendations(title, movies_df, pred_ratings, top_n=10):
    try:
        idx       = movies_df[movies_df['title'] == title].index[0]
        col       = idx if idx < pred_ratings.shape[1] else 0
        top_users = np.argsort(pred_ratings[:, col])[-50:]
        avg       = pred_ratings[top_users, :].mean(axis=0)
        avg[col]  = -1
        top_idx   = np.argsort(avg)[::-1][:top_n]
        top_scores = avg[top_idx]
        max_s      = top_scores.max() or 1
        norm       = top_scores / max_s
        valid      = [i for i in top_idx if i < len(movies_df)][:top_n]
        recs = movies_df.iloc[valid][['id', 'title', 'genres_list', 'vote_average', 'vote_count', 'popularity']].copy()
        recs['similarity_score'] = norm[:len(valid)]
        recs['model_score']      = norm[:len(valid)]
        recs['rank'] = range(1, len(recs) + 1)
        return recs
    except (IndexError, KeyError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_knn_model():
    return build_knn_model(movies_df)


@st.cache_resource
def load_collab_model():
    return build_collab_model(movies_df)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD CORE DATA
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner('🎬 Loading movie database...'):
    movies_df, cosine_sim, indices, m_value, C_value = load_data()


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore, discover, and get personalized movie picks</p>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    st.markdown("### 🔑 TMDB API Key")
    api_key_input = st.text_input(
        "Enter your TMDB API Key", value="", type="password",
        help="Get a free key at themoviedb.org"
    )
    if api_key_input:
        TMDB_API_KEY = api_key_input
        st.success("✅ API Key set!")
    else:
        st.info("💡 Add API key to show movie posters & enable TMDB links")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
        **Dataset:** TMDB 5000 Movies

        **Models:**
        - Content-Based Filtering
        - KNN-Based Filtering ⭐
        - Collaborative Filtering ⭐
        - Hybrid Approach

        **Tech Stack:**
        Python · Scikit-learn
        SciPy · Streamlit · Pandas
        TMDB API
    """)

    st.markdown("---")
    with st.expander("📚 How to get TMDB API Key?"):
        st.markdown("""
            1. Visit [themoviedb.org](https://www.themoviedb.org/)
            2. Create a free account
            3. Settings → API → Request key
            4. Copy & paste above ⬆️
        """)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📊 Dataset Stats", "ℹ️ How It Works"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 - Recommendations
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## 🔍 Find Similar Movies")

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_movie = st.selectbox(
            "Select a movie you like:",
            options=sorted(movies_df['title'].unique()),
            index=None,
            placeholder="Start typing to search..."
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🎬 Search", width='stretch')

    col_algo, col_num = st.columns(2)
    with col_algo:
        algorithm = st.selectbox(
            "🔍 Recommendation Algorithm",
            [
                "Hybrid (Content + Rating)",
                "Content-Based (TF-IDF + Cosine)",
                "KNN-Based Filtering",
                "Collaborative Filtering (SVD)",
            ]
        )
    with col_num:
        num_recommendations = st.slider("📊 Number of Recommendations", 5, 20, 10)

    show_posters = st.checkbox("🖼️ Show Posters and Description", value=True)

    if selected_movie or search_button:
        if not selected_movie:
            st.info("👆 Please select a movie first.")
        else:
            movie_details = movies_df[movies_df['title'] == selected_movie].iloc[0]

            # Fetch TMDB data for selected movie (single API call)
            sel_poster, sel_overview, sel_tmdb_id = get_movie_tmdb_data(selected_movie)
            sel_link = tmdb_url(sel_tmdb_id, selected_movie)

            # ── Selected movie card ──────────────────────────────────────────
            st.markdown(f"""
            <div class="movie-card">
                <span style="font-size:14px; color:#aaa;">🎬 YOU SELECTED: </span>
                <span style="font-size:24px; font-weight:bold;">
                    <a href="{sel_link}" target="_blank" style="color:inherit;text-decoration:none;">
                        {selected_movie}
                    </a>
                </span>
            </div>
            """, unsafe_allow_html=True)

            poster_url = sel_poster if (show_posters and TMDB_API_KEY) else None
            overview_text = sel_overview if (sel_overview and sel_overview != "No synopsis available.") \
                            else str(movie_details.get('overview', ''))

            if poster_url:
                cp, ci = st.columns([1, 3])
                with cp:
                    img = load_image_from_url(poster_url)
                    if img:
                        st.image(img, width='stretch')
                with ci:
                    st.markdown(clickable_title(movie_details['title'], sel_tmdb_id),
                                unsafe_allow_html=True)
                    gl = movie_details.get('genres_list')
                    if isinstance(gl, list) and gl:
                        st.markdown(f"**Genres:** {', '.join(gl)}")
                    if overview_text:
                        st.markdown(f"_{overview_text[:300]}..._")
                    m1, m2 = st.columns(2)
                    m1.metric("⭐ Rating", f"{movie_details['vote_average']:.1f}/10",
                              help="Average user rating from TMDB dataset (scale 0-10).")
                    m2.metric("👥 Votes", f"{int(movie_details['vote_count']):,}",
                              help="Total number of user ratings.")
            else:
                ca, cb, cc = st.columns([2, 1, 1])
                with ca:
                    st.markdown(clickable_title(movie_details['title'], sel_tmdb_id),
                                unsafe_allow_html=True)
                    gl = movie_details.get('genres_list')
                    if isinstance(gl, list) and gl:
                        st.markdown(f"**Genres:** {', '.join(gl)}")
                    if overview_text:
                        st.markdown(f"_{overview_text[:200]}..._")
                cb.metric("⭐ Rating", f"{movie_details['vote_average']:.1f}/10",
                          help="Average user rating from TMDB dataset (scale 0-10).")
                cc.metric("👥 Votes", f"{int(movie_details['vote_count']):,}",
                          help="Total number of user ratings.")

            # ── Algorithm badge ──────────────────────────────────────────────
            st.markdown("---")
            algo_map = {
                "Hybrid (Content + Rating)":       ("hybrid",  "🔀 Hybrid Algorithm"),
                "Content-Based (TF-IDF + Cosine)": ("content", "📄 Content-Based"),
                "KNN-Based Filtering":             ("knn",     "🔵 KNN-Based"),
                "Collaborative Filtering (SVD)":   ("collab",  "🤝 Collaborative Filtering"),
            }
            badge_cls, badge_label = algo_map[algorithm]
            st.markdown(
                f'<span class="algo-badge badge-{badge_cls}">{badge_label}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f"### 🎯 Top {num_recommendations} Recommendations")

            # ── Build model & compute recommendations ────────────────────────
            with st.spinner("🤖 Computing recommendations..."):
                if algorithm == "Hybrid (Content + Rating)":
                    recommendations = get_hybrid_recommendations(
                        selected_movie, movies_df, cosine_sim, indices,
                        m_value, C_value, top_n=num_recommendations
                    )
                elif algorithm == "Content-Based (TF-IDF + Cosine)":
                    recommendations = get_content_recommendations(
                        selected_movie, movies_df, cosine_sim, indices,
                        top_n=num_recommendations
                    )
                elif algorithm == "KNN-Based Filtering":
                    knn_model, tfidf_matrix = load_knn_model()
                    recommendations = get_knn_recommendations(
                        selected_movie, movies_df, knn_model, tfidf_matrix,
                        top_n=num_recommendations
                    )
                else:  # Collaborative Filtering (SVD)
                    predicted_ratings = load_collab_model()
                    recommendations = get_collab_recommendations(
                        selected_movie, movies_df, predicted_ratings,
                        top_n=num_recommendations
                    )

            # ── Render results ───────────────────────────────────────────────
            if recommendations is not None and len(recommendations) > 0:
                for _, row in recommendations.iterrows():
                    rank = int(row["rank"])
                    rank_class = {1: "gold-rank", 2: "silver-rank", 3: "bronze-rank"}.get(rank, "")
                    medal      = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")

                    # Unified metric values
                    rating      = f"{row['vote_average']:.1f}/10"
                    votes       = f"{int(row['vote_count']):,}"
                    similarity  = f"{row.get('similarity_score', 0) * 100:.0f}%"
                    model_score = f"{row.get('model_score', 0):.3f}"

                    # Fetch TMDB data for each result (cached per title)
                    rec_poster, rec_overview, rec_tmdb_id = None, "", None
                    if TMDB_API_KEY:
                        rec_poster, rec_overview, rec_tmdb_id = get_movie_tmdb_data(row['title'])
                    if not show_posters:
                        rec_poster = None

                    rec_link      = tmdb_url(rec_tmdb_id, row['title'])
                    title_html    = clickable_title(row['title'], rec_tmdb_id)

                    st.markdown(
                        f'<div class="movie-card">'
                        f'<h4 class="{rank_class}">{medal} #{rank}. '
                        f'<a href="{rec_link}" target="_blank" style="color:inherit;text-decoration:none;">'
                        f'{row["title"]}</a></h4>',
                        unsafe_allow_html=True
                    )

                    if rec_poster:
                        cp2, ci2, cm2 = st.columns([1, 3, 2])
                        with cp2:
                            rp = load_image_from_url(rec_poster)
                            if rp:
                                st.image(rp, width='stretch')
                        with ci2:
                            st.markdown(title_html, unsafe_allow_html=True)
                            gl2 = row.get('genres_list')
                            if isinstance(gl2, list) and gl2:
                                st.markdown(f"🎭 {', '.join(gl2)}")
                            if rec_overview:
                                st.markdown(
                                    f"<p style='font-size:13px;color:#888'>{rec_overview[:200]}...</p>",
                                    unsafe_allow_html=True
                                )
                        with cm2:
                            ma, mb = cm2.columns(2)
                            ma.metric("⭐ Rating",      rating,      help="Average user rating (scale 0-10).")
                            mb.metric("👥 Votes",       votes,       help="Total number of user votes.")
                            ma.metric("🎯 Similarity",  similarity,  help="Content similarity to the selected movie.")
                            mb.metric("🤖 Model Score", model_score, help="Score produced by the selected algorithm.")
                    else:
                        ca2, cb2, cc2, cd2, ce2 = st.columns([3, 1, 1, 1, 1])
                        with ca2:
                            st.markdown(title_html, unsafe_allow_html=True)
                            gl2 = row.get('genres_list')
                            if isinstance(gl2, list) and gl2:
                                st.markdown(f"🎭 {', '.join(gl2)}")
                        cb2.metric("⭐ Rating",      rating,      help="Average user rating (scale 0-10).")
                        cc2.metric("👥 Votes",       votes,       help="Total number of user votes.")
                        cd2.metric("🎯 Similarity",  similarity,  help="Content similarity to the selected movie.")
                        ce2.metric("🤖 Model Score", model_score, help="Score produced by the selected algorithm.")

                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.download_button(
                    "📥 Download Recommendations as CSV",
                    data=recommendations.to_csv(index=False),
                    file_name=f"recommendations_{selected_movie.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No recommendations found. Try a different movie!")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 - Dataset Stats
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 📊 Dataset Statistics")

    # Pre-compute genre data
    all_genres_flat = [g for gl in movies_df['genres_list'].dropna()
                    if isinstance(gl, list) for g in gl]
    genre_counts = pd.Series(all_genres_flat).value_counts()

    # ── Top-level metrics ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("🎬 Total Movies", f"{len(movies_df):,}"),
        ("⭐ Avg Rating", f"{movies_df['vote_average'].mean():.2f}/10"),
        ("👥 Total Votes", f"{movies_df['vote_count'].sum() / 1e6:.1f}M"),
        ("🎭 Unique Genres", str(len(genre_counts))),
    ]

    for col, (label, value) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.metric(label, value)

    st.markdown("---")
    
    # ── Row 1: Rating distribution + Top rated ───────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Rating Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(movies_df['vote_average'], bins=20, color='#E50914', edgecolor='black')
        ax.set_xlabel('Vote Average')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Movie Ratings')
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("### 🏆 Top 10 Rated Movies")
        st.dataframe(
            movies_df.nlargest(10, 'vote_average')[['title', 'vote_average', 'vote_count']],
            width='stretch', hide_index=True
        )

    st.markdown("---")

    # ── Row 2: Genre bar chart + Genre table ─────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 🎭 Genre Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        colors = plt.cm.Set3(np.linspace(0, 1, len(genre_counts)))
        ax2.barh(genre_counts.index[::-1], genre_counts.values[::-1], color=colors[::-1])
        ax2.set_xlabel('Number of Movies')
        ax2.set_title('Movies per Genre')
        ax2.tick_params(axis='y', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col4:
        st.markdown("### 📋 Movies per Genre")
        genre_table = genre_counts.reset_index()
        genre_table.columns = ['Genre', 'Movie Count']
        genre_table['% of Catalogue'] = (
            genre_table['Movie Count'] / len(movies_df) * 100
        ).map('{:.1f}%'.format)
        st.dataframe(genre_table, width='stretch', hide_index=True)

    st.markdown("---")

    # ── Row 3: Genre pie chart + Most popular movies ─────────────────────────
    st.markdown("### 🔥 Top 10 Most Popular Movies")
    top_pop = movies_df.nlargest(10, 'popularity')[
        ['title', 'popularity', 'vote_average', 'vote_count']
    ].copy()
    top_pop['popularity'] = top_pop['popularity'].map('{:.1f}'.format)
    st.dataframe(top_pop, width='stretch', hide_index=True)

    st.markdown("---")

    # ── Row 4: Vote count distribution ───────────────────────────────────────
    st.markdown("### 👥 Vote Count Distribution (log scale)")
    fig4, ax4 = plt.subplots(figsize=(10, 3))
    ax4.hist(movies_df['vote_count'], bins=50, color='#764ba2', edgecolor='black', log=True)
    ax4.set_xlabel('Vote Count')
    ax4.set_ylabel('Frequency (log)')
    ax4.set_title('Distribution of Vote Counts')
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("---")

    # ── Browse all ───────────────────────────────────────────────────────────
    st.markdown("### 🔍 Browse All Movies")
    st.dataframe(
        movies_df[['title', 'genres_list', 'vote_average', 'vote_count', 'popularity']].head(100),
        width='stretch', hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 - How It Works
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## ℹ️ How It Works")
    st.markdown("Each algorithm follows the same pipeline: **Input → Feature Extraction → Scoring → Output**.")
    st.markdown("---")

    # ── Uniform model cards ──────────────────────────────────────────────────
    models = [
        {
            "icon": "📄", "name": "Content-Based Filtering", "badge": "badge-content",
            "description": "Recommends movies whose metadata (overview, genres, cast, keywords, director) is most similar to the selected movie.",
            "input":    "Movie title",
            "features": "TF-IDF vectors (5 000 features) built from a combined text 'soup' (overview + genres + cast + keywords + director)",
            "scoring":  "Cosine similarity between the query vector and all other movie vectors (precomputed N×N matrix)",
            "output":   "Top-N movies with the highest cosine similarity score",
            "formula":  "similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)",
            "note":     "The full similarity matrix is precomputed once at startup for instant lookup.",
        },
        {
            "icon": "🔵", "name": "KNN-Based Filtering", "badge": "badge-knn",
            "description": "Finds the K nearest neighbours of the selected movie in TF-IDF space using a brute-force cosine distance search.",
            "input":    "Movie title",
            "features": "TF-IDF vectors (3 000 features) — same feature space as Content-Based, with reduced vocabulary for speed",
            "scoring":  "Cosine distance via sklearn NearestNeighbors (brute-force, parallelised); similarity = 1 − distance",
            "output":   "Top-N closest neighbours sorted by descending similarity",
            "formula":  "similarity = 1 − cosine_distance(A, B)",
            "note":     "More memory-efficient than a full N×N matrix; model is cached after the first build.",
        },
        {
            "icon": "🤝", "name": "Collaborative Filtering (SVD)", "badge": "badge-collab",
            "description": "Simulates user-movie ratings and decomposes the matrix with SVD to surface movies preferred by users with similar taste.",
            "input":    "Movie title",
            "features": "Simulated 100-user × N-movie rating matrix built from genre preferences and vote averages (vectorised numpy ops)",
            "scoring":  "Reconstructed ratings from SVD (k=15 factors); movies ranked by average predicted rating across the most similar users",
            "output":   "Top-N movies with the highest predicted average rating, normalised to [0, 1]",
            "formula":  "A ≈ U · Σ · Vᵀ  (k = 15 latent factors)",
            "note":     "In a production system the rating matrix would use real watch/rating history instead of simulated data.",
        },
        {
            "icon": "🔀", "name": "Hybrid Approach", "badge": "badge-hybrid",
            "description": "Combines content similarity with the IMDB-style weighted rating to balance relevance and overall quality.",
            "input":    "Movie title",
            "features": "Content-Based cosine similarity scores + IMDB weighted rating (vote_average, vote_count)",
            "scoring":  "Linear combination: 70 % content similarity + 30 % normalised weighted rating",
            "output":   "Top-N movies sorted by descending hybrid score",
            "formula":  "hybrid = 0.7 × similarity + 0.3 × (WR / 10)",
            "note":     "WR = (v/(v+m))×R + (m/(v+m))×C  where m = 70th-pct vote count, C = global mean rating.",
        },
    ]

    for m in models:
        st.markdown(f"""
            <h4>{m['icon']} {m['name']} &nbsp;<span class="algo-badge {m['badge']}">{m['icon']} {m['name']}</span></h4>
            <p>{m['description']}</p>
            <table>
                <tr><th>Step</th><th>Detail</th></tr>
                <tr><td>🔣 Features</td><td>{m['features']}</td></tr>
                <tr><td>📐 Scoring</td><td>{m['scoring']}</td></tr>
                <tr><td>📤 Output</td><td>{m['output']}</td></tr>
                <tr><td>🧮 Formula</td><td><code>{m['formula']}</code></td></tr>
                <tr><td>💡 Note</td><td><em>{m['note']}</em></td></tr>
            </table>
            <br>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Metrics legend ───────────────────────────────────────────────────────
    st.markdown("### 📊 Recommendation Metrics")
    st.markdown("""
        | Metric | Description |
        |---|---|
        | ⭐ Rating | Average user rating from TMDB dataset (scale 0-10) |
        | 👥 Votes | Total number of user votes used to compute the rating |
        | 🎯 Similarity | Feature-based cosine similarity to the selected movie (0-100 %) |
        | 🤖 Model Score | Final score output by the selected algorithm (0-1) |
        """)

    st.markdown("---")
    st.markdown("### 🚀 Tech Stack")
    st.markdown("Python · Pandas · NumPy · Scikit-learn · SciPy · Streamlit · TMDB API")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;'>
    <p>Built using Python & Streamlit | Data Science Portfolio Project</p>
</div>
""", unsafe_allow_html=True)