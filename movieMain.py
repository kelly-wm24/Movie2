import streamlit as st
import pandas as pd
import ast
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

# --- 1. DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    """Original data cleaning logic."""
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    ratings = pd.read_csv("ratings_small.csv")
    links = pd.read_csv("links_small.csv")

    movies = movies[['id', 'title', 'overview', 'genres', 'release_date', 'vote_average', 'vote_count']]
    movies = movies.dropna(subset=['id', 'title', 'overview'])
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id']).astype({'id': int})

    def extract_genres(genre_str):
        try:
            genre_list = ast.literal_eval(genre_str)
            return " ".join([g['name'] for g in genre_list])
        except: return ""

    movies['genres_clean'] = movies['genres'].apply(extract_genres)
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce').dropna().astype(int)
    
    merged = pd.merge(links, movies, left_on='tmdbId', right_on='id', how='inner')
    merged = merged.drop_duplicates(subset='title').reset_index(drop=True)
    merged['combined_features'] = merged['overview'].fillna('') + " " + (merged['genres_clean'].fillna('') + " ") * 3
    
    return merged, ratings

@st.cache_resource
def compute_similarity(movies_merged, ratings):
    """Similarity matrix calculations."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_merged['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    ratings_movies = pd.merge(ratings, movies_merged, on='movieId', how='inner')
    user_movie_matrix = ratings_movies.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    movie_similarity = cosine_similarity(user_movie_matrix.T)
    movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
    
    indices = pd.Series(movies_merged.index, index=movies_merged['title']).drop_duplicates()
    return cosine_sim, movie_similarity_df, indices

movies_merged, ratings_data = load_data()
cosine_sim, movie_similarity_df, indices = compute_similarity(movies_merged, ratings_data)

# --- 2. ADVANCED CORE ENGINES ---

def recommend_content(title, top_n=10):
    if title not in indices: return pd.DataFrame()
    idx = indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+30]
    recs = pd.DataFrame({'title': [movies_merged.iloc[i[0]]['title'] for i in sim_scores], 'model_score': [i[1] for i in sim_scores]})
    return pd.merge(recs, movies_merged, on='title').query('vote_count > 50')

def recommend_collaborative(title, top_n=10):
    if title not in movie_similarity_df.columns: return pd.DataFrame()
    sim_scores = movie_similarity_df[title].sort_values(ascending=False)[1:top_n+30]
    recs = pd.DataFrame({'title': sim_scores.index, 'model_score': sim_scores.values})
    return pd.merge(recs, movies_merged, on='title').query('vote_count > 50')

def hybrid_recommend(movie_title, top_n=10, alpha=0.5):
    if movie_title not in indices or movie_title not in movie_similarity_df.columns: return pd.DataFrame()
    c_recs = recommend_content(movie_title, 100).rename(columns={'model_score': 'c_score'})
    col_recs = recommend_collaborative(movie_title, 100).rename(columns={'model_score': 'col_score'})
    hybrid = pd.merge(c_recs[['title', 'c_score']], col_recs[['title', 'col_score']], on='title', how='outer').fillna(0)
    hybrid['model_score'] = (alpha * (hybrid['c_score']/hybrid['c_score'].max())) + ((1-alpha) * (hybrid['col_score']/hybrid['col_score'].max()))
    return pd.merge(hybrid, movies_merged, on='title').sort_values('model_score', ascending=False)

# --- 3. UI HELPER FUNCTIONS ---

def apply_filters(df, mood, personality, hidden_gems):
    """Mood and Personality mapping."""
    if mood != "None":
        mood_map = {"Happy": ["Comedy", "Family"], "Sad": ["Drama"], "Romantic": ["Romance"], "Excited": ["Action"], "Curious": ["Mystery"], "Scared": ["Horror"], "Relaxed": ["Music"]}
        pattern = "|".join(mood_map.get(mood, []))
        df = df[df['genres_clean'].str.contains(pattern, case=False, na=False)]
    if personality != "None":
        pers_map = {"Adventurer": ["Adventure"], "Romantic": ["Romance"], "Thinker": ["Science Fiction"], "Fun Lover": ["Comedy"], "Dreamer": ["Fantasy"]}
        pattern = "|".join(pers_map.get(personality, []))
        df = df[df['genres_clean'].str.contains(pattern, case=False, na=False)]
    if hidden_gems:
        df = df[(df['vote_average'] >= 6.5) & (df['vote_count'] < 500)]
    return df

# --- 4. STREAMLIT APP ---

st.set_page_config(page_title="Advanced Movie Expert System", layout="wide")
st.sidebar.title("🎬 Discovery Menu")
page = st.sidebar.radio("Select Interface", ["Smart Recommendations", "Advanced Search", "Model Deep-Dive", "System Features"])

if page == "Smart Recommendations":
    st.header("🎯 Personalized Movie Matches")
    mode = st.selectbox("Recommender Engine", ["Hybrid", "Content-Based", "Collaborative", "Explore", "New User"])
    
    movie_title = None
    if mode != "New User":
        # Searchable selection
        movie_title = st.selectbox("Pick a movie you love:", movies_merged['title'].unique())

    c1, c2, c3 = st.columns(3)
    mood = c1.selectbox("Your Current Vibe", ["None", "Happy", "Sad", "Romantic", "Excited", "Curious", "Scared", "Relaxed"])
    pers = c2.selectbox("Your Personality", ["None", "Adventurer", "Romantic", "Thinker", "Fun Lover", "Dreamer"])
    top_n = c3.slider("Results per page", 1, 15, 5)
    
    hidden_gems = st.checkbox("Focus on Hidden Gems (Underrated movies)")

    if st.button("Generate Experience"):
        if mode == "Content-Based": res = recommend_content(movie_title, 30)
        elif mode == "Collaborative": res = recommend_collaborative(movie_title, 30)
        elif mode == "New User":
            res = movies_merged.copy().sort_values(by=['vote_average', 'vote_count'], ascending=False).query('vote_count > 100')
        else: res = hybrid_recommend(movie_title, 30)

        res = apply_filters(res, mood, pers, hidden_gems)

        if not res.empty:
            st.success(f"Algorithm applied: {mode}")
            for _, row in res.head(top_n).iterrows():
                with st.expander(f"🎥 {row['title']} (Rating: {row['vote_average']})"):
                    st.write(f"**Genres:** {row['genres_clean']} | **Votes:** {row['vote_count']}")
                    st.write(f"**Overview:** {row['overview']}")
        else: st.error("The filters are too strict. Try changing your mood or personality.")

elif page == "Advanced Search":
    st.header("🔍 Semantic Search & Discovery")
    query = st.text_input("Search titles or themes (Typos allowed!):")
    if query:
        # Fuzzy logic to find matches
        all_titles = movies_merged['title'].tolist()
        fuzzy_matches = process.extract(query, all_titles, limit=20, score_cutoff=60)
        
        if fuzzy_matches:
            found_titles = [m[0] for m in fuzzy_matches]
            search_results = movies_merged[movies_merged['title'].isin(found_titles)]
            
            st.write(f"Found {len(search_results)} relevant matches.")
            
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.dataframe(search_results[['title', 'genres_clean', 'vote_average', 'release_date']])
            with col_right:
                st.write("**Rating Spread**")
                st.bar_chart(search_results['vote_average'])
        else: st.warning("No matches found. Try another keyword.")

elif page == "Model Deep-Dive":
    st.header("📊 Deep-Dive Model Comparison")
    comp_movie = st.selectbox("Select Movie to Analyze", movies_merged['title'].unique())
    
    if st.button("Run Model Comparison"):
        c_recs = recommend_content(comp_movie, 10).assign(Model='Content')
        cl_recs = recommend_collaborative(comp_movie, 10).assign(Model='Collaborative')
        h_recs = hybrid_recommend(comp_movie, 10).assign(Model='Hybrid')
        
        # Visualize rating distribution across models
        all_results = pd.concat([c_recs, cl_recs, h_recs])
        fig = px.box(all_results, x="Model", y="vote_average", color="Model",
                     title="Movie Ratings Distribution by Model Selection")
        st.plotly_chart(fig, use_container_width=True)
        
        # Side-by-Side View
        col1, col2, col3 = st.columns(3)
        col1.write("**1. Content-Based**"); col1.table(c_recs[['title']].head(5))
        col2.write("**2. Collaborative**"); col2.table(cl_recs[['title']].head(5))
        col3.write("**3. Hybrid (AI Choice)**"); col3.table(h_recs[['title']].head(5))
        
        # Intersection Analysis
        overlap = set(c_recs['title']) & set(cl_recs['title'])
        if overlap:
            st.success(f"Models agree on these High-Confidence matches: {', '.join(overlap)}")
        else:
            st.info("Models are diverse! Hybrid provides unique results not found in single methods.")

elif page == "System Features":
    st.header("✨ Comprehensive System Features")
    features = {
        "Hybrid Recommendation": "Combines Content (TF-IDF) & Collaborative Filtering.",
        "Explainable UI": "Uses expanders and success messages to tell users why movies were picked.",
        "Deep-Dive Comparison": "Statistical analysis of model performance using Plotly charts.",
        "Fuzzy Search": "Handles typos and partial strings for better user discovery.",
        "Hidden Gems Logic": "Filters for high-quality movies that haven't been 'mainstreamed'.",
        "Mood/Personality Matrix": "Context-aware filtering based on user psychographics.",
        "Caching System": "Uses @st.cache to ensure university-level performance standards."
    }
    for feat, desc in features.items():
        st.write(f"🔹 **{feat}**: {desc}")
