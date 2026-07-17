import os
import requests
import streamlit as st

# ============ CONFIG ============
API = "https://recommender-server-4y66.onrender.com" or "http://127.0.0.1:8000"
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ============ SIMPLE STYLES ============
st.markdown("""
<style>
.block-container { max-width: 1300px; padding-top: 2rem; }

/* Make every poster the SAME height */
.movie-poster img {
    height: 330px !important;
    width: 100% !important;
    object-fit: cover; 
    border-radius: 10px;
}

/* Make every title box the SAME height (2 lines) */
.movie-title {
    font-size: 0.9rem;
    line-height: 1.2rem;
    height: 2.4rem;     
    overflow: hidden;
    margin: 8px 0;
    text-align: center;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ============ HELPER: Call the API ============
def get_api(path, params=None):
    """Send a GET request to our backend and return the JSON."""
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        else:
            # Show the real error so we can fix it
            st.error(f"⚠️ API returned {r.status_code}: {r.text[:300]}")
    except Exception as e:
        st.error(f"⚠️ Could not connect to backend. Is it running? Error: {e}")
    return None


# ============ HELPER: Show movies in a grid ============
def show_movies(movies, key):
    """Display a list of movies as poster cards with an Open button."""
    if not movies:
        st.info("No movies to show.")
        return

    cols = st.columns(5)
    for i, m in enumerate(movies):
        imdb_id = m.get("imdb_id") or m.get("imdbID")
        title = m.get("title") or m.get("Title") or "Untitled"
        poster = m.get("poster_url") or m.get("Poster")
        if poster == "N/A":
            poster = None

        with cols[i % 5]:
            # Poster (same height for all)
            if poster:
                st.markdown(
                    f"<div class='movie-poster'><img src='{poster}'></div>",
                    unsafe_allow_html=True,
                )
            else:
                # Empty grey box so cards without posters stay aligned
                st.markdown(
                    "<div class='movie-poster'><img src='https://via.placeholder.com/300x450?text=No+Poster'></div>",
                    unsafe_allow_html=True,
                )

            # Title (same height for all)
            st.markdown(f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True)

            # Open button
            st.button("Open", key=f"{key}_{i}_{imdb_id}",
                      on_click=open_movie, args=(imdb_id,), use_container_width=True)


# ============ PAGE SWITCHING ============
if "movie_id" not in st.session_state:
    st.session_state.movie_id = None

def open_movie(imdb_id):
    st.session_state.movie_id = imdb_id
    st.rerun()

def go_home():
    st.session_state.movie_id = None
    st.rerun()


# ============ HEADER ============
st.title("🎬 Movie Recommender")
st.caption("Search a movie, open it, and get similar recommendations.")
st.divider()


# ==================================================
# PAGE 1: HOME (search + home feed)
# ==================================================
if st.session_state.movie_id is None:

    typed = st.text_input("🔍 Search a movie", placeholder="Type a movie name...")

    # --- If user is searching ---
    if typed and len(typed.strip()) >= 2:
        data = get_api("/omdb/search", {"query": typed.strip()})
        results = data.get("Search", []) if data else []
        st.subheader("Search Results")
        show_movies(results, key="search")

    # --- If not searching, show home feed ---
    else:
        category = st.selectbox(
            "Category",
            ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        )
        movies = get_api("/home", {"category": category, "limit": 20})
        st.subheader("🏠 Home Feed")
        show_movies(movies, key="home")


# ==================================================
# PAGE 2: MOVIE DETAILS + RECOMMENDATIONS
# ==================================================
else:
    imdb_id = st.session_state.movie_id

    if st.button("← Back to Home"):
        go_home()

    data = get_api(f"/movie/id/{imdb_id}")
    if not data:
        st.stop()

    left, right = st.columns([1, 2])

    with left:
        if data.get("poster_url"):
            st.image(data["poster_url"], use_container_width=True)
        else:
            st.write("🖼️ No poster")

    with right:
        st.header(data.get("title", ""))
        genres = ", ".join(g["name"] for g in data.get("genres", [])) or "-"
        st.write(f"**Release:** {data.get('release_date', '-')}")
        st.write(f"**Genres:** {genres}")
        st.subheader("Overview")
        st.write(data.get("overview") or "No overview available.")

    st.divider()

    st.subheader("✅ Recommendations")
    title = data.get("title", "").strip()
    bundle = get_api("/movie/search", {"query": title, "tfidf_top_n": 10, "genre_limit": 10})

    if bundle:
        tfidf = [x["omdb"] for x in bundle.get("tfidf_recommendations", []) if x.get("omdb")]
        st.markdown("#### 🔎 Similar Movies")
        show_movies(tfidf, key="tfidf")

        st.markdown("#### 🎭 More Like This (Genre)")
        show_movies(bundle.get("genre_recommendations", []), key="genre")
    else:
        st.info("No recommendations available.")