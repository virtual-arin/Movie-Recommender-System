import os
import requests
import streamlit as st

# =============================
# CONFIG
# =============================
API_BASE = os.getenv("API_URL", "https://server-2ou7.onrender.com")
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# =============================
# STYLES (minimal modern)
# =============================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color:#6b7280; font-size: 0.92rem; }
.movie-title { font-size: 0.9rem; line-height: 1.15rem; height: 2.3rem; overflow: hidden; }
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.7); }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# STATE + ROUTING (single-file pages)
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"  # home | details
if "selected_imdb_id" not in st.session_state:
    st.session_state.selected_imdb_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    st.session_state.selected_imdb_id = qp_id
    st.session_state.view = "details"

def goto_home():
    st.session_state.view = "home"
    st.session_state.selected_imdb_id = None
    st.query_params.clear()
    st.rerun()


def goto_details(imdb_id: str):
    st.session_state.view = "details"
    st.session_state.selected_imdb_id = imdb_id
    st.query_params["view"] = "details"
    st.query_params["id"] = imdb_id
    st.rerun()


# API HELPERS
# =============================
@st.cache_data(ttl=30)
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1

            imdb_id = m.get("imdb_id")
            title = m.get("title", "Untitled")
            poster = m.get("poster_url")

            with colset[c]:
                if poster:
                    st.image(poster, width="stretch")
                else:
                    st.write("🖼️ No poster")

                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}_{imdb_id}"):
                   goto_details(imdb_id)

                st.markdown(
                    f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True
                )


def to_cards_from_tfidf_items(tfidf_items):
    cards = []
    for x in tfidf_items or []:
        omdb = x.get("omdb") or {}
        if omdb.get("imdb_id"):
            cards.append(
                {
                    "imdb_id": omdb["imdb_id"],
                    "title": omdb.get("title") or x.get("title") or "Untitled",
                    "poster_url": omdb.get("poster_url"),
                }
            )
    return cards


# =============================
# IMPORTANT: Robust OMDB search parsing
# Supports BOTH API shapes:
# 1) raw OMDB: {"Search":[{imdbID,Title,Poster,...}]}
# 2) list cards: [{imdb_id,title,poster_url,...}]
# =============================
def parse_omdb_search_to_cards(data, keyword: str, limit: int = 24):
    """
    Returns suggestions and cards.
    """
    keyword_l = keyword.strip().lower()

    # A) If API returns dict with 'Search'
    if isinstance(data, dict) and "Search" in data:
        raw = data.get("Search") or []
        raw_items = []
        for m in raw:
            title = (m.get("Title") or "").strip()
            imdb_id = m.get("imdbID")
            poster_url = m.get("Poster")
            if poster_url == "N/A":
                poster_url = None
            if not title or not imdb_id:
                continue
            raw_items.append(
                {
                    "imdb_id": imdb_id,
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": m.get("Year", ""),
          
                }
            )

    # B) If API returns already as list
    elif isinstance(data, list):
        raw_items = []
        for m in data:
            # might be {imdb_id,title,poster_url}
            imdb_id = m.get("imdb_id") or m.get("imdbID")
            title = (m.get("title") or m.get("Title") or "").strip()
            poster_url = m.get("poster_url") or m.get("Poster")
            if poster_url == "N/A":
                poster_url = None
            if not title or not imdb_id:
                continue
            raw_items.append(
                {
                    "imdb_id": imdb_id,
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": m.get("release_date") or m.get("Year") or "",
                }
            )
    else:
        return [], []

    # Word-match filtering (contains)
    matched = [x for x in raw_items if keyword_l in x["title"].lower()]

    # If nothing matches closely, fallback to raw items
    final_list = matched if matched else raw_items

    suggestions = []
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["imdb_id"]))

    # Cards = top N
    cards = [
        {"imdb_id": x["imdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
        for x in final_list[:limit]
    ]

    return suggestions, cards


# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Type keyword → dropdown suggestions + matching results → open → details + recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================================
# VIEW: HOME
# ==========================================================
if st.session_state.view == "home":
    typed = st.text_input(
        "Search by movie title (keyword)", placeholder="Type at least 2 characters..."
    )

    st.divider()

    # SEARCH MODE (Autocomplete + word-match results)
    if typed:
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters for suggestions.")
        else:
            data, err = api_get_json("/omdb/search", params={"query": typed.strip()})

            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_omdb_search_to_cards(
                    data, typed.strip(), limit=24
                )

                # Dropdown
                if suggestions:
                    labels = ["-- Select a movie --"] + [s[0] for s in suggestions]
                    selected = st.selectbox("Suggestions", labels, index=0)

                    if selected != "-- Select a movie --":
                        # map label -> id
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")

                st.markdown("### Results")
                poster_grid(cards, key_prefix="search_results")

        st.stop()

    # HOME FEED MODE
    st.markdown("### 🏠 Home Feed")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
        label_visibility="collapsed"
    )

    home_cards, err = api_get_json(
        "/home", params={"category": home_category, "limit": 24}
    )
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()

    poster_grid(home_cards, key_prefix="home_feed")

# ==========================================================
# VIEW: DETAILS
# ==========================================================
elif st.session_state.view == "details":
    imdb_id = st.session_state.selected_imdb_id
    if not imdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    # Top bar
    a, b = st.columns([3, 1])
    with a:
        st.write("")
    with b:
        if st.button("← Back to Home"):
            goto_home()

    # Details (your FastAPI safe route)
    data, err = api_get_json(f"/movie/id/{imdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    # Layout: Poster LEFT, Details RIGHT
    left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"], width="stretch")
        else:
            st.write("🖼️ No poster")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title','')}")
        release = data.get("release_date") or "-"
        genres = ", ".join([g["name"] for g in data.get("genres", [])]) or "-"
        st.markdown(
            f"<div class='small-muted'>Release: {release}</div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='small-muted'>Genres: {genres}</div>", unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)
        if data.get("backdrop_url"):
            st.markdown("#### Backdrop")
            st.image(data["backdrop_url"], width="stretch")

    st.divider()
    st.markdown("### ✅ Recommendations")

    # Recommendations (TF-IDF + Genre) via your bundle endpoint
    title = (data.get("title") or "").strip()
    if title:
        bundle, err2 = api_get_json(
            "/movie/search",
            params={"query": title, "tfidf_top_n": 12, "genre_limit": 12},
        )

        if not err2 and bundle:
            st.markdown("#### 🔎 Similar Movies (TF-IDF)")
            poster_grid(
                to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
                key_prefix="details_tfidf",
            )

            st.markdown("#### 🎭 More Like This (Genre)")
            poster_grid(
                bundle.get("genre_recommendations", []),
                key_prefix="details_genre",
            )
        else:
            st.info("Showing Genre recommendations (fallback).")
            genre_only, err3 = api_get_json(
                "/recommend/genre", params={"imdb_id": imdb_id, "limit": 18}
            )
            if not err3 and genre_only:
                poster_grid(
                    genre_only, key_prefix="details_genre_fallback"
                )
            else:
                st.warning("No recommendations available right now.")
    else:
        st.warning("No title available to compute recommendations.")