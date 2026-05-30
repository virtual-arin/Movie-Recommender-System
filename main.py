import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

OMDB_BASE = "http://www.omdbapi.com/"

if not OMDB_API_KEY:
    # Don't crash import-time in production if you prefer; but for you better fail early:
    raise RuntimeError("OMDB_API_KEY missing. Put it in .env as OMDB_API_KEY=xxxx")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class OMDBMovieCard(BaseModel):
    imdb_id: str
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class OMDBMovieDetails(BaseModel):
    imdb_id: str
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    omdb: Optional[OMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: OMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[OMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


async def omdb_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe OMDB GET:
    - Network errors -> 502
    - OMDB API errors -> 502 with detail
    """
    q = dict(params)
    q["apikey"] = OMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(OMDB_BASE, params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"OMDB error {r.status_code}: {r.text}"
        )

    return r.json()


async def omdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[OMDBMovieCard]:
    out: List[OMDBMovieCard] = []
    for m in (results or [])[:limit]:
        poster = m.get("Poster")
        out.append(
            OMDBMovieCard(
                imdb_id=m.get("imdbID", ""),
                title=m.get("Title") or "",
                poster_url=poster if poster and poster != "N/A" else None,
                release_date=m.get("Year"),
                vote_average=None,
            )
        )
    return out


async def omdb_movie_details(imdb_id: str) -> OMDBMovieDetails:
    data = await omdb_get({"i": imdb_id, "plot": "full"})
    if data.get("Response") == "False":
        raise HTTPException(status_code=404, detail=data.get("Error"))
        
    poster = data.get("Poster")
    genres = [{"id": i, "name": g.strip()} for i, g in enumerate(data.get("Genre", "").split(","))] if data.get("Genre") else []

    return OMDBMovieDetails(
        imdb_id=data.get("imdbID", imdb_id),
        title=data.get("Title") or "",
        overview=data.get("Plot"),
        release_date=data.get("Released"),
        poster_url=poster if poster and poster != "N/A" else None,
        backdrop_url=None,
        genres=genres,
    )


async def omdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw OMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """
    return await omdb_get({"s": query, "page": page, "type": "movie"})


async def omdb_search_first(query: str) -> Optional[dict]:
    data = await omdb_search_movies(query=query, page=1)
    results = data.get("Search", [])
    return results[0] if results else None


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    # query vector
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # sort descending
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_omdb_card_by_title(title: str) -> Optional[OMDBMovieCard]:
    """
    Uses OMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await omdb_search_first(title)
        if not m:
            return None
        poster = m.get("Poster")
        return OMDBMovieCard(
            imdb_id=m.get("imdbID", ""),
            title=m.get("Title") or title,
            poster_url=poster if poster and poster != "N/A" else None,
            release_date=m.get("Year"),
            vote_average=None,
        )
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    # Load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF matrix (usually scipy sparse)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load tfidf vectorizer (optional, not used directly here)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


# =========================
# ROUTES
# =========================
@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Recommender API!"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (OMDB) ----------
@app.get("/home", response_model=List[OMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed for Streamlit (posters).
    OMDB doesn't have native categories, so we fake it via preset queries.
    """
    try:
        queries = {
            "trending": "marvel",
            "popular": "star wars",
            "top_rated": "godfather",
            "upcoming": "batman",
            "now_playing": "spider-man"
        }
        q = queries.get(category, "movie")
        data = await omdb_search_movies(q, page=1)
        return await omdb_cards_from_results(data.get("Search", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- OMDB KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/omdb/search")
async def omdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns RAW OMDB shape with 'Search' list.
    Streamlit will use it for:
      - dropdown suggestions
      - grid results
    """
    return await omdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS (SAFE ROUTE) ----------
@app.get("/movie/id/{imdb_id}", response_model=OMDBMovieDetails)
async def movie_details_route(imdb_id: str):
    return await omdb_movie_details(imdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[OMDBMovieCard])
async def recommend_genre(
    imdb_id: str = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Given an OMDB movie ID:
    - fetch details
    - pick first genre
    - discover movies in that genre (popular)
    """
    details = await omdb_movie_details(imdb_id)
    if not details.genres:
        return []

    genre_name = details.genres[0]["name"]
    data = await omdb_search_movies(query=genre_name, page=1)
    cards = await omdb_cards_from_results(data.get("Search", []), limit=limit)
    return [c for c in cards if c.imdb_id != imdb_id]


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details
      - TF-IDF recommendations (local) + posters
      - Genre recommendations (OMDB) + posters

    NOTE:
    - It selects the BEST match from OMDB for the given query.
    - If you want MULTIPLE matches, use /omdb/search
    """
    best = await omdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No OMDB movie found for query: {query}"
        )

    imdb_id = best["imdbID"]
    details = await omdb_movie_details(imdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by OMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_omdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, omdb=card))

    # 2) Genre recommendations (OMDB fake discover by first genre)
    genre_recs: List[OMDBMovieCard] = []
    if details.genres:
        genre_name = details.genres[0]["name"]
        data = await omdb_search_movies(query=genre_name, page=1)
        cards = await omdb_cards_from_results(
            data.get("Search", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.imdb_id != details.imdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )