import os
import pickle
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ============ SETUP ============
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_URL = "http://www.omdbapi.com/"

if not OMDB_API_KEY:
    raise RuntimeError("Missing OMDB_API_KEY in .env file")

app = FastAPI(title="Movie Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ LOAD MODEL FILES ============
BASE = os.path.dirname(os.path.abspath(__file__))
df = None
tfidf_matrix = None
title_to_index = {}

@app.on_event("startup")
def load_models():
    global df, tfidf_matrix, title_to_index

    with open(os.path.join(BASE, "models/df.pkl"), "rb") as f:
        df = pickle.load(f)
    with open(os.path.join(BASE, "models/tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(os.path.join(BASE, "models/indices.pkl"), "rb") as f:
        indices = pickle.load(f)

    # Build a simple map: movie title -> row number
    for title, idx in indices.items():
        title_to_index[str(title).strip().lower()] = int(idx)


# ============ DATA MODELS ============
class MovieCard(BaseModel):
    imdb_id: str
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None

class MovieDetails(BaseModel):
    imdb_id: str
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    genres: List[dict] = []

class RecItem(BaseModel):
    title: str
    score: float
    omdb: Optional[MovieCard] = None

class SearchBundle(BaseModel):
    query: str
    movie_details: MovieDetails
    tfidf_recommendations: List[RecItem]
    genre_recommendations: List[MovieCard]


# ============ OMDB HELPERS ============
async def omdb_call(params: Dict[str, Any]) -> Dict[str, Any]:
    """Call the OMDB API and return the JSON result."""
    params["apikey"] = OMDB_API_KEY
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(OMDB_URL, params=params)
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"OMDB request failed: {e}")


def clean_poster(poster):
    """Return None if poster is missing."""
    return poster if poster and poster != "N/A" else None


def make_cards(results, limit=20, skip_id=None):
    """Turn OMDB search results into a list of movie cards."""
    cards = []
    for m in (results or [])[:limit]:
        if skip_id and m.get("imdbID") == skip_id:
            continue
        cards.append(MovieCard(
            imdb_id=m.get("imdbID", ""),
            title=m.get("Title", ""),
            poster_url=clean_poster(m.get("Poster")),
            release_date=m.get("Year"),
        ))
    return cards


async def search_omdb(query, page=1):
    """Search movies on OMDB by keyword."""
    return await omdb_call({"s": query, "page": page, "type": "movie"})


async def get_details(imdb_id) -> MovieDetails:
    """Get full details of one movie by its IMDB id."""
    data = await omdb_call({"i": imdb_id, "plot": "full"})
    if data.get("Response") == "False":
        raise HTTPException(404, data.get("Error", "Movie not found"))

    genre_text = data.get("Genre", "")
    genres = [{"id": i, "name": g.strip()} for i, g in enumerate(genre_text.split(","))] if genre_text else []

    return MovieDetails(
        imdb_id=data.get("imdbID", imdb_id),
        title=data.get("Title", ""),
        overview=data.get("Plot"),
        release_date=data.get("Released"),
        poster_url=clean_poster(data.get("Poster")),
        genres=genres,
    )


# ============ TF-IDF RECOMMENDER ============
def recommend_by_tfidf(title, top_n=10):
    """Find similar movies using the TF-IDF matrix (cosine similarity)."""
    key = str(title).strip().lower()
    if key not in title_to_index:
        return []

    idx = title_to_index[key]
    scores = (tfidf_matrix @ tfidf_matrix[idx].T).toarray().ravel()
    best = np.argsort(-scores)

    results = []
    for i in best:
        if int(i) == idx:
            continue
        results.append((str(df.iloc[int(i)]["title"]), float(scores[int(i)])))
        if len(results) >= top_n:
            break
    return results


async def find_poster(title) -> Optional[MovieCard]:
    """Look up a poster for a movie title using OMDB."""
    data = await search_omdb(title)
    results = data.get("Search", [])
    if not results:
        return None
    m = results[0]
    return MovieCard(
        imdb_id=m.get("imdbID", ""),
        title=m.get("Title", title),
        poster_url=clean_poster(m.get("Poster")),
        release_date=m.get("Year"),
    )


# ============ ROUTES ============
@app.get("/")
def home_message():
    return {"message": "Movie Recommender API is running!"}


@app.get("/home", response_model=List[MovieCard])
async def home_feed(category: str = "popular", limit: int = 24):
    """Home page movies. OMDB has no categories, so we use preset searches."""
    presets = {
        "trending": "marvel",
        "popular": "star wars",
        "top_rated": "godfather",
        "now_playing": "spider-man",
        "upcoming": "batman",
    }
    data = await search_omdb(presets.get(category, "movie"))
    return make_cards(data.get("Search", []), limit)


@app.get("/omdb/search")
async def keyword_search(query: str = Query(..., min_length=1), page: int = 1):
    """Search movies by keyword (returns raw OMDB result)."""
    return await search_omdb(query, page)


@app.get("/movie/id/{imdb_id}", response_model=MovieDetails)
async def movie_details(imdb_id: str):
    """Get details of one movie."""
    return await get_details(imdb_id)


@app.get("/movie/search", response_model=SearchBundle)
async def search_bundle(query: str, tfidf_top_n: int = 12, genre_limit: int = 12):
    """Get movie details + similar movies (TF-IDF) + genre movies."""
    # Find the best matching movie
    data = await search_omdb(query)
    results = data.get("Search", [])
    if not results:
        raise HTTPException(404, f"No movie found for: {query}")

    details = await get_details(results[0]["imdbID"])

    # 1) TF-IDF similar movies
    tfidf_items = []
    for title, score in recommend_by_tfidf(details.title, tfidf_top_n):
        card = await find_poster(title)
        tfidf_items.append(RecItem(title=title, score=score, omdb=card))

    # 2) Genre based movies
    genre_recs = []
    if details.genres:
        genre_name = details.genres[0]["name"]
        gdata = await search_omdb(genre_name)
        genre_recs = make_cards(gdata.get("Search", []), genre_limit, skip_id=details.imdb_id)

    return SearchBundle(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )