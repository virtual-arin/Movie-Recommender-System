# 🎬 Movie Recommendation System

## 🎭 Business Domain
Entertainment & Streaming

## 🤔 Problem Statement
Finding relevant movies from a large collection can be time-consuming. A content-based recommendation system helps users discover similar movies based on their interests.

## 🎯 Project Objective
Build a movie recommendation system that suggests similar movies using content-based filtering and cosine similarity, while displaying movie details and posters through the OMDb API.

## 📊 Dataset Overview
The dataset contains movie metadata such as titles, genres, and descriptions used to generate recommendations.

**Goal:** Recommend similar movies based on user-selected input.

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Streamlit
- OMDb API

## 📂 Project Structure
```
├── models/
│   ├── df.pkl
│   ├── tfidf_matrix.pkl
|   ├── tfidf.pkl
│   └── indices.pkl
├── data/
│   └── movies_data.csv
├── notebook/
│   └── Notebook.ipynb
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## 🔄 Workflow

### 1. Data Preprocessing
- Cleaned and processed movie metadata.
- Generated TF-IDF feature vectors.

### 2. Recommendation Engine
- Computed cosine similarity between movies.
- Retrieved top similar movie recommendations.

### 3. API Integration
- Built FastAPI backend for recommendations.
- Fetched movie posters and metadata using the OMDb API.

### 4. Deployment
- Developed an interactive Streamlit interface.
- Connected frontend with FastAPI backend.

## 📈 Features
- 🎬 Content-based movie recommendations
- 🔍 Movie search functionality
- 🖼️ Fetch movie posters via OMDb API
- ⚡ FastAPI-powered backend
- 💻 Interactive Streamlit UI

## 🚀 How to Run

```bash
git clone https://github.com/virtual-arin/Movie-Recommender-System

cd Movie-Recommender-System

pip install -r requirements.txt

uvicorn main:app --reload

streamlit run app.py
```

## 🔑 API Setup

Get your OMDb API key:

http://www.omdbapi.com/apikey.aspx

Create a `.env` file:

```env
OMDB_API_KEY=YOUR_API_KEY
```

## 🚀 Business Impact
- Improves movie discovery experience.
- Provides personalized recommendations.
- Reduces search effort for users.
- Demonstrates practical NLP-based recommendation systems.

## 📬 Contact

**Arin Sharma**  
📧 arinsharma.infinity@gmail.com