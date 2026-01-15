# MoodMatch — Streamlit Mood-Based Recommender (Movies + Books)

**MoodMatch** is a premium **Streamlit** web app that recommends **Movies** and **Books** based on your **mood**, **context** (Solo/Pareja/Familia/Amigos), and an exploration slider (**Safe → Wild**).  
Under the hood, it uses a **hybrid recommendation engine**: keyword intent scoring + **TF-IDF** similarity + personalization from your likes, plus a **diversity-aware Top-K** selection to avoid repetitive results.

---

##  What’s inside

### Core UX
- **Movies + Books tabs**
- **Surprise me** (random pick from top candidates)
- **Quick Picks** (Tinder-style Like / Skip / No)
- **Curated Rows** (horizontal rails with clickable posters/covers)
- **Details panel** with “Why this was picked”
- **My List** builder + copy/paste export

### Recommendation Engine (what’s interesting technically)
This project is not just UI — it includes real ranking + NLP signals:

1) **Text feature engineering**
- Movies: `title + genres + tags_text + overview` → `combined_text`
- Books: `title + authors + tags_text` → `combined_text`

2) **Intent scoring via keyword rules**
- Mood and context provide **include/boost** and **exclude/penalty** keyword sets.
- A fast scoring function counts matches and applies penalties.

3) **TF-IDF content similarity (NLP)**
- Builds TF-IDF matrices for movies and books using:
  - `max_features=40000`, `ngram_range=(1,2)`, `min_df=2`
- Computes:
  - `tfidf_query`: similarity to a mood+context query
  - `tfidf_like`: similarity to a **user profile vector** (mean TF-IDF of liked items, normalized)

4) **Exploration vs. popularity control**
- The **Discovery (Safe → Wild)** slider dynamically changes weights:
  - “Safe” → more popular items
  - “Wild” → more exploratory / intent-driven items

5) **Diversity-aware Top-K**
- A greedy selection penalizes overlap in tokens to avoid near-duplicates
  (MMR-like diversification).

---

## Repo contents

**Main app**
- `app.py` — Streamlit UI + recommendation engine

**Utilities**
- `inspect_categories.py` — helper script to inspect dataset categories/tags (optional)

**Datasets**
- `data/movies_tags_text.csv` *(required)*
- `data/books_tags_text.csv` *(required)*

---

##  Requirements

Create a `requirements.txt` like this:

```txt
streamlit>=1.30
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
requests>=2.31
