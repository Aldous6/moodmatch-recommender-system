import streamlit as st
import pandas as pd
import requests
import random
import json
import numpy as np
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# 1. CONFIGURACIÓN E INTERFAZ
# =============================================================================
st.set_page_config(page_title="MoodMatch", layout="wide", initial_sidebar_state="collapsed")

PREMIUM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Manrope:wght@600;700;800&display=swap');

:root{
  --mm-bg: #07080C;
  --mm-bg-2: #0B0D12;
  --mm-panel: rgba(255,255,255,0.045);
  --mm-border: rgba(255,255,255,0.10);
  --mm-text: rgba(255,255,255,0.92);
  --mm-muted: rgba(255,255,255,0.70);
  --mm-muted-2: rgba(255,255,255,0.52);
  --mm-accent: #5B7CFF;
  --mm-radius: 18px;
  --mm-radius-lg: 26px;
  --mm-shadow: 0 28px 90px rgba(0,0,0,0.62);
  --mm-shadow-soft: 0 12px 44px rgba(0,0,0,0.42);
  --mm-blur: 22px;
}

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 520px at 12% -10%, rgba(91,124,255,0.22) 0%, transparent 62%),
    radial-gradient(900px 520px at 88% -12%, rgba(45,226,230,0.14) 0%, transparent 56%),
    radial-gradient(800px 500px at 55% 110%, rgba(255,255,255,0.06) 0%, transparent 60%),
    linear-gradient(180deg, var(--mm-bg) 0%, var(--mm-bg-2) 100%) !important;
  color: var(--mm-text) !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
}

/* Oculta menú principal y footer */
#MainMenu, footer { visibility: hidden; }

/* ✅ ARREGLO 1: Se eliminó la línea que ocultaba el stToolbar para no perder el botón */

/* HEADER = glass */
[data-testid="stHeader"]{
  background: rgba(7,8,12,0.28) !important;
  backdrop-filter: blur(16px) saturate(150%);
  -webkit-backdrop-filter: blur(16px) saturate(150%);
  border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* ✅ ARREGLO 2: Botón para re-abrir sidebar (Soporte doble selector) */
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
  position: fixed !important;
  top: 12px !important;
  left: 12px !important;
  z-index: 100000 !important;

  display: grid !important;
  place-items: center !important;

  width: 44px !important;
  height: 44px !important;
  border-radius: 999px !important;

  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  box-shadow: 0 12px 35px rgba(0,0,0,0.35) !important;

  visibility: visible !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}

/* Main container spacing */
section.main > div.block-container{ padding-top: 2.2rem; max-width: 1240px; }

h1, h2, h3, h4{ font-family: Manrope, Inter, sans-serif !important; letter-spacing: -0.02em; }
h1{ font-size: 2.35rem !important; }
p, li{ color: var(--mm-text) !important; }
small, .stCaption{ color: var(--mm-muted-2) !important; }

/* Sidebar glass */
[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.028) !important;
  border-right: 1px solid var(--mm-border) !important;
  backdrop-filter: blur(18px) saturate(140%);
  -webkit-backdrop-filter: blur(18px) saturate(140%);
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label{ color: var(--mm-muted) !important; }

/* Inputs */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
div[data-baseweb="slider"]{
  background: rgba(255,255,255,0.035) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
}
label{ color: var(--mm-muted) !important; }

/* Buttons */
div.stButton > button{
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.05) !important;
  color: var(--mm-text) !important;
  transition: transform .12s ease;
}
div.stButton > button:hover{
  transform: translateY(-1px);
  background: rgba(255,255,255,0.07) !important;
}
div.stButton > button[kind="primary"]{
  background: linear-gradient(180deg, rgba(91,124,255,0.98) 0%, rgba(91,124,255,0.78) 100%) !important;
  border-color: rgba(91,124,255,0.75) !important;
}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.055) 0%, rgba(255,255,255,0.025) 100%) !important;
  border: 1px solid var(--mm-border) !important;
  border-radius: var(--mm-radius-lg) !important;
  box-shadow: var(--mm-shadow-soft) !important;
}

[data-testid="stImage"] img{
  border-radius: 18px !important;
  box-shadow: 0 18px 60px rgba(0,0,0,0.50) !important;
}

hr{
  border: none !important;
  height: 1px !important;
  background: rgba(255,255,255,0.10) !important;
  margin: 18px 0 !important;
}

/* Hero */
.kicker{ color: var(--mm-muted-2); text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.76rem; font-weight: 700; }
.small-muted{ color: var(--mm-muted); font-size: 0.98rem; }
.hero{
  padding: 26px 26px;
  border-radius: var(--mm-radius-lg);
  border: 1px solid rgba(255,255,255,0.12);
  background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
  box-shadow: var(--mm-shadow);
  margin-bottom: 2.0rem;
}
.hero .hero-top{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.hero .pill{ padding: 7px 10px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.04); font-size: 0.82rem; }

/* Rails */
.mm-section-head{ display:flex; align-items:flex-end; justify-content:space-between; gap: 12px; margin: 0.25rem 0 0.2rem 0; }
.mm-section-title{ font-family: Manrope, sans-serif !important; letter-spacing:-0.02em; font-size: 1.15rem; margin: 0; }
.mm-section-sub{ color: var(--mm-muted-2); font-size: 0.92rem; margin: 0; }

.mm-rail{ display:flex; gap: 14px; overflow-x:auto; padding: 10px 6px 18px 6px; scroll-snap-type: x mandatory; }
.mm-rail::-webkit-scrollbar{ height: 8px; }
.mm-rail::-webkit-scrollbar-thumb{ background: rgba(255,255,255,0.15); border-radius: 999px; }

.mm-card{ flex: 0 0 auto; width: 172px; scroll-snap-align: start; text-decoration:none !important; color: var(--mm-text) !important; }
.mm-poster{
  width: 172px; aspect-ratio: 2 / 3;
  border-radius: 18px; overflow: hidden;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  position: relative; transition: transform .14s ease;
}
.mm-poster img{ width:100%; height:100%; object-fit: cover; display:block; }
.mm-poster::after{ content:""; position:absolute; inset:0; background: linear-gradient(180deg, transparent 55%, rgba(0,0,0,0.8) 100%); }
.mm-card:hover .mm-poster{ transform: translateY(-3px) scale(1.02); border-color: rgba(255,255,255,0.25); }
.mm-meta{ position:absolute; left: 10px; right: 10px; bottom: 10px; z-index: 2; }
.mm-title{ font-weight: 800; font-size: 0.9rem; line-height: 1.1; margin-bottom: 2px; text-shadow: 0 2px 10px rgba(0,0,0,0.8); }
.mm-subline{ color: rgba(255,255,255,0.7); font-size: 0.75rem; white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }

/* Details */
.mm-details{
  border-radius: var(--mm-radius-lg);
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.03);
  padding: 20px; margin: 10px 0 20px 0;
}
.mm-reason{ border-left: 2px solid var(--mm-accent); padding-left: 10px; margin: 6px 0; color: var(--mm-muted); font-size: 0.9rem; }
.mm-chip{ padding: 6px 12px; background: rgba(255,255,255,0.05); border-radius: 99px; font-size: 0.8rem; color: var(--mm-muted); }

/* MOBILE RESPONSIVE */
@media (max-width: 900px){
  section.main > div.block-container{
    padding: 1.1rem 0.9rem 2.2rem 0.9rem;
    max-width: 100%;
  }
  h1{ font-size: 1.9rem !important; }
  .hero{ padding: 18px 18px; margin-bottom: 1.2rem; }
  .mm-card{ width: 140px; }
  .mm-poster{ width: 140px; border-radius: 16px; }
}
"""

# ✅ ARREGLO 3: Inyección única del CSS
st.markdown(f"<style>{PREMIUM_CSS}</style>", unsafe_allow_html=True)

# Lógica del Backdrop (Mantenida del original)
st.markdown("""
<style>
/* Backdrop */
#mm-backdrop{
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.42);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  opacity: 0;
  pointer-events: none;
  transition: opacity .15s ease;
  z-index: 99990;
}

/* Cuando el sidebar está abierto en móvil, activamos backdrop + blur del main */
@media (max-width: 900px){
  html.mm-sidebar-open #mm-backdrop{ opacity: 1; pointer-events: auto; }
  html.mm-sidebar-open section.main{ filter: blur(6px); }
}
</style>

<div id="mm-backdrop"></div>

<script>
(() => {
  const root = document.documentElement;
  const backdrop = document.getElementById("mm-backdrop");

  const isSidebarOpen = () => {
    const sb = document.querySelector('[data-testid="stSidebar"]');
    if (!sb) return false;

    // Streamlit suele manejar aria-expanded en algunos builds.
    const aria = sb.getAttribute("aria-expanded");
    if (aria === "true") return true;
    if (aria === "false") return false;

    // Fallback: en móvil, si está visible y ocupa ancho, asumimos abierto
    const st = window.getComputedStyle(sb);
    const w = parseFloat(st.width || "0");
    const visible = st.display !== "none" && st.visibility !== "hidden";
    const mobile = window.matchMedia("(max-width: 900px)").matches;
    return mobile && visible && w > 80;
  };

  const update = () => {
    if (isSidebarOpen()) root.classList.add("mm-sidebar-open");
    else root.classList.remove("mm-sidebar-open");
  };

  // Observa cambios en el DOM (Streamlit re-render)
  new MutationObserver(update).observe(document.body, { childList: true, subtree: true, attributes: true });
  window.addEventListener("resize", update);
  backdrop?.addEventListener("click", () => root.classList.remove("mm-sidebar-open"));

  setTimeout(update, 0);
})();
</script>
""", unsafe_allow_html=True)


# ⭐ EXTRA: Botón flotante independiente (FAB)
# Esto asegura que si Streamlit cambia los IDs de nuevo, tú sigues teniendo un botón funcional.
st.markdown("""
<style>
#mm-fab{
  position: fixed;
  top: 12px;
  left: 12px;
  z-index: 100001;
  width: 44px;
  height: 44px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.9);
  backdrop-filter: blur(14px) saturate(140%);
  -webkit-backdrop-filter: blur(14px) saturate(140%);
  box-shadow: 0 12px 35px rgba(0,0,0,0.35);
  cursor: pointer;
  display: grid;
  place-items: center;
  font-size: 1.2rem;
  transition: transform 0.1s ease;
}
#mm-fab:hover{
  transform: scale(1.05);
  background: rgba(255,255,255,0.1);
}
</style>

<button id="mm-fab" title="Menu">☰</button>

<script>
(() => {
  const clickToggle = () => {
    // Intenta encontrar el botón nativo de Streamlit (ambas versiones)
    const btn =
      document.querySelector('[data-testid="stSidebarCollapsedControl"] button')
      || document.querySelector('[data-testid="collapsedControl"] button')
      || document.querySelector('[data-testid="stSidebarCollapsedControl"]'); // A veces el click va al div contenedor

    if (btn) {
        btn.click();
    } else {
        console.warn("MoodMatch: No se encontró el botón nativo del sidebar.");
    }
  };
  
  const fab = document.getElementById("mm-fab");
  if(fab) fab.addEventListener("click", clickToggle);
})();
</script>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================
CATEGORY_MAP = {
    "Todo": {"mov": [], "book": []},
    "Ciencia Ficción y Fantasía": {
        "mov": ["Sci-Fi", "Fantasy"],
        "book": ["science-fiction", "sci-fi", "fantasy", "magic", "dystopian", "space", "scifi", "sci-fi-fantasy", "supernatural"]
    },
    "Acción y Aventura": {
        "mov": ["Action", "Adventure", "War", "Western"],
        "book": ["adventure", "action", "survival", "war", "western"]
    },
    "Crimen y Misterio": {
        "mov": ["Crime", "Mystery", "Thriller", "Film-Noir"],
        "book": ["mystery", "crime", "thriller", "suspense", "mystery-thriller", "detective"]
    },
    "Drama y Emoción": {
        "mov": ["Drama", "Romance"],
        "book": ["drama", "romance", "contemporary", "historical-fiction", "historical", "love", "realistic-fiction"]
    },
    "Terror y Oscuro": {
        "mov": ["Horror"],
        "book": ["horror", "paranormal", "vampires", "gothic", "ghosts"]
    },
    "Comedia y Ligero": {
        "mov": ["Comedy", "Animation", "Family", "Musical"],
        "book": ["humor", "funny", "comedy", "chick-lit"]
    },
    "No Ficción y Aprendizaje": {
        "mov": ["Documentary"],
        "book": ["non-fiction", "history", "biography", "memoir", "philosophy", "psychology", "science", "self-help", "business"]
    }
}

MOOD_MOVIES = {
    "Relajado / feel-good": {"include": ["comedy", "feel-good", "heartwarming", "uplifting", "family", "friendship", "light"], "exclude": ["horror", "gore", "slasher"]},
    "Emocional / profundo": {"include": ["drama", "life", "loss", "grief", "coming of age", "relationship", "bittersweet"], "exclude": ["parody"]},
    "Intenso / tensión": {"include": ["thriller", "suspense", "crime", "mystery", "serial", "investigation", "psychological"], "exclude": ["family", "animation"]},
    "Inspirador / motivación": {"include": ["inspirational", "biography", "based on a true story", "sports", "history", "overcoming", "hope"], "exclude": []},
    "Cerebral / mind-blowing": {"include": ["science fiction", "mind-bending", "twist", "time travel", "dream", "philosophy", "existential", "surreal"], "exclude": []},
    "Terror / oscuro": {"include": ["horror", "haunted", "paranormal", "demonic", "occult", "gore", "slasher"], "exclude": ["family", "kids"]},
}

MOOD_BOOKS = {
    "Relajado / feel-good": {"include": ["feel-good", "humor", "friendship", "cozy", "uplifting", "light-hearted"], "exclude": ["dark", "horror", "erotica"]},
    "Emocional / profundo": {"include": ["drama", "heartbreaking", "family", "love", "loss", "grief", "literary fiction"], "exclude": []},
    "Intenso / tensión": {"include": ["thriller", "mystery", "suspense", "crime", "psychological", "detective"], "exclude": ["cozy"]},
    "Inspirador / motivación": {"include": ["self-help", "inspirational", "success", "productivity", "biography", "motivation"], "exclude": []},
    "Cerebral / mind-blowing": {"include": ["philosophy", "psychology", "science", "history", "ideas", "politics", "non-fiction"], "exclude": []},
    "Terror / oscuro": {"include": ["horror", "dark", "paranormal", "gothic", "creepy", "occult"], "exclude": ["children", "young-adult"]},
}

CONTEXT_MOVIES = {
    "Solo": {"boost": ["mind-bending", "psychological", "mystery", "documentary", "noir", "indie", "art"], "penalty": []},
    "Pareja": {"boost": ["romance", "relationship", "date", "love", "rom-com"], "penalty": ["gore", "slasher"]},
    "Familia": {"boost": ["family", "animation", "adventure", "kids", "pixar", "disney", "funny"], "penalty": ["horror", "gore", "slasher", "nudity", "erotic"]},
    "Amigos": {"boost": ["action", "comedy", "thriller", "adventure", "superhero", "sci-fi"], "penalty": ["slow", "arthouse"]},
}

CONTEXT_BOOKS = {
    "Solo": {"boost": ["philosophy", "psychology", "science", "history", "non-fiction", "self-help"], "penalty": []},
    "Pareja": {"boost": ["romance", "love", "relationships", "contemporary", "poetry"], "penalty": ["gore", "splatter"]},
    "Familia": {"boost": ["young-adult", "children", "fantasy", "adventure", "friendship", "humor"], "penalty": ["erotica", "explicit", "horror", "dark"]},
    "Amigos": {"boost": ["mystery", "thriller", "fantasy", "science fiction", "popular"], "penalty": []},
}

BOOK_BLOCKLIST_FAMILY = ["erotica", "explicit", "sex", "adult", "gore", "splatter", "rape"]

# =============================================================================
# 3. HELPERS
# =============================================================================
@st.cache_data
def load_data_optimized():
    mov = pd.read_csv("data/movies_tags_text.csv")
    bok = pd.read_csv("data/books_tags_text.csv")
    
    m_tags = mov.get("tags_text", pd.Series("", index=mov.index))
    m_ov = mov.get("overview", pd.Series("", index=mov.index))
    mov["combined_text"] = (
        mov["title"].fillna("").astype(str) + " " +
        mov["genres"].fillna("").astype(str) + " " +
        m_tags.fillna("").astype(str) + " " +
        m_ov.fillna("").astype(str)
    ).str.lower()
    
    b_tags = bok.get("tags_text", pd.Series("", index=bok.index))
    bok["combined_text"] = (
        bok["title"].fillna("").astype(str) + " " +
        bok["authors"].fillna("").astype(str) + " " +
        b_tags.fillna("").astype(str)
    ).str.lower()
    
    return mov, bok

@st.cache_data(show_spinner=False)
def tmdb_get_config(read_token: str):
    url = "https://api.themoviedb.org/3/configuration"
    headers = {"Authorization": f"Bearer {read_token}", "accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            images = r.json().get("images", {})
            base = images.get("secure_base_url") or images.get("base_url") or "https://image.tmdb.org/t/p/"
            sizes = images.get("poster_sizes", []) or ["w500"]
            size = "w500" if "w500" in sizes else sizes[-1]
            return base, size
    except: pass
    return "https://image.tmdb.org/t/p/", "w500"

@st.cache_data(show_spinner=False)
def tmdb_fetch_poster_url(tmdb_id: int, read_token: str, base: str, size: str):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    headers = {"Authorization": f"Bearer {read_token}", "accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=3)
        if r.status_code == 200:
            poster_path = (r.json() or {}).get("poster_path")
            if isinstance(poster_path, str) and poster_path.startswith("/"):
                return f"{base}{size}{poster_path}"
    except: pass
    return None

def banner(text: str):
    st.markdown(f"<div style='padding:12px 14px;border:1px solid var(--mm-border);border-radius:16px;background:rgba(255,255,255,0.03);color:var(--mm-muted);margin-bottom:1rem;'>{text}</div>", unsafe_allow_html=True)

def _qp_get(key: str):
    try:
        v = st.query_params.get(key)
        if isinstance(v, list): return v[0] if v else None
        return v
    except: return None

def _qp_clear():
    st.query_params.clear()

def _render_movie_rail(df_rows: pd.DataFrame):
    cards = []
    tmdb_token_local = st.secrets.get("TMDB_READ_TOKEN", "")
    for _, r in df_rows.iterrows():
        mid = int(r.get("movieId"))
        title = html.escape(str(r.get("title","")))
        genres = html.escape(str(r.get("genres","")).replace("|"," · "))
        poster_url = None
        tmdb_id = r.get("tmdbId")
        if pd.notna(tmdb_id) and tmdb_token_local:
            try:
                key = int(float(tmdb_id))
                poster_url = st.session_state.poster_cache.get(key)
            except: pass
        
        img_html = f"<img src='{poster_url}'/>" if poster_url else ""
        # IMPORTANT: No indentation in HTML string to prevent markdown code blocks
        cards.append(f"""<a class="mm-card" href="?pick_movie={mid}"><div class="mm-poster">{img_html}<div class="mm-meta"><div class="mm-title">{title}</div><div class="mm-subline">{genres}</div></div></div></a>""")
    return f"<div class='mm-rail'>{''.join(cards)}</div>"

def _render_book_rail(df_rows: pd.DataFrame):
    cards = []
    for _, r in df_rows.iterrows():
        bid = int(r.get("book_id"))
        title = html.escape(str(r.get("title","")))
        authors = html.escape(str(r.get("authors","")))
        cover = r.get("image_url")
        img_html = f"<img src='{cover}'/>" if pd.notna(cover) else ""
        cards.append(f"""<a class="mm-card" href="?pick_book={bid}"><div class="mm-poster">{img_html}<div class="mm-meta"><div class="mm-title">{title}</div><div class="mm-subline">{authors}</div></div></div></a>""")
    return f"<div class='mm-rail'>{''.join(cards)}</div>"

def _movie_reasons(row: pd.Series):
    reasons = []
    blob = (str(row.get("genres","")) + " " + str(row.get("tags_text","")) + " " + str(row.get("overview",""))).lower()
    mood_hits = [t for t in MOOD_MOVIES[mood]["include"] if t in blob][:3]
    if mood_hits: reasons.append("Mood match: " + ", ".join(mood_hits))
    if "like_boost" in row and float(row.get("like_boost", 0)) > 0: reasons.append("Aligned with recent likes")
    return reasons[:4]

def _book_reasons(row: pd.Series):
    reasons = []
    ttext = str(row.get("tags_text","")).lower()
    mood_hits = [t for t in MOOD_BOOKS[mood]["include"] if t in ttext][:3]
    if mood_hits: reasons.append("Mood match: " + ", ".join(mood_hits))
    if "like_boost" in row and float(row.get("like_boost", 0)) > 0: reasons.append("Similar to your likes")
    return reasons[:4]

# =============================================================================
# 4. LOGICA DE RECOMENDACION
# =============================================================================
@st.cache_resource(show_spinner=False)
def build_tfidf_index(df: pd.DataFrame, text_col: str):
    vec = TfidfVectorizer(max_features=40000, min_df=2, ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(df[text_col])
    return vec, X

def add_tfidf_signals(d: pd.DataFrame, vec: TfidfVectorizer, X, query_text: str, liked_idx):
    indices_to_keep = d.index
    Xd = X[indices_to_keep]
    q = vec.transform([query_text.lower()])
    d["tfidf_query"] = (Xd @ q.T).toarray().ravel()
    if liked_idx and len(liked_idx) > 0:
        try:
            user_mean = np.asarray(X[liked_idx].mean(axis=0)).reshape(1, -1)
            profile = normalize(user_mean)
            sim_l = Xd @ profile.T
            if hasattr(sim_l, "toarray"): sim_l = sim_l.toarray()
            d["tfidf_like"] = sim_l.ravel()
        except: d["tfidf_like"] = 0.0
    else: d["tfidf_like"] = 0.0
    return d

def _norm(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return s / (s.max() + 1e-9)

def _kw_score(text: pd.Series, include, exclude=None):
    score = pd.Series(0.0, index=text.index)
    for kw in include:
        score += text.str.contains(kw, na=False, regex=False).astype(float)
    if exclude:
        for kw in exclude:
            score -= 0.8 * text.str.contains(kw, na=False, regex=False).astype(float)
    return score

def diversify_topk(df: pd.DataFrame, k: int, text_col: str, lambda_penalty: float = 0.35):
    df = df.copy()
    if len(df) == 0: return df
    df["_base"] = df["score"].astype(float)
    used = set()
    picked = []
    pool = df.head(300).copy()
    for _ in range(min(k, len(pool))):
        best_idx, best_val = None, None
        for idx, row in pool.iterrows():
            text = str(row.get(text_col, "")).lower().replace("|", " ")
            tokens = [t for t in text.split() if len(t) >= 4]
            overlap = sum((t in used) for t in tokens)
            val = float(row["_base"]) - lambda_penalty * overlap
            if (best_val is None) or (val > best_val):
                best_val, best_idx = val, idx
        if best_idx is None: break
        chosen = pool.loc[best_idx]
        picked.append(chosen)
        text = str(chosen.get(text_col, "")).lower().replace("|", " ")
        for t in [t for t in text.split() if len(t) >= 4]:
            used.add(t)
        pool = pool.drop(index=best_idx)
    return pd.DataFrame(picked).drop(columns=["_base"], errors="ignore")

movies, books = load_data_optimized()
MOV_VEC, MOV_X = build_tfidf_index(movies, "combined_text")
BOK_VEC, BOK_X = build_tfidf_index(books, "combined_text")

# =============================================================================
# 5. SIDEBAR
# =============================================================================
DEFAULT_FILTERS = {"category": list(CATEGORY_MAP.keys())[0], "mood": list(MOOD_MOVIES.keys())[0], "company": "Solo", "family_friendly": False, "max_runtime": 120, "risk": 30}
if "filters" not in st.session_state: st.session_state.filters = DEFAULT_FILTERS.copy()

st.sidebar.markdown("<div class='kicker'>MoodMatch</div>", unsafe_allow_html=True)
st.sidebar.markdown("### Preferences")
with st.sidebar.form("filters_form", border=False):
    category = st.selectbox("Category", list(CATEGORY_MAP.keys()), index=list(CATEGORY_MAP.keys()).index(st.session_state.filters.get("category", "Todo")))
    mood = st.selectbox("Vibe", list(MOOD_MOVIES.keys()), index=list(MOOD_MOVIES.keys()).index(st.session_state.filters["mood"]))
    company = st.selectbox("Context", ["Solo", "Pareja", "Familia", "Amigos"], index=["Solo","Pareja","Familia","Amigos"].index(st.session_state.filters["company"]))
    default_family = True if company == "Familia" else st.session_state.filters["family_friendly"]
    family_friendly = st.checkbox("Family friendly", value=default_family)
    max_runtime = st.slider("Max runtime (min)", 20, 240, int(st.session_state.filters["max_runtime"]), step=10)
    risk = st.slider("Discovery (Safe → Wild)", 0, 100, int(st.session_state.filters["risk"]))
    applied = st.form_submit_button("Apply", type="primary", use_container_width=True)

if applied:
    st.session_state.filters = {"category": category, "mood": mood, "company": company, "family_friendly": family_friendly, "max_runtime": max_runtime, "risk": risk}

category = st.session_state.filters.get("category", "Todo")
mood = st.session_state.filters["mood"]
company = st.session_state.filters["company"]
family_friendly = st.session_state.filters["family_friendly"]
max_runtime = st.session_state.filters["max_runtime"]
risk = st.session_state.filters["risk"]

st.sidebar.divider()
st.sidebar.caption(f"Likes: {len(st.session_state.get('likes',set()))} Mov · {len(st.session_state.get('book_likes',set()))} Bok")
if st.sidebar.button("Reset Session", use_container_width=True):
    for k in ["likes", "dislikes", "tinder_idx", "book_likes", "book_dislikes", "book_tinder_idx", "watchlist", "poster_cache", "seed"]:
        if k in st.session_state: del st.session_state[k]
    st.rerun()

# =============================================================================
# 6. FILTRADO
# =============================================================================
def recommend_movies_strict(df: pd.DataFrame):
    d = df.copy()
    valid_genres = CATEGORY_MAP[category]["mov"]
    if valid_genres:
        pattern = "|".join(valid_genres)
        d = d[d["genres"].astype(str).str.contains(pattern, case=False, regex=True)]
    if len(d) == 0: return d 
    if family_friendly:
        d = d[d["adult"].astype(str).str.lower().isin(["false", "0", "nan", "none", ""])]
    d = d[pd.to_numeric(d["runtime"], errors="coerce").fillna(9999) <= max_runtime]
    
    text = d["combined_text"]
    mood_prof = MOOD_MOVIES[mood]
    ctx_prof = CONTEXT_MOVIES[company]
    d["mood_score"] = _kw_score(text, mood_prof["include"], mood_prof.get("exclude"))
    d = d[d["mood_score"] > 0]
    if len(d) == 0: return d

    d["ctx_score"] = _kw_score(text, ctx_prof["boost"], ctx_prof.get("penalty"))
    d["pop"] = pd.to_numeric(d["rating_count"], errors="coerce").fillna(0)
    w_pop = 0.35 - (risk / 100.0) * 0.20
    w_intent = 1.0 - w_pop
    d["score"] = (w_intent * 0.65) * _norm(d["mood_score"]) + (w_intent * 0.35) * _norm(d["ctx_score"]) + w_pop * _norm(d["pop"])
    
    query_text = " ".join(mood_prof["include"] + ctx_prof["boost"])
    likes = st.session_state.get("likes", set())
    liked_idx = df[df["movieId"].isin(list(likes))].index.tolist()
    d = add_tfidf_signals(d, MOV_VEC, MOV_X, query_text, liked_idx)
    
    alpha = 0.15 + 0.15 * (risk / 100.0)
    beta = 0.25 - 0.10 * (risk / 100.0)
    d["score"] = d["score"] + alpha * _norm(d["tfidf_query"]) + beta * _norm(d["tfidf_like"])
    
    dislikes = st.session_state.get("dislikes", set())
    if len(dislikes) > 0: d = d[~d["movieId"].isin(list(dislikes))]
    if len(likes) > 0: d["like_boost"] = d["tfidf_like"]
    return d.sort_values("score", ascending=False)

def recommend_books_strict(df: pd.DataFrame):
    d = df.copy()
    valid_tags = CATEGORY_MAP[category]["book"]
    if valid_tags:
        pattern = "|".join(valid_tags)
        d = d[d.get("tags_text", pd.Series("", index=d.index)).astype(str).str.contains(pattern, case=False, regex=True)]
    if len(d) == 0: return d
    if family_friendly:
        mask_bad = pd.Series(False, index=d.index)
        for kw in BOOK_BLOCKLIST_FAMILY:
            mask_bad |= d["combined_text"].str.contains(kw, na=False, regex=False)
        d = d[~mask_bad]
        
    text = d["combined_text"]
    mood_prof = MOOD_BOOKS[mood]
    ctx_prof = CONTEXT_BOOKS[company]
    d["mood_score"] = _kw_score(text, mood_prof["include"], mood_prof.get("exclude"))
    d = d[d["mood_score"] > 0]
    if len(d) == 0: return d

    d["ctx_score"] = _kw_score(text, ctx_prof["boost"], ctx_prof.get("penalty"))
    d["pop"] = pd.to_numeric(d["ratings_count"], errors="coerce").fillna(0)
    w_pop = 0.35 - (risk / 100.0) * 0.20
    w_intent = 1.0 - w_pop
    d["score"] = (w_intent * 0.65) * _norm(d["mood_score"]) + (w_intent * 0.35) * _norm(d["ctx_score"]) + w_pop * _norm(d["pop"])
    
    query_text = " ".join(mood_prof["include"] + ctx_prof["boost"])
    likes = st.session_state.get("book_likes", set())
    liked_idx = df[df["book_id"].isin(list(likes))].index.tolist()
    d = add_tfidf_signals(d, BOK_VEC, BOK_X, query_text, liked_idx)
    
    alpha = 0.15 + 0.15 * (risk / 100.0)
    beta = 0.25 - 0.10 * (risk / 100.0)
    d["score"] = d["score"] + alpha * _norm(d["tfidf_query"]) + beta * _norm(d["tfidf_like"])
    
    dislikes = st.session_state.get("book_dislikes", set())
    if len(dislikes) > 0: d = d[~d["book_id"].isin(list(dislikes))]
    if len(likes) > 0: d["like_boost"] = d["tfidf_like"]
    return d.sort_values("score", ascending=False)

# Ejecución
all_ranked_movies = recommend_movies_strict(movies)
top10_movies = diversify_topk(all_ranked_movies, k=10, text_col="tags_text", lambda_penalty=0.35)
movies_pool = all_ranked_movies.head(100).reset_index(drop=True)
surprise_pick = movies_pool.sample(1).iloc[0] if len(movies_pool) > 0 else None

all_ranked_books = recommend_books_strict(books)
top10_books = diversify_topk(all_ranked_books, k=10, text_col="tags_text", lambda_penalty=0.35)
books_pool = all_ranked_books.head(100).reset_index(drop=True)
surprise_book = books_pool.sample(1).iloc[0] if len(books_pool) > 0 else None

# =============================================================================
# 7. UI PRINCIPAL
# =============================================================================
st.markdown("""<div class="hero"><div class="hero-top"><div class="kicker">Recommendation Engine</div><div class="pill">Strict Curation</div></div><h1 style="margin:0.45rem 0 0.35rem 0;">MoodMatch</h1><div class="small-muted">High-signal picks curated by Category and Mood. No noise, just matches.</div></div>""", unsafe_allow_html=True)

tmdb_token = st.secrets.get("TMDB_READ_TOKEN", "")
if "poster_cache" not in st.session_state: st.session_state.poster_cache = {}
if "seed" not in st.session_state: st.session_state.seed = random.randint(0, 10000)

tab1, tab2 = st.tabs(["Movies", "Books"])

# --- MOVIES TAB ---
with tab1:
    base, size = "https://image.tmdb.org/t/p/", "w500"
    
    # 1. Surprise Section
    st.markdown("### Surprise me")
    with st.container(border=True):
        if st.button("Generate one pick", key="surprise_movie_btn"):
            if surprise_pick is not None:
                banner(f"Your pick: **{surprise_pick['title']}**")
                cols_s = st.columns([1, 4], gap="large")
                with cols_s[0]:
                    poster_url = None
                    try:
                        tmdb_id = surprise_pick.get("tmdbId")
                        if pd.notna(tmdb_id) and tmdb_token:
                            key = int(float(tmdb_id))
                            poster_url = st.session_state.poster_cache.get(key) or tmdb_fetch_poster_url(key, tmdb_token, base, size)
                            if poster_url: st.session_state.poster_cache[key] = poster_url
                    except: pass
                    if poster_url: st.image(poster_url, use_container_width=True)
                    else: st.caption("No Poster")
                with cols_s[1]:
                    st.write(f"**Genres:** {surprise_pick.get('genres','')}")
                    st.write(f"**Rating:** {float(surprise_pick.get('rating_mean',0)):.2f}")
                    st.caption(str(surprise_pick.get("overview", ""))[:300] + "...")
            else: st.info("No movies match these exact criteria.")
    
    st.divider()

    # 2. Quick Picks Section (Tinder)
    st.markdown("### Quick Picks")
    if "likes" not in st.session_state: st.session_state.likes = set()
    if "dislikes" not in st.session_state: st.session_state.dislikes = set()
    if "tinder_idx" not in st.session_state: st.session_state.tinder_idx = 0

    pool_tinder = movies_pool.sample(frac=1, random_state=st.session_state.seed).reset_index(drop=True)

    with st.container(border=True):
        if len(pool_tinder) == 0:
            st.info("No candidates left matching criteria.")
        else:
            idx_actual = st.session_state.tinder_idx % len(pool_tinder)
            pick = pool_tinder.iloc[idx_actual]
            t_col1, t_col2 = st.columns([1, 2], gap="large")
            with t_col1:
                poster_url = None
                tmdb_id = pick.get("tmdbId")
                if pd.notna(tmdb_id) and tmdb_token:
                    key = int(float(tmdb_id))
                    poster_url = st.session_state.poster_cache.get(key) or tmdb_fetch_poster_url(key, tmdb_token, base, size)
                    if poster_url: st.session_state.poster_cache[key] = poster_url
                if poster_url: st.image(poster_url, use_container_width=True)
                else: st.caption("No Poster")
            with t_col2:
                st.markdown(f"#### {pick.get('title','')}")
                st.caption(f"{pick.get('genres','')}")
                st.write(str(pick.get("overview", ""))[:260] + "...")
                
                b1, b2, b3 = st.columns(3)
                if b1.button("Like", key="like_btn", type="primary", use_container_width=True):
                    st.session_state.likes.add(int(pick["movieId"]))
                    st.session_state.tinder_idx += 1
                    st.rerun()
                if b2.button("Skip", key="meh_btn", use_container_width=True):
                    st.session_state.tinder_idx += 1
                    st.rerun()
                if b3.button("No", key="dislike_btn", use_container_width=True):
                    st.session_state.dislikes.add(int(pick["movieId"]))
                    st.session_state.tinder_idx += 1
                    st.rerun()

    st.divider()

    # 3. RAILS SECTION (CALCULATION + PREFETCH + RENDER)
    st.markdown("### Curated Rows")
    if "watchlist" not in st.session_state: st.session_state.watchlist = []

    # --- A) CALCULATE ALL DATAFRAMES FIRST ---
    shown_ids = set(top10_movies["movieId"].astype(int).tolist()) if len(top10_movies) else set()
    likes_set = st.session_state.get("likes", set())
    dislikes_set = st.session_state.get("dislikes", set())

    # Calc: Because you liked
    because_like = pd.DataFrame()
    if len(likes_set) > 0 and len(all_ranked_movies) > 0:
        tmp = all_ranked_movies.copy()
        if "tfidf_like" not in tmp.columns: tmp["tfidf_like"] = 0.0
        tmp = tmp[~tmp["movieId"].astype(int).isin(list(likes_set))]
        if len(dislikes_set) > 0:
            tmp = tmp[~tmp["movieId"].astype(int).isin(list(dislikes_set))]
        tmp = tmp[~tmp["movieId"].astype(int).isin(list(shown_ids))]
        because_like = tmp.sort_values(["tfidf_like", "score"], ascending=False).head(18)
        shown_ids |= set(because_like["movieId"].astype(int).tolist())

    # Calc: More for your vibe
    more_vibe = pd.DataFrame()
    if len(all_ranked_movies) > 0:
        tmp2 = all_ranked_movies.copy()
        if len(dislikes_set) > 0:
            tmp2 = tmp2[~tmp2["movieId"].astype(int).isin(list(dislikes_set))]
        tmp2 = tmp2[~tmp2["movieId"].astype(int).isin(list(shown_ids))]
        more_vibe = diversify_topk(tmp2, k=18, text_col="tags_text", lambda_penalty=0.32)
        shown_ids |= set(more_vibe["movieId"].astype(int).tolist()) if len(more_vibe) else set()

    # --- B) PREFETCH IMAGES FOR ALL 3 RAILS (CRITICAL FIX) ---
    if tmdb_token:
        # Combinamos todo lo que vamos a mostrar
        to_fetch = pd.concat([top10_movies, because_like, more_vibe], axis=0, ignore_index=True)
        ids_to_load = []
        for x in to_fetch.get("tmdbId", pd.Series([], dtype="float")).dropna().tolist():
            try: ids_to_load.append(int(float(x)))
            except: pass
        
        # Descargamos hasta 100 imágenes (suficiente para los 3 rieles)
        try:
            base, size = tmdb_get_config(tmdb_token)
        except: pass

        for key in ids_to_load[:100]: 
            if key not in st.session_state.poster_cache:
                st.session_state.poster_cache[key] = tmdb_fetch_poster_url(key, tmdb_token, base, size)

    # --- C) DETAILS PANEL (CLICK HANDLER) ---
    picked_movie = _qp_get("pick_movie")
    if picked_movie:
        try: picked_movie = int(picked_movie)
        except: picked_movie = None

    if picked_movie and len(all_ranked_movies) > 0:
        sel = all_ranked_movies[all_ranked_movies["movieId"].astype(int) == int(picked_movie)]
        if len(sel) > 0:
            row = sel.iloc[0]
            poster_url = None
            tmdb_id = row.get("tmdbId")
            if pd.notna(tmdb_id) and tmdb_token:
                try:
                    key = int(float(tmdb_id))
                    poster_url = st.session_state.poster_cache.get(key) or tmdb_fetch_poster_url(key, tmdb_token, base, size)
                    if poster_url: st.session_state.poster_cache[key] = poster_url
                except: pass
            
            st.markdown("<div class='mm-details'>", unsafe_allow_html=True)
            d1, d2 = st.columns([1, 2], gap="large")
            with d1:
                if poster_url: st.image(poster_url, use_container_width=True)
                else: st.caption("Poster unavailable")
            with d2:
                st.markdown(f"### {row.get('title','')}")
                st.caption(str(row.get("genres","")).replace("|"," · "))
                st.write(str(row.get("overview","") or "").strip())
                st.markdown("#### Why this was picked")
                for r in _movie_reasons(row):
                    st.markdown(f"<div class='mm-reason'>{html.escape(r)}</div>", unsafe_allow_html=True)
                
                a1, a2, a3 = st.columns([1,1,1])
                if a1.button("Add to List", key=f"add_m_{int(row['movieId'])}", type="primary", use_container_width=True):
                    item = {"type": "movie", "title": row.get("title")}
                    if item not in st.session_state.watchlist:
                        st.session_state.watchlist.append(item)
                        banner("Added to list")
                if a2.button("Close", key="cls_m", use_container_width=True):
                    _qp_clear()
                    st.rerun()
                a3.markdown("<span class='mm-chip'>Click posters to explore</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- D) RENDER RAILS ---
    st.markdown("<div class='mm-section-head'><p class='mm-section-title'>Top curated</p><p class='mm-section-sub'>High-signal picks for your filters</p></div>", unsafe_allow_html=True)
    st.markdown(_render_movie_rail(top10_movies), unsafe_allow_html=True)

    if len(because_like) > 0:
        st.markdown("<div class='mm-section-head'><p class='mm-section-title'>Because you liked</p><p class='mm-section-sub'>Matches to your recent likes</p></div>", unsafe_allow_html=True)
        st.markdown(_render_movie_rail(because_like), unsafe_allow_html=True)

    if len(more_vibe) > 0:
        st.markdown("<div class='mm-section-head'><p class='mm-section-title'>More for your vibe</p><p class='mm-section-sub'>Explore deeper cuts</p></div>", unsafe_allow_html=True)
        st.markdown(_render_movie_rail(more_vibe), unsafe_allow_html=True)

with tab2:
    st.markdown("### Surprise me")
    with st.container(border=True):
        if st.button("Generate one pick", key="surprise_book_btn"):
            if surprise_book is not None:
                banner(f"Your pick: **{surprise_book.get('title','')}**")
                c1, c2 = st.columns([1, 4], gap="large")
                with c1:
                    if pd.notna(surprise_book.get("image_url")): st.image(surprise_book["image_url"], use_container_width=True)
                with c2:
                    st.write(f"**Author:** {surprise_book.get('authors','')}")
                    st.caption("Tags: " + " · ".join(str(surprise_book.get("tags_text", "")).split(" | ")[:8]))
            else: st.info("No books match these exact criteria.")

    st.divider()
    st.markdown("### Quick Picks")
    if "book_likes" not in st.session_state: st.session_state.book_likes = set()
    if "book_dislikes" not in st.session_state: st.session_state.book_dislikes = set()
    if "book_tinder_idx" not in st.session_state: st.session_state.book_tinder_idx = 0
    pool_b = books_pool.sample(frac=1, random_state=st.session_state.seed).reset_index(drop=True)
    with st.container(border=True):
        if len(pool_b) == 0: st.info("No candidates.")
        else:
            idx_actual_b = st.session_state.book_tinder_idx % len(pool_b)
            pickb = pool_b.iloc[idx_actual_b]
            bcol1, bcol2 = st.columns([1, 2], gap="large")
            with bcol1:
                if pd.notna(pickb.get("image_url")): st.image(pickb["image_url"], use_container_width=True)
                else: st.caption("No Cover")
            with bcol2:
                st.markdown(f"#### {pickb.get('title','')}")
                st.caption(f"by {pickb.get('authors','')}")
                st.write("**Tags:** " + " · ".join(str(pickb.get("tags_text", "")).split(" | ")[:10]))
                bb1, bb2, bb3 = st.columns(3)
                if bb1.button("Like", key="lb_btn", type="primary", use_container_width=True):
                    st.session_state.book_likes.add(int(pickb["book_id"]))
                    st.session_state.book_tinder_idx += 1
                    st.rerun()
                if bb2.button("Skip", key="sb_btn", use_container_width=True):
                    st.session_state.book_tinder_idx += 1
                    st.rerun()
                if bb3.button("No", key="db_btn", use_container_width=True):
                    st.session_state.book_dislikes.add(int(pickb["book_id"]))
                    st.session_state.book_tinder_idx += 1
                    st.rerun()

    st.divider()
    st.markdown("### Curated Rows")
    
    picked_book = _qp_get("pick_book")
    if picked_book:
        try: picked_book = int(picked_book)
        except: picked_book = None
    
    if picked_book and len(all_ranked_books) > 0:
        selb = all_ranked_books[all_ranked_books["book_id"].astype(int) == int(picked_book)]
        if len(selb) > 0:
            row = selb.iloc[0]
            st.markdown("<div class='mm-details'>", unsafe_allow_html=True)
            d1, d2 = st.columns([1, 2], gap="large")
            with d1:
                if pd.notna(row.get("image_url")): st.image(row["image_url"], use_container_width=True)
                else: st.caption("Cover unavailable")
            with d2:
                st.markdown(f"### {row.get('title','')}")
                st.caption(f"{row.get('authors','')}")
                st.markdown("#### Why this was picked")
                for r in _book_reasons(row):
                    st.markdown(f"<div class='mm-reason'>{html.escape(r)}</div>", unsafe_allow_html=True)
                b1, b2, b3 = st.columns([1,1,1])
                if b1.button("Add to List", key=f"add_b_{int(row['book_id'])}", type="primary", use_container_width=True):
                    item = {"type": "book", "title": row.get("title","")}
                    if item not in st.session_state.watchlist:
                        st.session_state.watchlist.append(item)
                        banner("Added to list")
                if b2.button("Close", key="cls_b", use_container_width=True):
                    _qp_clear()
                    st.rerun()
                b3.markdown("<span class='mm-chip'>Click covers to explore</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # 3 RAILS BOOKS
    shown_b = set(top10_books["book_id"].astype(int).tolist()) if len(top10_books) else set()
    likes_b = st.session_state.get("book_likes", set())

    st.markdown("<div class='mm-section-head'><p class='mm-section-title'>Top curated</p><p class='mm-section-sub'>High-signal picks for your filters</p></div>", unsafe_allow_html=True)
    st.markdown(_render_book_rail(top10_books), unsafe_allow_html=True)

    if len(likes_b) > 0 and len(all_ranked_books) > 0:
        tmp = all_ranked_books[~all_ranked_books["book_id"].astype(int).isin(list(shown_b))]
        tmp = tmp[~tmp["book_id"].astype(int).isin(list(likes_b))]
        because_like_b = tmp.sort_values(["tfidf_like", "score"], ascending=False).head(18)
        if len(because_like_b) > 0:
            st.markdown("<div class='mm-section-head'><p class='mm-section-title'>Because you liked</p><p class='mm-section-sub'>Matches to your recent likes</p></div>", unsafe_allow_html=True)
            st.markdown(_render_book_rail(because_like_b), unsafe_allow_html=True)
            shown_b |= set(because_like_b["book_id"].astype(int).tolist())

    if len(all_ranked_books) > 0:
        tmp2 = all_ranked_books[~all_ranked_books["book_id"].astype(int).isin(list(shown_b))]
        more_vibe_b = diversify_topk(tmp2, k=18, text_col="tags_text", lambda_penalty=0.32)
        if len(more_vibe_b) > 0:
            st.markdown("<div class='mm-section-head'><p class='mm-section-title'>More for your vibe</p><p class='mm-section-sub'>Explore deeper cuts</p></div>", unsafe_allow_html=True)
            st.markdown(_render_book_rail(more_vibe_b), unsafe_allow_html=True)

st.divider()
st.subheader("My List")
if st.session_state.watchlist:
    movies_txt = "\n".join([f"- {x['title']}" for x in st.session_state.watchlist if x["type"] == "movie"])
    books_txt = "\n".join([f"- {x['title']}" for x in st.session_state.watchlist if x["type"] == "book"])
    text = "*MoodMatch — My Picks*\n\n"
    if movies_txt: text += "*Movies:*\n" + movies_txt + "\n\n"
    if books_txt: text += "*Books:*\n" + books_txt + "\n\n"
    st.text_area("Copy & Paste", value=text.strip(), height=200)
    if st.button("Clear List"):
        st.session_state.watchlist = []
        st.rerun()
else:
    st.info("Your list is empty.")