import json
import time
import boto3
import pandas as pd
import requests
import streamlit as st

# ── page config ───────────────────────────────────────────
st.set_page_config(
    page_title="News Recommender System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    margin: 0;
    color: white;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
}

.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 600;
    color: #0f172a;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}

.article-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.article-card-clicked {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.article-title {
    font-size: 1rem;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 0.4rem;
    line-height: 1.4;
}
.article-abstract {
    font-size: 0.85rem;
    color: #64748b;
    line-height: 1.5;
    margin-bottom: 0.6rem;
}
.article-meta {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    align-items: center;
}
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: capitalize;
}
.badge-category { background: #dbeafe; color: #1e40af; }
.badge-source   { background: #f1f5f9; color: #475569; }
.badge-clicked  { background: #dcfce7; color: #166534; }

.pipeline-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    height: 100%;
}
.pipeline-box .icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
.pipeline-box .name { font-weight: 600; color: #0f172a; font-size: 0.9rem; }
.pipeline-box .desc { color: #64748b; font-size: 0.75rem; margin-top: 0.2rem; }

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-ok    { background: #22c55e; }
.status-warn  { background: #f59e0b; }
.status-error { background: #ef4444; }

.click-hint {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #1e40af;
    margin-bottom: 1rem;
}
.history-item {
    display: inline-block;
    background: #f1f5f9;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    color: #475569;
    margin: 0.2rem;
}
.dup-box-ok {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
}
.dup-box-warn {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
}
.code-small {
    font-size: 0.78rem;
}
</style>
""", unsafe_allow_html=True)

# ── config ────────────────────────────────────────────────
BASE_API_URL = "https://qe9e1zaok3.execute-api.ap-southeast-7.amazonaws.com/prod"
REC_URL      = f"{BASE_API_URL}/recommend"
CLICK_URL    = f"{BASE_API_URL}/click"

REGION       = "ap-southeast-7"
RAW_BUCKET   = "news-recommending-raw-st125934"
PROC_BUCKET  = "news-recommending-processed-st125934"
MODEL_BUCKET = "news-recommending-models-st125934"

SAMPLE_USERS = ["U13740", "U91836", "U73700", "U34670", "U8125", "U52780"]
SOURCE_LABELS = {
    "microsoft_mind": "MIND Dataset",
    "mind": "MIND Dataset",
    "newsapi": "Live News",
    "unknown": "Unknown",
}

REQUEST_HEADERS = {"Content-Type": "application/json"}

# ── session state ─────────────────────────────────────────
if "clicked_articles" not in st.session_state:
    st.session_state.clicked_articles = {}
if "current_recs" not in st.session_state:
    st.session_state.current_recs = []
if "current_user" not in st.session_state:
    st.session_state.current_user = "U13740"
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0
if "last_model_type" not in st.session_state:
    st.session_state.last_model_type = "unknown"
if "dup_summary" not in st.session_state:
    st.session_state.dup_summary = {
        "dup_ids": 0,
        "dup_titles": 0,
        "dup_urls": 0
    }

# ── helpers ───────────────────────────────────────────────
def safe_json_response(resp):
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}

def fetch_recommendations(user_id: str, k: int):
    start = time.time()
    resp = requests.post(
        REC_URL,
        json={"user_id": user_id, "k": k},
        headers=REQUEST_HEADERS,
        timeout=30
    )
    elapsed_ms = (time.time() - start) * 1000
    data = safe_json_response(resp)
    return resp, data, elapsed_ms

def record_click(user_id: str, article_id: str):
    resp = requests.post(
        CLICK_URL,
        json={"user_id": user_id, "article_id": article_id},
        headers=REQUEST_HEADERS,
        timeout=15
    )
    data = safe_json_response(resp)
    return resp, data

def get_dynamo_count(table_name: str):
    try:
        dynamo = boto3.resource("dynamodb", region_name=REGION)
        table = dynamo.Table(table_name)
        return table.scan(Select="COUNT").get("Count", 0)
    except Exception:
        return None

def analyse_duplicates(recs):
    article_ids = [str(x.get("article_id", "")).strip() for x in recs]
    titles = [str(x.get("title", "")).strip().lower() for x in recs]
    urls = [str(x.get("url", "")).strip().lower() for x in recs]

    dup_ids = len(article_ids) - len(set([x for x in article_ids if x]))
    dup_titles = len(titles) - len(set([x for x in titles if x]))
    dup_urls = len(urls) - len(set([x for x in urls if x]))

    return {
        "dup_ids": dup_ids,
        "dup_titles": dup_titles,
        "dup_urls": dup_urls
    }

# ── header ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📰 Personalized News Recommender</h1>
    <p>Data Engineering &amp; MLOps Project — AWS Serverless Pipeline with Real-Time Personalisation</p>
</div>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Project Info")
    st.markdown("""
<div style="background:#f1f5f9;border-radius:8px;padding:0.8rem;font-size:0.8rem;color:#475569;margin-bottom:1rem">
    <strong>Stack</strong><br>
    AWS S3 · Glue · DynamoDB<br>
    Lambda · API Gateway · Athena<br>
    scikit-learn · MLflow · Streamlit<br><br>
    <strong>Serving APIs</strong><br>
    /recommend → news-recommender-serving<br>
    /click → news-click-tracker<br><br>
    <strong>Region</strong><br>
    ap-southeast-7
</div>
""", unsafe_allow_html=True)

    st.markdown("### Quick Test")
    quick_user = st.selectbox("Sample user", SAMPLE_USERS)
    quick_k = st.slider("Recommendations", 1, 10, 5)

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        st.session_state.current_user = quick_user
        st.session_state.clicked_articles = {}
        st.session_state.current_recs = []

    st.markdown("---")
    st.markdown("### API Endpoints")
    st.code(REC_URL, language="text")
    st.code(CLICK_URL, language="text")

    st.markdown("---")
    st.markdown("### How personalisation works")
    st.markdown("""
1. Enter a user ID  
2. Click **Recommend**  
3. Click **Read** on an article  
4. The app sends a request to **/click**  
5. Hit **Refresh**  
6. The app calls **/recommend** again using the updated user profile
""")

# ── tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Demo",
    "🏗️ Pipeline",
    "📊 Model Performance",
    "🖥️ System Monitor"
])

# ════════════════════════════════════════════════════════
# TAB 1 — LIVE DEMO
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Personalised Recommendation Demo")

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        user_id = st.text_input("User ID", value=st.session_state.current_user)
    with col2:
        k = st.number_input("# Results", min_value=1, max_value=20, value=5)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        get_recs = st.button("Recommend →", type="primary", use_container_width=True)
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("↺ Refresh", use_container_width=True)

    if get_recs:
        st.session_state.clicked_articles = {}
        st.session_state.current_user = user_id
        st.session_state.current_recs = []

    should_fetch = get_recs or refresh or not st.session_state.current_recs

    if should_fetch:
        with st.spinner("Fetching from AWS API..."):
            try:
                resp, result, elapsed_ms = fetch_recommendations(user_id, int(k))

                if resp.status_code == 200:
                    st.session_state.current_recs = result.get("recommendations", [])
                    st.session_state.last_latency = result.get("latency_ms", elapsed_ms)
                    st.session_state.last_model_type = result.get("model_type", "unknown")
                    st.session_state.current_user = user_id
                    st.session_state.dup_summary = analyse_duplicates(st.session_state.current_recs)
                else:
                    st.session_state.current_recs = []
                    raw_error = result.get("error") or result.get("raw_text") or "Unknown error"
                    st.error(f"API error {resp.status_code}: {raw_error}")

            except Exception as e:
                st.session_state.current_recs = []
                st.error(f"Connection error: {e}")

    clicked_count = len(st.session_state.clicked_articles)

    if clicked_count == 0:
        st.markdown("""
        <div class="click-hint">
            💡 <strong>Try the personalisation loop:</strong>
            Click any article below, then hit <strong>↺ Refresh</strong>
            to see recommendations update based on your reading interest.
        </div>
        """, unsafe_allow_html=True)
    else:
        items = "".join([
            f'<span class="history-item">✓ {title[:40]}...</span>'
            for title in st.session_state.clicked_articles.values()
        ])
        st.markdown(f"""
        <div class="click-hint" style="background:#f0fdf4;border-color:#bbf7d0;color:#166534">
            ✅ <strong>{clicked_count} article(s) read.</strong>
            Hit <strong>↺ Refresh</strong> to update recommendations.<br>{items}
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.current_recs:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="value">{len(st.session_state.current_recs)}</div><div class="label">Recommendations</div></div>',
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="value">{st.session_state.last_latency:.0f}ms</div><div class="label">API Latency</div></div>',
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="value">{clicked_count}</div><div class="label">Articles Clicked</div></div>',
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f'<div class="metric-card"><div class="value" style="font-size:0.95rem;padding-top:0.5rem">{st.session_state.last_model_type}</div><div class="label">Model Used</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        dup_ids = st.session_state.dup_summary["dup_ids"]
        dup_titles = st.session_state.dup_summary["dup_titles"]
        dup_urls = st.session_state.dup_summary["dup_urls"]

        if dup_ids == 0 and dup_titles == 0 and dup_urls == 0:
            st.markdown(
                '<div class="dup-box-ok">✅ <strong>Duplicate check passed.</strong> '
                'No duplicate article_id, title, or URL was found in this recommendation set.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="dup-box-warn">⚠️ <strong>Duplicate check warning.</strong> '
                f'Duplicate article_id: {dup_ids}, title: {dup_titles}, url: {dup_urls}</div>',
                unsafe_allow_html=True
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.code(f"duplicate article_id: {dup_ids}", language="text")
        with c2:
            st.code(f"duplicate title: {dup_titles}", language="text")
        with c3:
            st.code(f"duplicate url: {dup_urls}", language="text")

        st.markdown(f"#### Recommended for `{user_id}`")

        for i, article in enumerate(st.session_state.current_recs):
            art_id = article.get("article_id", "")
            title = article.get("title", "Unknown")
            abstract = article.get("abstract", "")
            category = str(article.get("category", "general")).lower()
            source = article.get("source", "unknown")
            is_clicked = art_id in st.session_state.clicked_articles

            card_class = "article-card-clicked" if is_clicked else "article-card"
            clicked_badge = '<span class="badge badge-clicked">✓ Read</span>' if is_clicked else ''

            card_col, btn_col = st.columns([5, 1])

            with card_col:
                st.markdown(
                    f'<div class="{card_class}">'
                    f'<div class="article-title">#{i+1} &nbsp; {title}</div>'
                    f'<div class="article-abstract">{abstract}</div>'
                    f'<div class="article-meta">'
                    f'<span class="badge badge-category">{category}</span>'
                    f'<span class="badge badge-source">{SOURCE_LABELS.get(source, source)}</span>'
                    f'{clicked_badge}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with btn_col:
                st.markdown("<br><br>", unsafe_allow_html=True)

                if not is_clicked and art_id:
                    if st.button("👆 Read", key=f"click_{art_id}_{i}", use_container_width=True):
                        with st.spinner("Recording click..."):
                            try:
                                click_resp, click_result = record_click(user_id, art_id)

                                if click_resp.status_code == 200:
                                    st.session_state.clicked_articles[art_id] = title
                                    st.success("Click recorded. Hit Refresh.")
                                    st.rerun()
                                else:
                                    raw_error = click_result.get("error") or click_result.get("raw_text") or "Unknown click error"
                                    st.error(f"Click API error {click_resp.status_code}: {raw_error}")

                            except Exception as e:
                                st.error(f"Click error: {e}")
                elif is_clicked:
                    st.markdown("✅ Read")
    else:
        st.info("Enter a user ID and click **Recommend →**")

# ════════════════════════════════════════════════════════
# TAB 2 — PIPELINE
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("### System Architecture")
    st.markdown("End-to-end serverless data pipeline built on AWS with separate recommend and click APIs.")

    stages = [
        ("📡", "NewsAPI", "REST pull every 30 min"),
        ("📦", "AWS S3 Raw", "JSON + TSV storage"),
        ("⚙️", "AWS Glue", "PySpark ETL + Parquet"),
        ("🗄️", "DynamoDB", "Feature store"),
        ("🧠", "Model Registry", "Best model in S3"),
        ("🚀", "API Gateway + Lambda", "Recommend + click"),
    ]
    cols = st.columns(len(stages))
    for col, (icon, name, desc) in zip(cols, stages):
        with col:
            st.markdown(
                f'<div class="pipeline-box">'
                f'<div class="icon">{icon}</div>'
                f'<div class="name">{name}</div>'
                f'<div class="desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Data Flow")
        for step, detail in [
            ("EventBridge", "Triggers ingestion Lambda"),
            ("Ingestion Lambda", "Pulls live news into raw S3"),
            ("MIND dataset", "Historical dataset stored in S3"),
            ("Glue ETL", "Transforms raw data to processed Parquet"),
            ("write_embeddings.py", "Writes article embeddings to DynamoDB"),
            ("write_user_vectors.py", "Writes user vectors to DynamoDB"),
            ("Model training", "Saves best model into S3 registry"),
        ]:
            st.markdown(f"**{step}** — {detail}")

    with right:
        st.markdown("#### Personalisation Loop")
        for step, detail in [
            ("Recommend", "POST /recommend → news-recommender-serving"),
            ("User reads", "Article chosen in Streamlit"),
            ("Click update", "POST /click → news-click-tracker"),
            ("User profile", "recent_clicks and top_category updated"),
            ("Refresh", "Streamlit calls /recommend again"),
            ("Result", "Recommendations adapt to latest reading history"),
        ]:
            st.markdown(f"**{step}** — {detail}")

# ════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Evaluation Results")

    st.dataframe(pd.DataFrame({
        "Model": ["Popularity Baseline", "SVD Matrix Factorisation", "KNN Content-Based"],
        "nDCG@10": [0.0543, 0.0028, 0.0507],
        "Precision@5": [0.0160, 0.0020, 0.0140],
        "Recall@10": [0.0948, 0.0047, 0.0905],
        "Personalised": ["No", "Yes", "Yes"],
    }), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### nDCG@10")
        st.bar_chart(pd.DataFrame(
            {"nDCG@10": [0.0543, 0.0028, 0.0507]},
            index=["Popularity", "SVD", "KNN"]
        ))
    with c2:
        st.markdown("#### Recall@10")
        st.bar_chart(pd.DataFrame(
            {"Recall@10": [0.0948, 0.0047, 0.0905]},
            index=["Popularity", "SVD", "KNN"]
        ))

# ════════════════════════════════════════════════════════
# TAB 4 — SYSTEM MONITOR
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Live System Status")

    if st.button("🔄 Refresh Status"):
        st.rerun()

    with st.spinner("Checking AWS..."):
        try:
            s3 = boto3.client("s3", region_name=REGION)
            dynamo = boto3.resource("dynamodb", region_name=REGION)
            lam = boto3.client("lambda", region_name=REGION)

            st.markdown("#### S3 Buckets")
            cols = st.columns(3)
            for col, (bucket, label, prefix) in zip(cols, [
                (RAW_BUCKET, "Raw", "newsapi/"),
                (PROC_BUCKET, "Processed", "articles/"),
                (MODEL_BUCKET, "Models", "registry/"),
            ]):
                with col:
                    try:
                        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                        count = resp.get("KeyCount", 0)
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<span class="status-dot status-ok"></span>'
                            f'<strong>{label}</strong><br>'
                            f'<span style="font-size:1.4rem;font-weight:600">{count}</span> '
                            f'<span style="color:#64748b;font-size:0.8rem">files</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(str(e))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### DynamoDB Tables")
            c1, c2 = st.columns(2)
            for col, tname in [(c1, "article-embeddings"), (c2, "user-vectors")]:
                with col:
                    count = get_dynamo_count(tname)
                    if count is not None:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<span class="status-dot status-ok"></span>'
                            f'<strong>{tname}</strong><br>'
                            f'<span style="font-size:1.6rem;font-weight:600">{count:,}</span> '
                            f'<span style="color:#64748b;font-size:0.8rem">items</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.error(f"Could not read {tname}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Lambda Functions")
            c1, c2, c3 = st.columns(3)
            for col, fn in [
                (c1, "news-ingestion-function"),
                (c2, "news-recommender-serving"),
                (c3, "news-click-tracker"),
            ]:
                with col:
                    try:
                        cfg = lam.get_function(FunctionName=fn)["Configuration"]
                        state = cfg["State"]
                        dot = "status-ok" if state == "Active" else "status-error"
                        state_color = "#22c55e" if state == "Active" else "#ef4444"
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<span class="status-dot {dot}"></span>'
                            f'<strong>{fn}</strong><br>'
                            f'<span style="color:{state_color};font-weight:600">{state}</span><br>'
                            f'<span style="color:#64748b;font-size:0.75rem">{cfg["Runtime"]} · {cfg["MemorySize"]}MB</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(str(e))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Deployed API")
            st.code(REC_URL, language="text")
            st.code(CLICK_URL, language="text")

        except Exception as e:
            st.error(f"AWS connection error: {e}")

# ── footer ────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:0.8rem;border-top:1px solid #e2e8f0;padding-top:1rem">'
    'Personalized News Recommender · AWS REST API · Recommend + Click Loop · Streamlit Demo'
    '</div>',
    unsafe_allow_html=True
)