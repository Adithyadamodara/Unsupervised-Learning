"""
IDEC Review Intelligence Dashboard
====================================
Streamlit app for exploring deep-clustered productivity app reviews.
Deploy to Hugging Face Spaces (streamlit SDK).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import os

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Review Intelligence | IDEC",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Light background */
.stApp {
    background-color: #f8fafc;
    color: #1e293b;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background: #eff6ff !important;
    color: #3b82f6 !important;
    font-weight: 500;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Cards */
.review-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 14px;
    line-height: 1.6;
    color: #334155;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.review-card .score {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #3b82f6;
    margin-bottom: 6px;
    font-weight: 500;
}

/* Section headers */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 12px;
    margin-top: 24px;
    font-weight: 500;
}

/* Cluster badge */
.cluster-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
}

/* Table styling */
.comparison-table {
    width: 100%;
    border-collapse: collapse;
}
.comparison-table th {
    background: #f1f5f9;
    color: #475569;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    padding: 10px 14px;
    text-align: left;
    letter-spacing: 1px;
    border-top: 1px solid #e2e8f0;
    border-bottom: 2px solid #cbd5e1;
}
.comparison-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #e2e8f0;
    font-size: 13px;
    color: #334155;
    background: #ffffff;
}
.comparison-table tr:hover td {
    background: #f8fafc;
}
.best-value {
    color: #16a34a;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}
.worst-value {
    color: #ef4444;
    font-family: 'DM Mono', monospace;
}
.mid-value {
    color: #d97706;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Cluster metadata ──────────────────────────────────────────────
CLUSTER_THEMES = {
    0: {"name": "Feature Limitations & UX Friction",   "color": "#3b82f6", "emoji": "🔧"},
    1: {"name": "Sync & Account Degradation",           "color": "#f97316", "emoji": "🔄"},
    2: {"name": "General Negative Feedback & Design",   "color": "#ef4444", "emoji": "👎"},
    3: {"name": "Ad Intrusiveness & App Decay",         "color": "#8b5cf6", "emoji": "📢"},
    4: {"name": "Positive Engagement & Gamification",   "color": "#22c55e", "emoji": "🎮"},
    5: {"name": "Casual Usage & Feature Requests",      "color": "#eab308", "emoji": "💡"},
}

BASELINE_RESULTS = {
    "TF-IDF + K-Means":  {"silhouette": 0.0059, "davies_bouldin": 10.5128, "clusters": 6},
    "BERT + K-Means":    {"silhouette": 0.0283, "davies_bouldin": 3.9668,  "clusters": 6},
    "UMAP + HDBSCAN":    {"silhouette": 0.4039, "davies_bouldin": None,    "clusters": 2},
    "IDEC (ours)":       {"silhouette": 0.8420, "davies_bouldin": 0.2000,  "clusters": 6},
}

IDEC_TRAINING_LOG = [
    (1,  0.01206, 0.10626, 0.4548, 6.02),
    (5,  0.01695, 0.15442, 0.5950, 2.84),
    (10, 0.01854, 0.16972, 0.6670, 1.44),
    (15, 0.01918, 0.17593, 0.7258, 0.72),
    (20, 0.01942, 0.17827, 0.7808, 0.50),
    (25, 0.01853, 0.16934, 0.8106, 0.54),
    (30, 0.01871, 0.17116, 0.8354, 0.22),
]

# ── Data loading ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all saved artifacts. Returns None values if files not found."""
    try:
        df = pd.read_csv("reviews_processed.csv")
        labels = np.load("labels_idec.npy")
        Z = np.load("latent_vectors_idec.npy")
        Q = np.load("soft_assignments_idec.npy")
        df["cluster"] = labels
        df["confidence"] = Q.max(axis=1)
        return df, labels, Z, Q
    except FileNotFoundError as e:
        st.error(f"Missing artifact: {e}. Make sure all .npy and .csv files are in the same directory as app.py.")
        return None, None, None, None

@st.cache_data
def compute_umap_projection(Z):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(Z)

@st.cache_data
def get_top_keywords(texts, n=12):
    """c-TF-IDF style top keywords per cluster."""
    if len(texts) < 3:
        return []
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2)
    try:
        matrix = tfidf.fit_transform(texts)
        mean_tfidf = matrix.mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:n]
        return [tfidf.get_feature_names_out()[i] for i in top_idx]
    except Exception:
        return []

# ── Load data ─────────────────────────────────────────────────────
df, labels, Z, Q = load_data()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Review Intelligence")
    st.markdown("<div style='font-family: DM Mono, monospace; font-size: 11px; color: #64748b; letter-spacing: 2px;'>IDEC · DEEP CLUSTERING</div>", unsafe_allow_html=True)
    st.divider()

    if df is not None:
        st.markdown("<div class='section-header'>Dataset</div>", unsafe_allow_html=True)
        st.metric("Total Reviews", f"{len(df):,}")
        st.metric("Clusters", "6")
        st.metric("Silhouette Score", "0.8420 ± 0.0025")
        st.metric("Applications", "15")

        st.divider()
        st.markdown("<div class='section-header'>Cluster Overview</div>", unsafe_allow_html=True)
        for cid, meta in CLUSTER_THEMES.items():
            count = (labels == cid).sum()
            pct = count / len(labels) * 100
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>"
                f"<span style='width:10px; height:10px; border-radius:50%; background:{meta['color']}; display:inline-block; flex-shrink:0;'></span>"
                f"<span style='font-size:12px; color:#475569;'>C{cid}: {count:,} ({pct:.0f}%)</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown(
        "<div style='font-size:11px; color:#94a3b8; font-family: DM Mono, monospace;'>"
        "Adithya Damodara · LPU<br>IDEC Pipeline · 2024"
        "</div>",
        unsafe_allow_html=True
    )

# ── Main content ──────────────────────────────────────────────────
st.markdown("# Review Intelligence Dashboard")
st.markdown("<div style='color:#64748b; font-size:15px; margin-bottom:24px;'>Unsupervised pain point discovery across 15 productivity apps using Improved Deep Embedded Clustering</div>", unsafe_allow_html=True)

if df is None:
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Latent Space",
    "🔬  Cluster Explorer",
    "📈  Baselines",
    "⚙️  Training Dynamics"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — LATENT SPACE VISUALIZATION
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>UMAP Projection of IDEC Latent Space</div>", unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        color_by = st.selectbox("Color by", ["Cluster", "Review Score", "Confidence"])
    with col_ctrl2:
        sample_size = st.slider("Sample size", 500, min(5000, len(df)), 2000, 500)

    with st.spinner("Computing UMAP projection..."):
        sample_idx = np.random.default_rng(42).choice(len(Z), sample_size, replace=False)
        Z_2d = compute_umap_projection(Z)
        Z_sample = Z_2d[sample_idx]
        df_sample = df.iloc[sample_idx].copy()
        df_sample["umap_x"] = Z_sample[:, 0]
        df_sample["umap_y"] = Z_sample[:, 1]
        df_sample["cluster_name"] = df_sample["cluster"].map(
            lambda c: f"C{c}: {CLUSTER_THEMES[c]['name']}"
        )
        df_sample["hover_text"] = df_sample["content_clean"].str[:120] + "..."

    if color_by == "Cluster":
        color_col = "cluster_name"
        color_map = {
            f"C{c}: {CLUSTER_THEMES[c]['name']}": CLUSTER_THEMES[c]["color"]
            for c in CLUSTER_THEMES
        }
        fig = px.scatter(
            df_sample, x="umap_x", y="umap_y",
            color=color_col,
            color_discrete_map=color_map,
            hover_data={"umap_x": False, "umap_y": False,
                        "hover_text": True, "score": True, "cluster_name": False},
            labels={"hover_text": "Review", "score": "Score"},
            height=560
        )
    elif color_by == "Review Score":
        fig = px.scatter(
            df_sample, x="umap_x", y="umap_y",
            color="score",
            color_continuous_scale="RdYlGn",
            hover_data={"umap_x": False, "umap_y": False, "hover_text": True},
            labels={"hover_text": "Review"},
            height=560
        )
    else:
        fig = px.scatter(
            df_sample, x="umap_x", y="umap_y",
            color="confidence",
            color_continuous_scale="Viridis",
            hover_data={"umap_x": False, "umap_y": False, "hover_text": True},
            labels={"hover_text": "Review", "confidence": "Confidence"},
            height=560
        )

    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0.5, color="#ffffff")))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155", family="DM Sans"),
        legend=dict(
            bgcolor="#ffffff",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, showticklabels=False, title=""),
        margin=dict(l=0, r=0, t=20, b=0),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="DM Sans")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Each point is one review projected to 2D via UMAP from the 32-dimensional IDEC latent space. Distinct islands confirm that joint training successfully separated cluster geometry.")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTER EXPLORER
# ══════════════════════════════════════════════════════════════════
with tab2:
    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        st.markdown("<div class='section-header'>Select Cluster</div>", unsafe_allow_html=True)
        selected_cluster = st.radio(
            "Cluster",
            options=list(CLUSTER_THEMES.keys()),
            format_func=lambda c: f"{CLUSTER_THEMES[c]['emoji']} C{c}: {CLUSTER_THEMES[c]['name']}",
            label_visibility="collapsed"
        )

    with col_info:
        meta = CLUSTER_THEMES[selected_cluster]
        cluster_df = df[df["cluster"] == selected_cluster]
        count = len(cluster_df)
        pct = count / len(df) * 100
        avg_score = cluster_df["score"].mean()
        avg_conf = cluster_df["confidence"].mean()

        st.markdown(
            f"<h3 style='color:{meta['color']}; margin-bottom:4px;'>"
            f"{meta['emoji']} {meta['name']}</h3>",
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reviews", f"{count:,}")
        m2.metric("% of Dataset", f"{pct:.1f}%")
        m3.metric("Avg Score", f"{avg_score:.2f} ★")
        m4.metric("Avg Confidence", f"{avg_conf:.3f}")

    st.divider()

    col_reviews, col_keywords = st.columns([3, 2])

    with col_reviews:
        st.markdown("<div class='section-header'>Representative Reviews</div>", unsafe_allow_html=True)

        # Sort by confidence (closest to centroid = most representative)
        top_reviews = cluster_df.nlargest(8, "confidence")

        for _, row in top_reviews.iterrows():
            score_stars = "★" * int(row["score"]) + "☆" * (5 - int(row["score"]))
            st.markdown(
                f"<div class='review-card'>"
                f"<div class='score'>{score_stars} &nbsp;·&nbsp; conf: {row['confidence']:.3f}</div>"
                f"{row['content_clean'][:280]}"
                f"</div>",
                unsafe_allow_html=True
            )

    with col_keywords:
        st.markdown("<div class='section-header'>Top Keywords (c-TF-IDF)</div>", unsafe_allow_html=True)

        keywords = get_top_keywords(
            cluster_df["content_processed"].dropna().tolist(), n=15
        )
        if keywords:
            for i, kw in enumerate(keywords):
                opacity = 1.0 - (i * 0.05)
                bar_width = 100 - (i * 6)
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:10px; margin-bottom:7px;'>"
                    f"<div style='font-family: DM Mono, monospace; font-size:12px; "
                    f"color:{meta['color']}; opacity:{opacity:.2f}; width:160px; "
                    f"white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{kw}</div>"
                    f"<div style='height:6px; background:{meta['color']}; opacity:{opacity:.2f}; "
                    f"border-radius:3px; width:{bar_width}%;'></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("<div class='section-header' style='margin-top:24px;'>Score Distribution</div>", unsafe_allow_html=True)
        score_counts = cluster_df["score"].value_counts().sort_index()
        fig_score = px.bar(
            x=score_counts.index,
            y=score_counts.values,
            labels={"x": "Score", "y": "Count"},
            color_discrete_sequence=[meta["color"]],
            height=200
        )
        fig_score.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", size=11),
            margin=dict(l=0, r=0, t=10, b=30),
            showlegend=False,
            xaxis=dict(tickmode="array", tickvals=[1,2,3,4,5],
                       gridcolor="#e2e8f0", tickfont=dict(size=11)),
            yaxis=dict(gridcolor="#e2e8f0", tickfont=dict(size=11))
        )
        st.plotly_chart(fig_score, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Method Comparison</div>", unsafe_allow_html=True)

    # Metrics table
    table_html = """
    <table class='comparison-table'>
    <thead>
        <tr>
            <th>METHOD</th>
            <th>CLUSTERS FOUND</th>
            <th>SILHOUETTE ↑</th>
            <th>DAVIES-BOULDIN ↓</th>
            <th>NOTE</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>TF-IDF + K-Means</td>
            <td>6</td>
            <td class='worst-value'>0.0059</td>
            <td class='worst-value'>10.5128</td>
            <td style='color:#64748b; font-size:12px;'>50% of data in one catch-all cluster</td>
        </tr>
        <tr>
            <td>BERT + K-Means</td>
            <td>6</td>
            <td class='mid-value'>0.0283</td>
            <td class='mid-value'>3.9668</td>
            <td style='color:#64748b; font-size:12px;'>Captures sentiment, not topics</td>
        </tr>
        <tr>
            <td>UMAP + HDBSCAN</td>
            <td>2 (auto)</td>
            <td class='mid-value'>0.4039</td>
            <td style='color:#64748b;'>—</td>
            <td style='color:#64748b; font-size:12px;'>Only finds coarse sentiment split</td>
        </tr>
        <tr>
            <td><strong style='color:#3b82f6;'>IDEC (ours)</strong></td>
            <td><strong style='color:#16a34a;'>6</strong></td>
            <td class='best-value'>0.8420 ± 0.0025</td>
            <td class='best-value'>0.2000</td>
            <td style='color:#64748b; font-size:12px;'>3 seeds · balanced distribution</td>
        </tr>
    </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual comparison — silhouette bar chart
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<div class='section-header'>Silhouette Score Comparison</div>", unsafe_allow_html=True)
        methods = list(BASELINE_RESULTS.keys())
        sil_vals = [BASELINE_RESULTS[m]["silhouette"] for m in methods]
        colors_bar = ["#ef4444", "#eab308", "#8b5cf6", "#22c55e"]

        fig_sil = go.Figure(go.Bar(
            x=methods, y=sil_vals,
            marker_color=colors_bar,
            text=[f"{v:.4f}" for v in sil_vals],
            textposition="outside",
            textfont=dict(family="DM Mono", size=11, color="#334155")
        ))
        fig_sil.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="DM Sans"),
            margin=dict(l=0, r=0, t=30, b=60),
            height=280,
            yaxis=dict(gridcolor="#e2e8f0", range=[0, 1.0]),
            xaxis=dict(tickfont=dict(size=11)),
            showlegend=False
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    with col_chart2:
        st.markdown("<div class='section-header'>Cluster Size Distribution</div>", unsafe_allow_html=True)

        # Show IDEC cluster sizes
        cluster_sizes = [int((labels == k).sum()) for k in range(6)]
        cluster_labels = [f"C{k}" for k in range(6)]
        cluster_colors = [CLUSTER_THEMES[k]["color"] for k in range(6)]

        fig_dist = go.Figure(go.Bar(
            x=cluster_labels,
            y=cluster_sizes,
            marker_color=cluster_colors,
            text=cluster_sizes,
            textposition="outside",
            textfont=dict(family="DM Mono", size=11, color="#334155")
        ))
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="DM Sans"),
            margin=dict(l=0, r=0, t=30, b=40),
            height=280,
            yaxis=dict(gridcolor="#e2e8f0"),
            showlegend=False,
            title=dict(text="IDEC — Balanced Distribution", font=dict(size=12, color="#64748b"))
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Pipeline ablation story
    st.divider()
    st.markdown("<div class='section-header'>Ablation: Value Added Per Stage</div>", unsafe_allow_html=True)

    ablation_data = {
        "Stage": [
            "Raw BERT Embeddings (384d)",
            "+ Autoencoder Compression (32d)",
            "+ IDEC Joint Training"
        ],
        "Silhouette": [0.0283, 0.2032, 0.8420],
        "Improvement": ["baseline", "+8.6×", "+4.1× further"]
    }
    abl_df = pd.DataFrame(ablation_data)

    fig_abl = go.Figure()
    fig_abl.add_trace(go.Bar(
        x=abl_df["Stage"],
        y=abl_df["Silhouette"],
        marker_color=["#ef4444", "#eab308", "#22c55e"],
        text=abl_df["Silhouette"],
        texttemplate="%{text:.4f}",
        textposition="outside",
        textfont=dict(family="DM Mono", size=12, color="#334155")
    ))
    fig_abl.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", family="DM Sans"),
        margin=dict(l=0, r=0, t=20, b=80),
        height=280,
        yaxis=dict(gridcolor="#e2e8f0", range=[0, 1.0], title="Silhouette Score"),
        xaxis=dict(tickfont=dict(size=11)),
        showlegend=False
    )
    st.plotly_chart(fig_abl, use_container_width=True)
    st.caption("Each pipeline stage adds measurable improvement in cluster separability. The autoencoder compression alone provides 8.6× improvement; IDEC joint training provides a further 4.1× on top of that.")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — TRAINING DYNAMICS
# ══════════════════════════════════════════════════════════════════
with tab4:
    log_df = pd.DataFrame(
        IDEC_TRAINING_LOG,
        columns=["Epoch", "Total Loss", "KL Loss", "Silhouette", "% Changed"]
    )

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("<div class='section-header'>Silhouette Score Progression</div>", unsafe_allow_html=True)
        fig_sil_prog = go.Figure()
        fig_sil_prog.add_hline(
            y=0.2032, line_dash="dash",
            line_color="#94a3b8",
            annotation_text="Pre-IDEC baseline (0.2032)",
            annotation_font_color="#64748b",
            annotation_font_size=11
        )
        fig_sil_prog.add_trace(go.Scatter(
            x=log_df["Epoch"], y=log_df["Silhouette"],
            mode="lines+markers",
            line=dict(color="#22c55e", width=2.5),
            marker=dict(size=7, color="#22c55e"),
            fill="tozeroy",
            fillcolor="rgba(34, 197, 94, 0.15)",
            name="Silhouette"
        ))
        fig_sil_prog.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="DM Sans"),
            margin=dict(l=0, r=0, t=20, b=40),
            height=280,
            yaxis=dict(gridcolor="#e2e8f0", range=[0, 1.0]),
            xaxis=dict(gridcolor="#e2e8f0", title="Epoch"),
            showlegend=False
        )
        st.plotly_chart(fig_sil_prog, use_container_width=True)

    with col_t2:
        st.markdown("<div class='section-header'>Assignment Stability (% Changed)</div>", unsafe_allow_html=True)
        fig_changed = go.Figure()
        fig_changed.add_hline(
            y=1.0, line_dash="dash",
            line_color="#ef4444",
            annotation_text="1% convergence threshold",
            annotation_font_color="#ef4444",
            annotation_font_size=11
        )
        fig_changed.add_trace(go.Scatter(
            x=log_df["Epoch"], y=log_df["% Changed"],
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=7, color="#3b82f6"),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.15)",
            name="% Changed"
        ))
        fig_changed.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="DM Sans"),
            margin=dict(l=0, r=0, t=20, b=40),
            height=280,
            yaxis=dict(gridcolor="#e2e8f0", title="% Assignments Changed"),
            xaxis=dict(gridcolor="#e2e8f0", title="Epoch"),
            showlegend=False
        )
        st.plotly_chart(fig_changed, use_container_width=True)

    col_t3, col_t4 = st.columns(2)

    with col_t3:
        st.markdown("<div class='section-header'>Loss Components</div>", unsafe_allow_html=True)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=log_df["Epoch"], y=log_df["Total Loss"],
            mode="lines+markers",
            line=dict(color="#eab308", width=2),
            marker=dict(size=6),
            name="Total Loss"
        ))
        fig_loss.add_trace(go.Scatter(
            x=log_df["Epoch"], y=log_df["KL Loss"],
            mode="lines+markers",
            line=dict(color="#ef4444", width=2, dash="dot"),
            marker=dict(size=6),
            name="KL Clustering Loss"
        ))
        fig_loss.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="DM Sans"),
            margin=dict(l=0, r=0, t=20, b=40),
            height=280,
            yaxis=dict(gridcolor="#e2e8f0"),
            xaxis=dict(gridcolor="#e2e8f0", title="Epoch"),
            legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=11), bordercolor="#e2e8f0", borderwidth=1)
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_t4:
        st.markdown("<div class='section-header'>Training Log</div>", unsafe_allow_html=True)
        st.dataframe(
            log_df.style.format({
                "Total Loss": "{:.5f}",
                "KL Loss": "{:.5f}",
                "Silhouette": "{:.4f}",
                "% Changed": "{:.2f}"
            }).background_gradient(
                subset=["Silhouette"],
                cmap="Greens"
            ),
            use_container_width=True,
            height=280
        )

    st.divider()
    st.markdown("<div class='section-header'>Overtraining Finding</div>", unsafe_allow_html=True)

    overtrain_data = {
        "Checkpoint": ["Epoch 50 (selected)", "Epoch 100", "Epoch 150", "Epoch 200 (overtrained)"],
        "Recon Loss": [0.001456, 0.001115, 0.000836, 0.000794],
        "Silhouette": [0.2385, 0.18, 0.12, 0.0776]
    }
    ot_df = pd.DataFrame(overtrain_data)

    fig_ot = go.Figure()
    fig_ot.add_trace(go.Scatter(
        x=ot_df["Checkpoint"], y=ot_df["Recon Loss"],
        mode="lines+markers",
        name="Reconstruction Loss",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=8),
        yaxis="y"
    ))
    fig_ot.add_trace(go.Scatter(
        x=ot_df["Checkpoint"], y=ot_df["Silhouette"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#22c55e", width=2),
        marker=dict(size=8),
        yaxis="y2"
    ))
    fig_ot.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", family="DM Sans"),
        margin=dict(l=0, r=60, t=20, b=60),
        height=260,
        yaxis=dict(title=dict(text="Reconstruction Loss", font=dict(color="#ef4444")), gridcolor="#e2e8f0",
                   tickfont=dict(color="#ef4444")),
        yaxis2=dict(title=dict(text="Silhouette Score", font=dict(color="#16a34a")), overlaying="y", side="right",
                    tickfont=dict(color="#16a34a")),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=11), bordercolor="#e2e8f0", borderwidth=1)
    )
    st.plotly_chart(fig_ot, use_container_width=True)
    st.caption("Key finding: reconstruction loss continued decreasing beyond epoch 50, but cluster separability (silhouette) peaked at epoch 50 and degraded to 0.0776 by epoch 200. Pretraining should be monitored via downstream metrics, not reconstruction loss alone.")