"""
XAI Visualizer — Vietnamese Audio Deepfake Detection
=====================================================
Tạo các biểu đồ matplotlib từ SHAP results để nhúng vào Streamlit.

Tất cả hàm trả về matplotlib.figure.Figure để Streamlit render
bằng st.pyplot(fig).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — bắt buộc khi chạy trên Streamlit server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Optional

# ---------------------------------------------------------------------------
# Màu sắc nhất quán
# ---------------------------------------------------------------------------
COLOR_FAKE  = "#EF4444"   # đỏ — tăng khả năng AI/fake
COLOR_REAL  = "#22C55E"   # xanh lá — giảm khả năng AI/fake
COLOR_NEU   = "#6B7280"   # xám trung tính
BG_DARK     = "#0F172A"   # nền tối
BG_CARD     = "#1E293B"   # card
TEXT_LIGHT  = "#F1F5F9"
GRID_COLOR  = "#334155"

def _apply_dark_style(fig, axes=None):
    """Áp dụng theme tối cho toàn bộ figure."""
    fig.patch.set_facecolor(BG_DARK)
    if axes is None:
        axes = fig.get_axes()
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=TEXT_LIGHT, labelsize=9)
        ax.xaxis.label.set_color(TEXT_LIGHT)
        ax.yaxis.label.set_color(TEXT_LIGHT)
        if ax.get_title():
            ax.title.set_color(TEXT_LIGHT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)


# ---------------------------------------------------------------------------
# 1. Waterfall chart — top-k SHAP values cho 1 model
# ---------------------------------------------------------------------------

def plot_waterfall(model_result: dict, model_name: str,
                   top_k: int = 12) -> plt.Figure:
    """
    Waterfall chart hiển thị từng feature đóng góp vào kết quả như thế nào.
    Đỏ = đẩy về phía "AI/Fake", Xanh = đẩy về phía "Real".
    """
    top = model_result["top_k"][:top_k]
    labels  = [t["feature"] for t in top]
    values  = [t["shap_value"] for t in top]
    base    = model_result["base_value"]

    # Rút gọn tên feature dài
    labels = [l[:35] + "…" if len(l) > 35 else l for l in labels]

    fig, ax = plt.subplots(figsize=(9, max(4, top_k * 0.55)))

    colors = [COLOR_FAKE if v > 0 else COLOR_REAL for v in values]
    bars = ax.barh(range(len(values)), values, color=colors,
                   edgecolor="none", height=0.65)

    # Giá trị text
    for i, (v, bar) in enumerate(zip(values, bars)):
        ha = "left" if v >= 0 else "right"
        offset = 0.003 * (max(values) - min(values)) if values else 0
        ax.text(v + (offset if v >= 0 else -offset), i,
                f"{v:+.4f}", va="center", ha=ha,
                fontsize=8, color=TEXT_LIGHT)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color=TEXT_LIGHT, linewidth=0.8, alpha=0.6)

    # Base value marker
    ax.axvline(base, color="#FBBF24", linewidth=1.2,
               linestyle="--", alpha=0.8, label=f"Base = {base:.3f}")

    ax.set_xlabel("SHAP value (đóng góp vào xác suất AI)")
    ax.set_title(f"🔬 Giải thích — {model_name}", fontsize=11, pad=10)

    legend_patches = [
        mpatches.Patch(color=COLOR_FAKE,  label="Tăng nghi ngờ AI 🔴"),
        mpatches.Patch(color=COLOR_REAL,  label="Giảm nghi ngờ AI 🟢"),
        mpatches.Patch(color="#FBBF24",   label=f"Base value: {base:.3f}"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor=BG_CARD, edgecolor=GRID_COLOR,
              labelcolor=TEXT_LIGHT, fontsize=8)

    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Ensemble Radar / Pie chart — tầm quan trọng từng model
# ---------------------------------------------------------------------------

def plot_ensemble_weights(summary: dict) -> plt.Figure:
    """
    Biểu đồ donut thể hiện mức độ ảnh hưởng của từng model
    trong quyết định ensemble.
    """
    weights = summary.get("model_weights", {})
    labels  = list(weights.keys())
    
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    
    if not labels:
        ax.text(0.5, 0.5, "Không có dữ liệu đóng góp mô hình",
                ha="center", va="center", color=TEXT_LIGHT, fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        return fig

    total_weight = sum(weights.values())
    if total_weight < 1e-6:
        pie_sizes = [1.0] * len(labels)
        display_sizes = [0.0] * len(labels)
    else:
        pie_sizes = [weights[k] * 100 for k in labels]
        display_sizes = pie_sizes

    palette = ["#6366F1", "#06B6D4", "#F59E0B", "#EF4444", "#10B981"]

    wedges, texts, autotexts = ax.pie(
        pie_sizes,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        colors=palette[:len(labels)],
        pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor=BG_DARK, linewidth=2),
    )

    for at, ds in zip(autotexts, display_sizes):
        at.set_text(f"{ds:.1f}%")
        at.set_color(TEXT_LIGHT)
        at.set_fontsize(9)

    # Legend ngoài
    ax.legend(
        wedges, [f"{l}  ({s:.1f}%)" for l, s in zip(labels, display_sizes)],
        loc="lower center", bbox_to_anchor=(0.5, -0.18),
        ncol=2, facecolor=BG_CARD, edgecolor=GRID_COLOR,
        labelcolor=TEXT_LIGHT, fontsize=8.5
    )

    dominant = summary.get("dominant_model", "N/A")
    ax.set_title(f"⚖️ Tầm quan trọng từng Model\n(Model chi phối: {dominant})",
                 fontsize=10, color=TEXT_LIGHT, pad=14)

    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# 3. Confidence gauge — đồng hồ đo xác suất deepfake
# ---------------------------------------------------------------------------

def plot_confidence_gauge(prob_fake: float) -> plt.Figure:
    """
    Bán vòng tròn (gauge) thể hiện xác suất deepfake.
    """
    fig, ax = plt.subplots(figsize=(4.5, 2.5),
                           subplot_kw={"projection": "polar"})

    # Nền bán vòng: từ π đến 0 (nửa trên)
    theta = np.linspace(np.pi, 0, 200)

    # Gradient nền (xanh → đỏ)
    n_seg = 100
    for i in range(n_seg):
        t0 = np.pi - i * np.pi / n_seg
        t1 = np.pi - (i + 1) * np.pi / n_seg
        r  = 0.85
        ratio = i / n_seg
        color = (ratio, 1 - ratio * 0.8, 0.2)  # RGB gradient
        ax.fill_between([t0, t1], [0.65, 0.65], [r, r], color=color, alpha=0.35)

    # Kim chỉ
    needle_angle = np.pi - prob_fake * np.pi
    ax.plot([needle_angle, needle_angle], [0, 0.75],
            color=TEXT_LIGHT, linewidth=2.5, solid_capstyle="round")
    ax.plot(needle_angle, 0.75, "o",
            color=COLOR_FAKE if prob_fake >= 0.5 else COLOR_REAL,
            markersize=7, zorder=5)

    # Text xác suất trung tâm (đẩy lên cao hơn để không đè chữ)
    ax.text(np.pi / 2, 0.40, f"{prob_fake*100:.1f}%",
            ha="center", va="center",
            fontsize=20, fontweight="bold",
            color=COLOR_FAKE if prob_fake >= 0.5 else COLOR_REAL,
            transform=ax.transData)

    label = "AI / DEEPFAKE" if prob_fake >= 0.5 else "GIỌNG THẬT"
    ax.text(np.pi / 2, -0.05, label,
            ha="center", va="center", fontsize=12,
            color=TEXT_LIGHT, transform=ax.transData)

    # Nhãn góc
    for angle, lbl in [(np.pi, "0%"), (np.pi/2, "50%"), (0, "100%")]:
        ax.text(angle, 0.95, lbl, ha="center", va="center",
                fontsize=8, color=TEXT_LIGHT, transform=ax.transData)

    ax.set_ylim(0, 1)
    ax.set_theta_zero_location("N")
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis("off")
    ax.set_facecolor(BG_DARK)
    fig.patch.set_facecolor(BG_DARK)
    ax.set_title("Chỉ số Deepfake", color=TEXT_LIGHT,
                 fontsize=10, pad=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Feature-group bar chart — MFCC group importance (aggregate)
# ---------------------------------------------------------------------------

def plot_mfcc_group_importance(model_result: dict,
                                model_name: str) -> plt.Figure:
    """
    Nhóm 40 SHAP values theo dải MFCC (1-13, 14-26, 27-40)
    và hiển thị mức đóng góp trung bình của từng nhóm.
    """
    sv   = model_result["shap_values"]
    # Chỉ hỗ trợ model có đúng 40 features
    if len(sv) != 40:
        return None

    groups = {
        "MFCC 1-13\n(Formant)":   sv[0:13],
        "MFCC 14-26\n(Spectral)": sv[13:26],
        "MFCC 27-40\n(Fine)":     sv[26:40],
    }

    labels   = list(groups.keys())
    pos_vals = [np.sum(np.maximum(v, 0)) for v in groups.values()]
    neg_vals = [np.sum(np.minimum(v, 0)) for v in groups.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(x - width/2, pos_vals, width, label="→ Fake",
           color=COLOR_FAKE, alpha=0.85, edgecolor="none")
    ax.bar(x + width/2, neg_vals, width, label="→ Real",
           color=COLOR_REAL, alpha=0.85, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Tổng SHAP trong nhóm")
    ax.set_title(f"Nhóm MFCC — {model_name}", fontsize=10, pad=8)
    ax.axhline(0, color=TEXT_LIGHT, linewidth=0.6, alpha=0.5)
    ax.legend(facecolor=BG_CARD, edgecolor=GRID_COLOR,
              labelcolor=TEXT_LIGHT, fontsize=9)

    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Heatmap — SHAP theo từng model × top features
# ---------------------------------------------------------------------------

def plot_cross_model_heatmap(results: dict, top_k: int = 10) -> plt.Figure:
    """
    Heatmap so sánh top features (từ XGBoost) có SHAP như thế nào
    ở các model khác nhau (chỉ cho model có ≤ 40 features để so sánh được).
    """
    # Lấy model 40-feature
    models_40 = {k: v for k, v in results.items()
                 if not k.startswith("_") and len(v["shap_values"]) == 40}

    if len(models_40) < 2:
        return None

    # Lấy top-k features từ model đầu tiên
    first_key = list(models_40.keys())[0]
    top_idx = np.argsort(
        np.abs(models_40[first_key]["shap_values"]))[::-1][:top_k]
    feat_labels = [
        models_40[first_key]["feature_names"][i][:25] for i in top_idx
    ]

    matrix = np.array([
        models_40[k]["shap_values"][top_idx]
        for k in models_40
    ])

    fig, ax = plt.subplots(figsize=(max(7, top_k * 0.7), 4))

    vmax = np.abs(matrix).max() or 1e-6
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(top_k))
    ax.set_xticklabels(feat_labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(models_40)))
    ax.set_yticklabels(list(models_40.keys()), fontsize=9)
    ax.set_title(f"So sánh SHAP xuyên-model (Top-{top_k} features)",
                 fontsize=10, pad=8, color=TEXT_LIGHT)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=TEXT_LIGHT)
    cbar.outline.set_edgecolor(GRID_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_LIGHT, fontsize=8)

    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig
