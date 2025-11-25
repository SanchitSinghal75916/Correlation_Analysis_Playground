# utils/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def small_text_heatmap(corr: pd.DataFrame, title: str):
    fig, ax = plt.subplots(
        figsize=(min(0.5 * len(corr.columns) + 4, 12), min(0.5 * len(corr.columns) + 4, 12))
    )
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", ax=ax, annot_kws={"size": 6})
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', labelsize=7, rotation=60)
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

def barplot_metric(combo_df: pd.DataFrame, metric: str, title: str):
    # Drop NaNs and ensure metric exists
    if metric not in combo_df.columns:
        st.info(f"Metric '{metric}' not found.")
        return
    data = combo_df.dropna(subset=[metric]).copy()
    if data.empty:
        st.info(f"No finished models yet for '{metric}'.")
        return

    # Smaller, nicer plot
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.barplot(data=data, x="Model", y=metric, hue="Scenario", ax=ax)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(title="", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
