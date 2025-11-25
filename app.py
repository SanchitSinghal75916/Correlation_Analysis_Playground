# app.py
import streamlit as st


st.set_page_config(page_title="Scotia Deposits Modeler", layout="wide")


st.title("Scotia Deposits Modeler")


st.markdown(
"""
Welcome! This tool helps you **compare multiple regression models** on your CSV quickly—
with **table-driven transformations**, **with vs without zero values in target variable**, and easy **exports**.


### How it works
- **Upload CSV**: Bring your dataset (columns = features; one column is your target).
- **Pick Target & Features**: Choose what you want to predict (target) and which columns might help (features).
- **Transformations Table**: We suggest a transform for each column (e.g., *Log1p*, *Standardize*). You can edit them.
- **Apply & Clean**: We apply transforms, then drop rows with missing values (we’ll tell you how many remain).
- **With vs Without Zeros**: We train and compare models both **including** and **excluding** rows where the **original** target is zero.
- **Train Models**: We run 7 models **in parallel** to speed things up.
- **Feature Relevance**: Get several signals (correlation, mutual info, tree importances, ridge coeffs).
- **Export to Excel**: One click to download the transformations + both result tables.


### Where to start
Open **Modeler** from the left sidebar (or the top “pages” menu) to begin.

Each section includes a short
"what this does" explainer. You can mostly just: **upload → apply → train**.
"""
)


st.divider()


st.markdown(
"""
#### Tips & FAQs (simple)
- **Zeros in target?** We compare models with and without those rows; sometimes zeros represent special cases.
- **Why transforms?** They can stabilize variance and improve model fit.
- **Which model to pick?** Prefer **lower RMSE** and **higher R²** on the test set.
- **My results disappeared?** They won’t now—everything persists during clicks.
"""
)


st.page_link("pages/1_Modeler.py", label="➡️ Go to Modeler", icon=":material/auto_graph:")