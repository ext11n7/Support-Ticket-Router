import joblib
import numpy as np
import pandas as pd
import streamlit as st
import re

MODEL_PATH = "artifacts/model.joblib"

st.set_page_config(page_title="Support Ticket Router", layout="centered")


def looks_like_gibberish(s: str) -> bool:
    s = s.strip()
    if len(s) < 15:
        return True
    letters = re.findall(r"[A-Za-zА-Яа-я]", s)
    return (len(letters) / max(len(s), 1)) < 0.3


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def explain_linear_tfidf(pipe, text: str, class_label: str, top_n: int = 12):
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    X = tfidf.transform([text])
    feat_names = tfidf.get_feature_names_out()

    class_idx = list(clf.classes_).index(class_label)
    w = clf.coef_[class_idx]

    contrib = X.toarray()[0] * w
    nz = np.where(contrib != 0)[0]
    if len(nz) == 0:
        return []

    top = nz[np.argsort(-np.abs(contrib[nz]))][:top_n]
    return [(feat_names[i], float(contrib[i])) for i in top]


# ---------------- UI ----------------
st.title("Support Ticket Router")
st.caption("Predict queue from ticket text (subject + body).")

EX_BILLING = (
    "Subject: Refund request\n"
    "Body: I was charged twice for the same order. Please refund the extra payment. "
    "Invoice number: INV-1042."
)
EX_TECH = (
    "Subject: App crashes on startup\n"
    "Body: After the last update, the mobile app crashes immediately when I open it. "
    "Device: iPhone 13, iOS 17.2. Please advise."
)
EX_ACCOUNT = (
    "Subject: Can't log in\n"
    "Body: I forgot my password and the reset email never arrives. "
    "My account email is user@example.com."
)

def set_text(v: str):
    st.session_state.ticket_text = v

cols = st.columns(3)
cols[0].button("Example: Billing", on_click=set_text, args=(EX_BILLING,))
cols[1].button("Example: Tech", on_click=set_text, args=(EX_TECH,))
cols[2].button("Example: Account", on_click=set_text, args=(EX_ACCOUNT,))

st.session_state.setdefault("ticket_text", "")
text = st.text_area(
    "Ticket text",
    key="ticket_text",
    height=180,
    placeholder="Paste subject + body here..."
)

top_k = st.slider("Top-K predictions", 1, 5, 3)

if st.button("Predict", type="primary"):
    if not text or len(text.strip()) < 5:
        st.error("Please enter a longer text.")
        st.stop()

    if looks_like_gibberish(text):
        st.warning("Text looks too short/unnatural. Add more details (error, action, product, account).")
        st.stop()

    model = load_model()
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    idx = np.argsort(-proba)[:top_k]

    best_label = classes[idx[0]]
    best_p = float(proba[idx[0]])

    st.subheader(f"Predicted queue: {best_label}")

    if best_p >= 0.70:
        conf_label = "High"
    elif best_p >= 0.45:
        conf_label = "Medium"
    else:
        conf_label = "Low"

    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{best_p:.2f}", border=True)
    c2.metric("Confidence level", conf_label, border=True)

    df_top = pd.DataFrame({
        "Queue": [classes[i] for i in idx],
        "Probability": [float(proba[i]) for i in idx],
    }).sort_values("Probability", ascending=False)

    df_top["Probability"] = df_top["Probability"].map(lambda x: round(x, 3))

    st.markdown("### Top predictions")
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    st.markdown("### Probabilities (bar chart)")
    st.bar_chart(df_top.set_index("Queue")["Probability"], use_container_width=True)

    with st.expander("Why this prediction? (top word contributions)"):
        items = explain_linear_tfidf(model, text, best_label, top_n=12)
        if not items:
            st.write("No informative tokens found in the vocabulary.")
        else:
            for term, score in items:
                st.write(f"- {term}: {score:+.4f}")
