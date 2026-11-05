import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

st.set_page_config(page_title="Election Data Mining Agent", page_icon="🤖", layout="wide")
st.title("🤖 Data Mining Agent – Indian Election Dataset")
st.write("Explore election statistics using association rules, ID3, C4.5, and clustering.")

DEFAULT_FILE = "dataset2.csv"
uploaded = st.file_uploader("Upload a different CSV dataset (optional)", type=["csv"])
try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source_name = uploaded.name
    else:
        df = pd.read_csv(DEFAULT_FILE)
        source_name = DEFAULT_FILE
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

# Feature engineering useful for classification targets and association mining.
def add_levels(frame):
    out = frame.copy()
    numeric = out.select_dtypes(include=np.number).columns
    for col in numeric:
        if col != "Year" and out[col].nunique() > 3:
            try:
                out[f"{col}_Level"] = pd.qcut(out[col], q=3, labels=["Low", "Medium", "High"], duplicates="drop").astype(str)
            except Exception:
                pass
    return out

engineered = add_levels(df)

st.success(f"Loaded **{source_name}**")
a, b, c = st.columns(3)
a.metric("Records", f"{len(df):,}")
b.metric("Columns", f"{len(df.columns):,}")
c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
with st.expander("Dataset preview and column information", expanded=True):
    st.dataframe(df.head(15), use_container_width=True)
    st.dataframe(pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(x) for x in df.dtypes],
        "Missing values": [int(df[x].isna().sum()) for x in df.columns],
        "Unique values": [int(df[x].nunique()) for x in df.columns]
    }), use_container_width=True)

tab_assoc, tab_tree, tab_cluster = st.tabs(["Apriori & FP-Growth", "ID3 & C4.5 Decision Trees", "Clustering"])

# ---------- Association rules ----------
with tab_assoc:
    st.header("Association Rule Mining")
    st.write("Each state-year record is converted into items such as `State=Andhra Pradesh`, `Year=2004`, and `Female_Winner_Level=Low`.")
    st.caption("This row-to-item conversion makes the election dataset suitable for Apriori and FP-Growth even though it is not a market-basket dataset.")
    assoc_cols = st.multiselect(
        "Columns to convert into transaction items",
        engineered.columns.tolist(),
        default=[x for x in ["Year", "State", "Total_Seats_Level", "Male_Candidate_Level",
                             "Female_Candidate_Level", "Male_Winner_Level", "Female_Winner_Level"]
                 if x in engineered.columns]
    )
    support = st.slider("Minimum support", 0.01, 0.80, 0.08, 0.01)
    confidence = st.slider("Minimum confidence", 0.05, 1.0, 0.50, 0.05)
    algorithms = st.multiselect("Algorithms", ["Apriori", "FP-Growth"], default=["Apriori", "FP-Growth"])
    if st.button("Run association mining", type="primary"):
        try:
            if not assoc_cols:
                raise ValueError("Select at least one column.")
            tx_rows = []
            for _, row in engineered[assoc_cols].fillna("Unknown").iterrows():
                items = []
                for col in assoc_cols:
                    value = row[col]
                    if col == "Year":
                        value = str(int(value)) if pd.notna(value) else "Unknown"
                    items.append(f"{col}={value}")
                tx_rows.append(items)
            all_items = sorted({item for trans in tx_rows for item in trans})
            basket = pd.DataFrame(False, index=range(len(tx_rows)), columns=all_items)
            for i, trans in enumerate(tx_rows):
                basket.loc[i, trans] = True
            st.write(f"Converted **{len(tx_rows)} records** into **{len(all_items)} distinct items**.")
            for algo in algorithms:
                with st.expander(f"{algo} results", expanded=True):
                    freq = (apriori(basket, min_support=support, use_colnames=True)
                            if algo == "Apriori" else
                            fpgrowth(basket, min_support=support, use_colnames=True))
                    if freq.empty:
                        st.warning("No frequent itemsets found. Lower the minimum support.")
                        continue
                    freq["Items"] = freq["itemsets"].apply(lambda s: ", ".join(sorted(s)))
                    st.subheader("Frequent itemsets")
                    st.dataframe(freq[["Items", "support"]].sort_values("support", ascending=False), use_container_width=True)
                    try:
                        rules = association_rules(freq, metric="confidence", min_threshold=confidence)
                        if rules.empty:
                            st.info("No association rules met the confidence threshold.")
                        else:
                            rules["IF"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
                            rules["THEN"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
                            show = rules[["IF", "THEN", "support", "confidence", "lift"]].sort_values(
                                ["confidence", "lift"], ascending=False)
                            st.subheader("Association rules")
                            st.dataframe(show, use_container_width=True)
                    except Exception as exc:
                        st.warning(f"Rule generation issue: {exc}")
        except Exception as exc:
            st.error(f"Association mining error: {exc}")

# ---------- Custom ID3 / C4.5 tree ----------
def entropy(y):
    counts = np.array(list(Counter(y).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())

def split_score(frame, target, feature, method):
    y = frame[target].astype(str)
    parent = entropy(y)
    vals = frame[feature]
    best = None
    if pd.api.types.is_numeric_dtype(vals) and vals.nunique() > 1:
        unique = np.sort(vals.dropna().unique())
        if len(unique) > 30:
            unique = np.unique(np.quantile(unique, np.linspace(0, 1, 31)))
        thresholds = (unique[:-1] + unique[1:]) / 2
        candidates = [("numeric", float(t)) for t in thresholds]
    else:
        candidates = [("categorical", None)]
    for kind, threshold in candidates:
        if kind == "numeric":
            left = frame[vals <= threshold]
            right = frame[vals > threshold]
            parts = [left, right]
            split_info = 0.0
            for part in parts:
                if len(part):
                    p = len(part) / len(frame)
                    split_info -= p * np.log2(p)
        else:
            parts = [g for _, g in frame.groupby(feature, dropna=False)]
            split_info = 0.0
            for part in parts:
                p = len(part) / len(frame)
                split_info -= p * np.log2(p)
        if len(parts) < 2:
            continue
        remainder = sum((len(part) / len(frame)) * entropy(part[target].astype(str)) for part in parts if len(part))
        gain = parent - remainder
        score = gain if method == "ID3" else (gain / split_info if split_info > 0 else 0)
        if best is None or score > best["score"]:
            best = {"kind": kind, "threshold": threshold, "gain": gain, "score": score,
                    "parts": parts, "split_info": split_info}
    return best

def build_tree(frame, features, target, method, depth, max_depth, min_samples):
    labels = frame[target].astype(str)
    majority = labels.value_counts().idxmax()
    if labels.nunique() == 1 or depth >= max_depth or len(frame) < min_samples or not features:
        return {"leaf": majority, "samples": len(frame)}
    best_feature, best_split = None, None
    for feature in features:
        candidate = split_score(frame, target, feature, method)
        if candidate and (best_split is None or candidate["score"] > best_split["score"]):
            best_feature, best_split = feature, candidate
    if best_split is None or best_split["gain"] <= 1e-10:
        return {"leaf": majority, "samples": len(frame)}
    node = {"feature": best_feature, "kind": best_split["kind"], "threshold": best_split["threshold"],
            "majority": majority, "gain": best_split["gain"], "score": best_split["score"], "children": {}}
    if best_split["kind"] == "numeric":
        t = best_split["threshold"]
        node["children"]["<="] = build_tree(best_split["parts"][0], features, target, method, depth+1, max_depth, min_samples)
        node["children"][">"] = build_tree(best_split["parts"][1], features, target, method, depth+1, max_depth, min_samples)
    else:
        for value, part in frame.groupby(best_feature, dropna=False):
            remaining = [f for f in features if f != best_feature]
            node["children"][str(value)] = build_tree(part, remaining, target, method, depth+1, max_depth, min_samples)
    return node

def predict_one(node, row):
    if "leaf" in node:
        return node["leaf"]
    feature = node["feature"]
    if node["kind"] == "numeric":
        branch = "<=" if float(row[feature]) <= node["threshold"] else ">"
    else:
        branch = str(row[feature])
    child = node["children"].get(branch)
    return predict_one(child, row) if child else node["majority"]

def tree_lines(node, prefix=""):
    if "leaf" in node:
        return [prefix + f"→ Predict {node['leaf']} (n={node['samples']})"]
    f = node["feature"]
    lines = [prefix + f"[{f}] (gain={node['gain']:.3f}, score={node['score']:.3f})"]
    for branch, child in node["children"].items():
        label = f"{f} {branch} {node['threshold']:.3f}" if node["kind"] == "numeric" and branch in ("<=", ">") else f"{f} = {branch}"
        lines.extend(tree_lines(child, prefix + "  " + label + " : "))
    return lines

with tab_tree:
    st.header("Decision Tree Classification")
    st.write("The application implements ID3 using information gain and C4.5 using gain ratio, with numeric split thresholds.")
    target_options = engineered.columns.tolist()
    suggested = "Female_Winner_Level" if "Female_Winner_Level" in target_options else target_options[-1]
    target = st.selectbox("Target / class column", target_options, index=target_options.index(suggested))
    feature_options = [x for x in target_options if x != target]
    default_features = [x for x in ["Year", "State", "Total_Seats", "Male_Candidate",
                                    "Female_Candidate", "Male_Winner", "Female_Winner"] if x in feature_options]
    features = st.multiselect("Input features", feature_options, default=default_features)
    method = st.radio("Algorithm", ["ID3", "C4.5"], horizontal=True)
    max_depth = st.slider("Maximum tree depth", 1, 8, 4)
    test_fraction = st.slider("Test set size", 0.10, 0.40, 0.25, 0.05)
    min_samples = st.slider("Minimum records to split", 2, 20, 5)
    if st.button("Train decision tree", type="primary"):
        try:
            if not features:
                raise ValueError("Select at least one input feature.")
            work = engineered[features + [target]].dropna().copy()
            work[target] = work[target].astype(str)
            for col in features:
                if not pd.api.types.is_numeric_dtype(work[col]):
                    work[col] = work[col].astype(str)
            if work[target].nunique() < 2:
                raise ValueError("The target must contain at least two classes.")
            strat = work[target] if work[target].value_counts().min() >= 2 else None
            train, test = train_test_split(work, test_size=test_fraction, random_state=42, stratify=strat)
            tree = build_tree(train, features, target, method, 0, max_depth, min_samples)
            preds = [predict_one(tree, row) for _, row in test.iterrows()]
            actual = test[target].tolist()
            st.metric("Test accuracy", f"{accuracy_score(actual, preds):.3f}")
            st.text(classification_report(actual, preds, zero_division=0))
            st.subheader(f"{method} tree rules")
            st.code("\n".join(tree_lines(tree)), language="text")
            st.download_button("Download tree rules", "\n".join(tree_lines(tree)), f"{method.lower()}_tree_rules.txt")
            st.caption("ID3 selects splits by information gain. C4.5 selects splits by gain ratio (information gain divided by split information). Numeric features are split using candidate thresholds.")
        except Exception as exc:
            st.error(f"Decision-tree error: {exc}")

# ---------- Clustering ----------
with tab_cluster:
    st.header("Clustering Analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    chosen = st.multiselect("Numeric features for clustering",
                            numeric_cols,
                            default=[x for x in ["Total_Seats", "Male_Candidate", "Female_Candidate",
                                                 "Male_Winner", "Female_Winner"] if x in numeric_cols])
    algo = st.selectbox("Clustering algorithm", ["K-Means", "Agglomerative Clustering", "DBSCAN"])
    if algo == "DBSCAN":
        eps = st.slider("DBSCAN epsilon", 0.1, 5.0, 0.8, 0.1)
        min_samples_db = st.slider("Minimum samples", 2, 20, 5)
        k = None
    else:
        k = st.slider("Number of clusters", 2, 10, 3)
        eps = min_samples_db = None
    if st.button("Run clustering", type="primary"):
        try:
            if not chosen:
                raise ValueError("Select at least one numeric feature.")
            clean = df[chosen].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) < 3:
                raise ValueError("Not enough complete records.")
            scaled = StandardScaler().fit_transform(clean)
            if algo == "K-Means":
                labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
            elif algo == "Agglomerative Clustering":
                labels = AgglomerativeClustering(n_clusters=k).fit_predict(scaled)
            else:
                labels = DBSCAN(eps=eps, min_samples=min_samples_db).fit_predict(scaled)
            result = df.loc[clean.index].copy()
            result["Cluster"] = labels
            st.dataframe(result, use_container_width=True)
            st.subheader("Records per cluster")
            st.bar_chart(pd.Series(labels).value_counts().sort_index())
            unique = set(labels)
            if 1 < len(unique) < len(labels):
                st.metric("Silhouette score", f"{silhouette_score(scaled, labels):.3f}")
            else:
                st.info("Silhouette score is not available for this result.")
            if scaled.shape[1] >= 2:
                points = PCA(n_components=2).fit_transform(scaled)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(points[:, 0], points[:, 1], c=labels)
                ax.set_title("Cluster plot (PCA projection)")
                ax.set_xlabel("PCA 1")
                ax.set_ylabel("PCA 2")
                st.pyplot(fig)
            st.download_button("Download clustered dataset", result.to_csv(index=False).encode("utf-8"),
                               "election_data_with_clusters.csv", "text/csv")
        except Exception as exc:
            st.error(f"Clustering error: {exc}")

st.divider()
st.caption("Dataset: election statistics by Indian state and year. Results are exploratory; association rules depend on the selected discretisation and support threshold.")
