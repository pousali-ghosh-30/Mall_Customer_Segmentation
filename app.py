from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


APP_DIRECTORY = Path(__file__).resolve().parent
INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"

SEGMENT_NAMES = {
    0: "Steady shoppers",
    1: "High-value enthusiasts",
    2: "Careful big spenders",
    3: "Affluent savers",
    4: "Budget-minded browsers",
}
SEGMENT_DESCRIPTIONS = {
    0: "Typical income and spending",
    1: "Higher income and higher spending",
    2: "Lower income and higher spending",
    3: "Higher income and lower spending",
    4: "Lower income and lower spending",
}


@st.cache_resource
def load_model():
    """Load the saved clustering model and its matching scaler."""
    model = joblib.load(APP_DIRECTORY / "kmeans_model.pkl")
    scaler = joblib.load(APP_DIRECTORY / "scaler.pkl")
    return model, scaler


@st.cache_data
def load_customers():
    """Load the customer records bundled with this app."""
    return pd.read_csv(APP_DIRECTORY / "customers_with_clusters.csv")


def main():
    st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
    st.title("Get to know your mall customers")
    st.write(
        "Explore customer groups based on annual income and spending habits, "
        "then see which group a new customer most resembles."
    )

    model, scaler = load_model()
    customers = load_customers()
    customer_features = customers[[INCOME, SPENDING]]
    customers["Cluster"] = model.predict(scaler.transform(customer_features))
    customers["Segment"] = customers["Cluster"].map(SEGMENT_NAMES)

    with st.sidebar:
        st.header("Try a customer profile")
        st.caption("Adjust the values, then see which group they fit best.")
        new_income = st.slider("Annual income (thousands of dollars)", 10, 150, 60)
        new_spending = st.slider("Spending score", 1, 100, 50)
        find_segment = st.button("Find their segment", type="primary")

    predicted_cluster = None
    if find_segment:
        new_customer = pd.DataFrame(
            [[new_income, new_spending]], columns=[INCOME, SPENDING]
        )
        new_customer = scaler.transform(new_customer)
        predicted_cluster = int(model.predict(new_customer)[0])
        segment_name = SEGMENT_NAMES[predicted_cluster]
        st.sidebar.success(
            f"This customer fits **{segment_name}** "
            f"(Cluster {predicted_cluster})."
        )
        st.sidebar.caption(SEGMENT_DESCRIPTIONS[predicted_cluster])

    total_customers = len(customers)
    average_spending = customers[SPENDING].mean()
    metric_columns = st.columns(3)
    metric_columns[0].metric("Customers analyzed", f"{total_customers:,}")
    metric_columns[1].metric("Customer groups", len(SEGMENT_NAMES))
    metric_columns[2].metric("Average spending score", f"{average_spending:.1f}")

    chart_tab, summary_tab, data_tab = st.tabs(
        ["Customer map", "Group summary", "Customer records"]
    )

    with chart_tab:
        st.subheader("Where each group sits")
        st.caption(
            "Each dot represents a customer. Groups are based on income and "
            "spending patterns."
        )
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            data=customers,
            x=INCOME,
            y=SPENDING,
            hue="Segment",
            hue_order=list(SEGMENT_NAMES.values()),
            palette="tab10",
            s=70,
            ax=axis,
        )
        axis.set_xlabel("Annual income (thousands of dollars)")
        axis.set_ylabel("Spending score (1-100)")
        if predicted_cluster is not None:
            axis.scatter(
                new_income,
                new_spending,
                color="black",
                marker="*",
                s=300,
                label="New customer",
                zorder=3,
            )
            axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    with summary_tab:
        st.subheader("What makes each group different?")
        segment_summary = (
            customers.groupby(["Cluster", "Segment"], sort=True)
            .agg(
                Customers=("CustomerID", "count"),
                **{
                    "Average income (k$)": (INCOME, "mean"),
                    "Average spending score": (SPENDING, "mean"),
                },
            )
            .round(1)
            .reset_index()
        )
        segment_summary["Typical profile"] = segment_summary["Cluster"].map(
            SEGMENT_DESCRIPTIONS
        )
        st.dataframe(
            segment_summary.drop(columns="Cluster"),
            hide_index=True,
            width="stretch",
        )

    with data_tab:
        st.subheader("Browse customer records")
        available_segments = list(SEGMENT_NAMES.values())
        selected_segments = st.multiselect(
            "Show groups",
            options=available_segments,
            default=available_segments,
        )
        visible_customers = customers[
            customers["Segment"].isin(selected_segments)
        ]
        st.caption(f"Showing {len(visible_customers):,} of {total_customers:,} customers")
        st.dataframe(
            visible_customers.drop(columns="Cluster"),
            hide_index=True,
            width="stretch",
        )


if __name__ == "__main__":
    main()
