import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"

COLORS = {
    "fatal": "#b8402a",
    "serious": "#d9822b",
    "slight": "#2b7a78",
    "ink": "#102a43",
    "steel": "#4f6d8a",
    "panel": "#f8fafb",
    "leader": "#2b7a78",
    "regular": "#8ea9c2",
}


st.set_page_config(
    page_title="Accident Severity Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #eef3f8 0%, #e9edf2 100%);
        color: #102a43;
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
    }
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    .dashboard-title {
        margin-bottom: 0.75rem;
        color: #102a43;
        letter-spacing: 0.04em;
        font-size: 2rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .section-title {
        color: #5f748a;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0.65rem 0 0.75rem 0;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #dce5ee;
        box-shadow: 0 4px 14px rgba(26, 47, 74, 0.07);
        padding: 0.95rem 1rem 0.7rem 1rem;
        margin-bottom: 0.35rem;
        border-bottom: 5px solid #4f6d8a;
        min-height: 106px;
    }
    .metric-title {
        color: #6b7f92;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .metric-value {
        color: #233a52;
        font-size: 2.25rem;
        font-weight: 800;
        margin-top: 0.22rem;
        margin-bottom: 0;
        line-height: 1.05;
    }
    .metric-sub {
        color: #7d8ea0;
        font-size: 0.86rem;
        font-weight: 600;
        margin-top: 0.15rem;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #dce5ee;
        box-shadow: 0 6px 16px rgba(26, 47, 74, 0.06);
        border-radius: 14px;
        padding: 0.72rem 0.72rem 0.25rem 0.72rem;
    }
    .panel-title {
        color: #4d6378;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-size: 0.98rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return load_json(PROCESSED_DIR / "preprocessing_metadata.json")


@st.cache_data(show_spinner=False)
def load_test_frame() -> pd.DataFrame:
    x_test = pd.read_pickle(PROCESSED_DIR / "X_test.pkl")
    y_test = pd.read_pickle(PROCESSED_DIR / "y_test.pkl")
    frame = x_test.copy()
    frame["Severity"] = y_test.values
    frame["SevereFlag"] = frame["Severity"].isin(["Fatal", "Serious"]).astype(float)
    return frame


def get_speed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    speed = frame[["Speed_limit", "SevereFlag"]].dropna().copy()
    speed["Speed_limit"] = speed["Speed_limit"].round().astype(int)
    summary = (
        speed.groupby("Speed_limit", as_index=False)
        .agg(Accidents=("SevereFlag", "size"), SevereRate=("SevereFlag", "mean"))
        .sort_values("Speed_limit")
    )
    return summary[summary["Accidents"] > 500]


def get_vehicle_summary(frame: pd.DataFrame) -> pd.DataFrame:
    vehicle = frame[["Vehicle_Type", "SevereFlag"]].dropna().copy()
    vehicle["VehicleCode"] = vehicle["Vehicle_Type"].round().astype(int).astype(str)
    grouped = (
        vehicle.groupby("VehicleCode", as_index=False)
        .agg(Accidents=("SevereFlag", "size"), SevereRate=("SevereFlag", "mean"))
        .sort_values("Accidents", ascending=False)
        .head(10)
    )
    return grouped.sort_values("Accidents", ascending=True)


def get_risk_decile_summary(frame: pd.DataFrame) -> pd.DataFrame:
    risk = frame[["road_risk_score", "SevereFlag"]].dropna().copy()
    risk["RiskDecile"] = pd.qcut(risk["road_risk_score"], q=10, labels=False, duplicates="drop")
    decile = (
        risk.groupby("RiskDecile", as_index=False)
        .agg(Accidents=("SevereFlag", "size"), SevereRate=("SevereFlag", "mean"))
        .sort_values("RiskDecile")
    )
    decile["RiskDecile"] = decile["RiskDecile"].astype(int) + 1
    return decile


def get_engine_capacity_by_severity(frame: pd.DataFrame) -> pd.DataFrame:
    engine = frame[["Engine_Capacity_.CC.", "Severity"]].dropna().copy()
    engine = engine[engine["Engine_Capacity_.CC."] > 0]
    engine = engine[engine["Engine_Capacity_.CC."] <= engine["Engine_Capacity_.CC."].quantile(0.99)]
    bins = pd.cut(engine["Engine_Capacity_.CC."], bins=8)
    grouped = (
        engine.assign(EngineBand=bins.astype(str))
        .groupby(["EngineBand", "Severity"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    return grouped


def get_severity_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    severity_order = ["Fatal", "Serious", "Slight"]
    dist = frame["Severity"].value_counts().rename_axis("Severity").reset_index(name="Count")
    dist["Share"] = dist["Count"] / dist["Count"].sum()
    dist["Severity"] = pd.Categorical(dist["Severity"], categories=severity_order, ordered=True)
    return dist.sort_values("Severity")


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_card(title: str, value: str, subtitle: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-bottom-color:{accent};">
            <div class="metric-title">{title}</div>
            <p class="metric-value">{value}</p>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_open(title: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )


def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def chart_class_donut(distribution: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(distribution)
        .mark_arc(innerRadius=75, outerRadius=130)
        .encode(
            theta=alt.Theta("Share:Q"),
            color=alt.Color(
                "Severity:N",
                scale=alt.Scale(
                    domain=["Fatal", "Serious", "Slight"],
                    range=[COLORS["fatal"], COLORS["serious"], COLORS["slight"]],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[alt.Tooltip("Severity:N"), alt.Tooltip("Share:Q", format=".1%")],
        )
        .properties(height=260)
    )


def chart_speed_volume(speed_summary: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(speed_summary)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Speed_limit:O", title="Speed Limit"),
            y=alt.Y("Accidents:Q", title="Accidents"),
            color=alt.Color(
                "SevereRate:Q",
                scale=alt.Scale(scheme="oranges"),
                title="Severe Rate",
            ),
            tooltip=[
                alt.Tooltip("Speed_limit:O", title="Speed limit"),
                alt.Tooltip("Accidents:Q", format=","),
                alt.Tooltip("SevereRate:Q", format=".1%"),
            ],
        )
        .properties(height=260)
    )


def chart_risk_decile(risk_decile: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(risk_decile)
        .mark_line(point=True, strokeWidth=3, color=COLORS["ink"])
        .encode(
            x=alt.X("RiskDecile:O", title="Road Risk Score Decile"),
            y=alt.Y("SevereRate:Q", title="Severe Rate", scale=alt.Scale(domain=[0, risk_decile["SevereRate"].max() * 1.15])),
            tooltip=[alt.Tooltip("RiskDecile:O", title="Decile"), alt.Tooltip("Accidents:Q", format=","), alt.Tooltip("SevereRate:Q", format=".1%")],
        )
        .properties(height=260)
    )


def chart_vehicle_mix(vehicle_summary: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(vehicle_summary)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("Accidents:Q", title="Accidents"),
            y=alt.Y("VehicleCode:N", sort="-x", title="Vehicle Type Code"),
            color=alt.Color("SevereRate:Q", scale=alt.Scale(scheme="tealblues"), title="Severe Rate"),
            tooltip=[alt.Tooltip("VehicleCode:N", title="Vehicle code"), alt.Tooltip("Accidents:Q", format=","), alt.Tooltip("SevereRate:Q", format=".1%")],
        )
        .properties(height=270)
    )


def chart_engine_mix(engine_by_severity: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(engine_by_severity)
        .mark_bar()
        .encode(
            x=alt.X("EngineBand:N", title="Engine Capacity Band"),
            y=alt.Y("Count:Q", title="Accidents"),
            color=alt.Color(
                "Severity:N",
                scale=alt.Scale(
                    domain=["Fatal", "Serious", "Slight"],
                    range=[COLORS["fatal"], COLORS["serious"], COLORS["slight"]],
                ),
            ),
            tooltip=[alt.Tooltip("EngineBand:N"), alt.Tooltip("Severity:N"), alt.Tooltip("Count:Q", format=",")],
        )
        .properties(height=270)
    )


def chart_top_outliers(metadata: dict) -> alt.Chart:
    outliers = metadata["preprocessing_metadata"]["outlier_stats"].get("columns_affected", [])
    frame = pd.DataFrame(outliers)
    if frame.empty:
        return alt.Chart(pd.DataFrame({"Column": [], "Outlier": []})).mark_bar()
    top = frame.sort_values("pct_outliers", ascending=False).head(7)
    top = top.rename(columns={"column": "Column", "pct_outliers": "Outlier"})
    return (
        alt.Chart(top)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, color="#e76f51")
        .encode(
            x=alt.X("Outlier:Q", title="Outlier %"),
            y=alt.Y("Column:N", sort="-x"),
            tooltip=[alt.Tooltip("Column:N"), alt.Tooltip("Outlier:Q", format=".2f")],
        )
        .properties(height=270)
    )



def main() -> None:
    alt.theme.enable("none")

    metadata = load_metadata()
    frame = load_test_frame()
    severity_dist = get_severity_distribution(frame)
    speed_summary = get_speed_summary(frame)
    risk_decile = get_risk_decile_summary(frame)
    vehicle_summary = get_vehicle_summary(frame)
    engine_by_severity = get_engine_capacity_by_severity(frame)

    train_rows = metadata["data_shapes"]["X_train"][0]
    val_rows = metadata["data_shapes"]["X_val"][0]
    test_rows = metadata["data_shapes"]["X_test"][0]
    total_rows = train_rows + val_rows + test_rows
    fatal_count = int((frame["Severity"] == "Fatal").sum())
    serious_count = int((frame["Severity"] == "Serious").sum())
    severe_share = (fatal_count + serious_count) / len(frame)
    peak_speed = speed_summary.sort_values("Accidents", ascending=False).iloc[0]["Speed_limit"] if not speed_summary.empty else 0

    st.markdown('<div class="dashboard-title">UK Road Accident Severity Dashboard</div>', unsafe_allow_html=True)

    top = st.columns(5)
    with top[0]:
        metric_card("Total Accidents", f"{total_rows:,.0f}", "2005-2017 STATS19", "#3f8fd6")
    with top[1]:
        metric_card("Fatal Accidents", f"{fatal_count:,.0f}", "test sample", COLORS["fatal"])
    with top[2]:
        metric_card("Serious Accidents", f"{serious_count:,.0f}", f"{format_pct(serious_count / len(frame))} of test set", COLORS["serious"])
    with top[3]:
        metric_card("Peak Speed Limit", f"{int(peak_speed)}", "highest crash volume", "#1ea990")
    with top[4]:
        metric_card("Severe Share", format_pct(severe_share), "fatal + serious", COLORS["ink"])

    st.markdown('<div class="section-title">Severity And Temporal Risk Patterns</div>', unsafe_allow_html=True)
    row1 = st.columns([1.0, 1.25, 1.0])

    with row1[0]:
        panel_open("Severity Split")
        st.altair_chart(chart_class_donut(severity_dist), width="stretch")
        panel_close()

    with row1[1]:
        panel_open("Accidents By Speed Limit")
        st.altair_chart(chart_speed_volume(speed_summary), width="stretch")
        panel_close()

    with row1[2]:
        panel_open("Severe Rate By Risk Decile")
        st.altair_chart(chart_risk_decile(risk_decile), width="stretch")
        panel_close()

    st.markdown('<div class="section-title">Risk Factors And Data Quality</div>', unsafe_allow_html=True)
    row2 = st.columns([1.2, 1.0, 1.0])

    with row2[0]:
        panel_open("Top Vehicle Codes By Crash Volume")
        st.altair_chart(chart_vehicle_mix(vehicle_summary), width="stretch")
        panel_close()

    with row2[1]:
        panel_open("Engine Capacity Mix By Severity")
        st.altair_chart(chart_engine_mix(engine_by_severity), width="stretch")
        panel_close()

    with row2[2]:
        panel_open("Top Outlier Pressure")
        st.altair_chart(chart_top_outliers(metadata), width="stretch")
        panel_close()

    eda_image = ROOT / "reports" / "accident_dashboard.png"
    if eda_image.exists():
        st.markdown('<div class="section-title">EDA Notebook Dashboard Snapshot</div>', unsafe_allow_html=True)
        panel_open("Notebook 5-EDA Output")
        st.image(str(eda_image), width="stretch")
        panel_close()


if __name__ == "__main__":
    main()