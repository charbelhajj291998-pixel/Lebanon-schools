"""Lebanon's schools: public and private education, town by town.

MSBA 325 - Streamlit interactivity activity
Author: Charbel Youssef El Hajj

Data: Educational Resources - Lebanon 2023 (Impact Open Data, served through
AUB's CODEC linked-data platform). One row per town or village outside Beirut.

Run locally:  streamlit run streamlit_app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Lebanon's schools: public vs private",
    page_icon="🏫",
    layout="wide",
)

# The CSV sits next to this file, so the path works locally and on Streamlit Cloud.
DATA_FILE = Path(__file__).parent / "dataset.csv"

ALL_LEBANON = "All Lebanon"
ALL_DISTRICTS = "All districts"

SOURCE_COLUMNS = {
    "Type and size of educational resources - public schools": "public",
    "Type and size of educational resources - private schools": "private",
    "Type and size of educational resources - universities": "universities",
    "Type and size of educational resources - vocational institute": "vocational",
    "Public school coverage index (number of schools per citizen)": "coverage_index",
}

# The source's refArea column mixes two administrative levels. Seven labels name a
# governorate instead of a district; in every case those towns belong to the one
# district of that governorate that never appears by name (checked on town names:
# Beit Ed-Dine and Damour are in Chouf, Amioun and Kousba in Koura, Rachaya El Wadi
# in Rashaya, Roum and Bkassine in Jezzine, Arnoun in Nabatieh, Aarsal in Baalbek).
GOVERNORATE_LABELS = {
    "Akkar_Governorate": ("Akkar", "Akkar"),  # a one-district governorate
    "Mount_Lebanon_Governorate": ("Mount Lebanon", "Chouf"),
    "North_Governorate": ("North", "Koura"),
    "Beqaa_Governorate": ("Beqaa", "Rashaya"),
    "South_Governorate": ("South", "Jezzine"),
    "Nabatieh_Governorate": ("Nabatieh", "Nabatieh"),
    "Baalbek-Hermel_Governorate": ("Baalbek-Hermel", "Baalbek"),
}
DISTRICT_TO_GOVERNORATE = {
    "Aley": "Mount Lebanon", "Baabda": "Mount Lebanon", "Byblos": "Mount Lebanon",
    "Keserwan": "Mount Lebanon", "Matn": "Mount Lebanon",
    "Batroun": "North", "Bsharri": "North", "Miniyeh–Danniyeh": "North",
    "Tripoli": "North", "Zgharta": "North",
    "Western Beqaa": "Beqaa", "Zahlé": "Beqaa",
    "Sidon": "South", "Tyre": "South",
    "Bint Jbeil": "Nabatieh", "Hasbaya": "Nabatieh", "Marjeyoun": "Nabatieh",
    "Hermel": "Baalbek-Hermel",
}
TOWN_DISPLAY_NAMES = {"Trablous": "Tripoli (Trablous)"}

LEAN_PUBLIC, LEAN_EQUAL, LEAN_PRIVATE, LEAN_NONE = (
    "More public", "Equal", "More private", "No school",
)


# --------------------------------------------------------------------------- #
# Colours: one meaning per colour across the whole page.
# Blue = public-leaning, orange = private-leaning, grey = context.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Palette:
    public: str
    private: str
    neutral: str
    context_bar: str
    context_dot: str
    ink: str
    ink_2: str
    muted: str
    grid: str
    baseline: str
    card: str
    card_border: str
    ring: str


LIGHT = Palette(
    public="#2a78d6", private="#eb6834", neutral="#8a8983",
    context_bar="#dcdbd5", context_dot="#dedcd6",
    ink="#0b0b0b", ink_2="#52514e", muted="#77766f",
    grid="#e8e7e1", baseline="#c3c2b7",
    card="#f7f7f5", card_border="rgba(11,11,11,0.10)", ring="#ffffff",
)
DARK = Palette(
    public="#3987e5", private="#d95926", neutral="#a3a29a",
    context_bar="#3a3a37", context_dot="#3d3d39",
    ink="#ffffff", ink_2="#c3c2b7", muted="#9a998f",
    grid="#2c2c2a", baseline="#4a4a46",
    card="#1a1c23", card_border="rgba(255,255,255,0.10)", ring="#0e1117",
)
PAL = DARK if st.context.theme.type == "dark" else LIGHT
FONT = '"Source Sans Pro", "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif'


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def repair_text(value: str) -> str:
    """Undo double encoding (UTF-8 bytes read as Latin-1): 'ZahlÃ©' -> 'Zahlé'."""
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def parse_area(label: str) -> tuple[str, str, bool]:
    """Return (governorate, district, district_was_inferred) for a refArea label."""
    if label in GOVERNORATE_LABELS:
        governorate, district = GOVERNORATE_LABELS[label]
        return governorate, district, label != "Akkar_Governorate"
    district = label.replace("_District", "").replace(",_Lebanon", "").replace("_", " ")
    return DISTRICT_TO_GOVERNORATE[district], district, False


@st.cache_data
def load_towns() -> pd.DataFrame:
    raw = pd.read_csv(DATA_FILE)
    df = raw.rename(columns=SOURCE_COLUMNS)[["Town", "refArea", *SOURCE_COLUMNS.values()]].copy()

    df["Town"] = df["Town"].str.strip().map(repair_text)
    df["Town"] = df["Town"].replace(TOWN_DISPLAY_NAMES)
    area = df["refArea"].map(repair_text).map(parse_area)
    df["governorate"] = [a[0] for a in area]
    df["district"] = [a[1] for a in area]
    df["district_inferred"] = [a[2] for a in area]

    df["schools"] = df["public"] + df["private"]
    # Despite its label ("schools per citizen") the index runs from 13 to 82,000:
    # it is residents per public school. A zero means "no value", not perfect coverage.
    df["residents_per_public_school"] = df["coverage_index"].where(df["coverage_index"] > 0)

    df["lean"] = np.select(
        [df["schools"] == 0, df["private"] > df["public"], df["private"] < df["public"]],
        [LEAN_NONE, LEAN_PRIVATE, LEAN_PUBLIC],
        default=LEAN_EQUAL,
    )

    # Towns whose school counts are hard to square with their size: 5+ schools but
    # fewer than 100 residents per school implied by the coverage index.
    implied_residents = df["residents_per_public_school"] * df["public"]
    df["counts_look_inconsistent"] = (df["schools"] >= 5) & (implied_residents / df["schools"] < 100)

    # Plot positions for the town scatter: log(1 + count) so Tripoli does not squash
    # every village into one corner, plus a small fixed jitter so towns with identical
    # counts (hundreds sit at 1 public / 0 private) do not hide behind each other.
    rng = np.random.default_rng(325)
    df["x_plot"] = np.log1p(df["public"]) + rng.uniform(-0.07, 0.07, len(df))
    df["y_plot"] = np.log1p(df["private"]) + rng.uniform(-0.07, 0.07, len(df))
    return df


def summarise(towns: pd.DataFrame, by: str) -> pd.DataFrame:
    parent = {"governorate": ("governorate", "first")} if by == "district" else {}
    out = towns.groupby(by).agg(
        **parent,
        towns=("Town", "size"),
        public=("public", "sum"),
        private=("private", "sum"),
        universities=("universities", "sum"),
        vocational=("vocational", "sum"),
        no_school=("schools", lambda s: int((s == 0).sum())),
    )
    out["schools"] = out["public"] + out["private"]
    out["private_share"] = 100 * out["private"] / out["schools"]
    return out.reset_index()


towns = load_towns()
districts = summarise(towns, "district")
governorates = summarise(towns, "governorate")
district_share = districts.set_index("district")["private_share"]
district_governorate = districts.set_index("district")["governorate"]
governorate_share = governorates.set_index("governorate")["private_share"]

N_TOWNS = len(towns)
N_PUBLIC, N_PRIVATE = int(towns["public"].sum()), int(towns["private"].sum())
TOWNS_WITH_PUBLIC = int((towns["public"] > 0).sum())
TOWNS_WITH_PRIVATE = int((towns["private"] > 0).sum())
TOWNS_NO_SCHOOL = int((towns["schools"] == 0).sum())
NATIONAL_SHARE = 100 * N_PRIVATE / (N_PUBLIC + N_PRIVATE)

with_index = towns[(towns["public"] > 0) & towns["residents_per_public_school"].notna()]
MEDIAN_PRIVATE_LED = with_index.loc[with_index["private"] > with_index["public"], "residents_per_public_school"].median()
MEDIAN_PUBLIC_ONLY = with_index.loc[with_index["private"] == 0, "residents_per_public_school"].median()
N_PRIVATE_LED = int((with_index["private"] > with_index["public"]).sum())
N_PUBLIC_ONLY = int((with_index["private"] == 0).sum())
bourj = towns.set_index("Town").loc["Bourj El-Brajneh"]

# Numbers quoted in the design notes and the data notes, computed rather than typed.
TOWNS_WITH_SCHOOL = int((towns["schools"] > 0).sum())
per_district = towns[towns["schools"] > 0].groupby("district").size()
MIN_TOWNS_DISTRICT, MAX_TOWNS_DISTRICT = int(per_district.min()), int(per_district.max())
N_GOVERNORATE_ONLY = int(towns["refArea"].isin(GOVERNORATE_LABELS).sum())
N_INFERRED = int(towns["district_inferred"].sum())
N_ZERO_INDEX_WITH_PUBLIC = int(((towns["public"] > 0) & towns["residents_per_public_school"].isna()).sum())
N_FLAGGED = int(towns["counts_look_inconsistent"].sum())
FLAGGED_SCHOOLS = int(towns.loc[towns["counts_look_inconsistent"], "schools"].sum())
clean_index = with_index[~with_index["counts_look_inconsistent"]]
MEDIAN_PRIVATE_LED_CLEAN = clean_index.loc[clean_index["private"] > clean_index["public"], "residents_per_public_school"].median()
MEDIAN_PUBLIC_ONLY_CLEAN = clean_index.loc[clean_index["private"] == 0, "residents_per_public_school"].median()


# --------------------------------------------------------------------------- #
# Small helpers for text
# --------------------------------------------------------------------------- #
def fmt(n: float) -> str:
    return f"{n:,.0f}"


def pct(n: float) -> str:
    return f"{n:.0f}%"


def dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def pct_range(values: pd.Series) -> str:
    """'30%' when the rounded values agree, otherwise '69–70%'."""
    lo, hi = round(values.min()), round(values.max())
    return f"{lo}%" if lo == hi else f"{lo}–{hi}%"


def lean_word(share: float) -> str:
    if share > 50.5:
        return "private majority"
    if share < 49.5:
        return "public majority"
    return "even split"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def base_layout(fig: go.Figure, height: int) -> None:
    # Importing streamlit switches Plotly's default template to "streamlit", which does
    # not turn on automargin, so long tick labels were being clipped. Pin both.
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=16, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=PAL.ink_2),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=PAL.card, bordercolor=PAL.baseline,
            font=dict(family=FONT, size=13, color=PAL.ink),
        ),
    )


def district_bar_chart(highlight: set[str]) -> go.Figure:
    """Private share of schools per district, as bars growing out of the 50% line."""
    d = districts.sort_values("private_share").reset_index(drop=True)
    share = d["private_share"]

    colours = []
    for name, s in zip(d["district"], share):
        if name not in highlight:
            colours.append(PAL.context_bar)
        elif s > 50.5:
            colours.append(PAL.private)
        elif s < 49.5:
            colours.append(PAL.public)
        else:
            colours.append(PAL.neutral)

    # A 50/50 district (Koura, and Sidon at 67 vs 68) would be an invisible zero-length
    # bar: draw it as a short neutral block centred on the line instead.
    even = (share - 50).abs() < 0.5
    length = np.where(even, 1.4, share - 50)
    start = np.where(even, 49.3, 50.0)

    partial = len(highlight) < len(d)  # bold names only when a subset is selected
    tick_text = [
        (f"<b>{n}</b>" if (n in highlight and partial) else
         n if n in highlight else f"<span style='color:{PAL.muted}'>{n}</span>")
        + f" <span style='color:{PAL.muted}'>({s})</span>"
        for n, s in zip(d["district"], d["schools"])
    ]
    labels = [f"{s:.0f}%" for s in share]
    label_colours = [PAL.ink if n in highlight else PAL.muted for n in d["district"]]
    hover = [
        f"<b>{r.district}</b> · {r.governorate}<br>"
        f"{r.private_share:.0f}% private · {r.schools} schools ({r.public} public, {r.private} private)<br>"
        f"{r.no_school} of {r.towns} towns have no school"
        for r in d.itertuples()
    ]

    fig = go.Figure(go.Bar(
        y=d["district"], x=length, base=start, orientation="h",
        marker=dict(color=colours, line=dict(width=0), cornerradius=4),
        width=0.62,
        text=labels, textposition="outside", cliponaxis=False,
        textfont=dict(color=label_colours, size=12),
        hovertext=hover, hoverinfo="text",
    ))
    base_layout(fig, height=26 * len(d) + 80)
    fig.update_yaxes(
        tickmode="array", tickvals=d["district"], ticktext=tick_text,
        showgrid=False, ticks="", tickfont=dict(size=13),
    )
    fig.update_xaxes(
        range=[16, 84], tickvals=[20, 30, 40, 50, 60, 70, 80],
        ticktext=["20%", "30%", "40%", "50%", "60%", "70%", "80%"],
        showgrid=True, gridcolor=PAL.grid, gridwidth=1, zeroline=False,
        tickfont=dict(color=PAL.muted, size=12), side="top",
    )
    fig.add_vline(x=50, line_color=PAL.baseline, line_width=1.5)
    for x, text, anchor in [(49, "◀ public majority", "right"), (51, "private majority ▶", "left")]:
        fig.add_annotation(
            x=x, y=0, yref="paper", yanchor="top", yshift=-6, text=text, showarrow=False,
            xanchor=anchor, font=dict(size=12, color=PAL.muted),
        )
    fig.update_layout(margin=dict(l=8, r=16, t=30, b=34))
    return fig


def marker_size(residents):
    """Dot diameter in px: area grows with residents per public school (capped at 60,000)."""
    return 7 + 25 * np.sqrt(np.clip(residents, 0, 60_000) / 60_000)


def town_scatter(highlight: set[str]) -> go.Figure:
    """Every town with at least one school: public (x) against private (y)."""
    plotted = towns[towns["schools"] > 0]
    selected = plotted[plotted["district"].isin(highlight)]
    context = plotted[~plotted["district"].isin(highlight)]
    top = np.log1p(110)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, top], y=[0, top], mode="lines", hoverinfo="skip",
        line=dict(color=PAL.baseline, width=1.5, dash="dash"),
    ))
    if len(context):
        fig.add_trace(go.Scatter(
            x=context["x_plot"], y=context["y_plot"], mode="markers", hoverinfo="skip",
            marker=dict(size=6, color=PAL.context_dot, line=dict(width=0)),
        ))

    def hover_lines(frame: pd.DataFrame) -> list[str]:
        lines = []
        for _, r in frame.iterrows():
            if r["public"] == 0:
                cover = "No public school"
            elif pd.isna(r["residents_per_public_school"]):
                cover = "Coverage index missing in the source"
            else:
                cover = f"≈{fmt(r['residents_per_public_school'])} residents per public school"
            extra = []
            if r["universities"]:
                extra.append(f"{r['universities']} universit{'y' if r['universities'] == 1 else 'ies'}")
            if r["vocational"]:
                extra.append(f"{r['vocational']} vocational institute{'s' if r['vocational'] > 1 else ''}")
            text = (
                f"<b>{r['Town']}</b> · {r['district']}<br>"
                f"{r['public']} public · {r['private']} private<br>{cover}"
            )
            if extra:
                text += "<br>" + " · ".join(extra)
            if r["counts_look_inconsistent"]:
                text += "<br><i>Counts look high for the town's size (see data notes)</i>"
            lines.append(text)
        return lines

    colour_of = {LEAN_PUBLIC: PAL.public, LEAN_EQUAL: PAL.neutral, LEAN_PRIVATE: PAL.private}
    for lean, colour in colour_of.items():
        grp = selected[selected["lean"] == lean]
        sized = grp[grp["residents_per_public_school"].notna()]
        hollow = grp[grp["residents_per_public_school"].isna()]
        if len(sized):
            size = marker_size(sized["residents_per_public_school"])
            fig.add_trace(go.Scatter(
                x=sized["x_plot"], y=sized["y_plot"], mode="markers",
                marker=dict(size=size, color=colour, opacity=0.82,
                            line=dict(width=1, color=PAL.ring)),
                hovertext=hover_lines(sized), hoverinfo="text",
            ))
        if len(hollow):
            fig.add_trace(go.Scatter(
                x=hollow["x_plot"], y=hollow["y_plot"], mode="markers",
                marker=dict(size=9, color=colour, symbol="circle-open", line=dict(width=2)),
                hovertext=hover_lines(hollow), hoverinfo="text",
            ))

    # Label the three towns with the most schools in the selection.
    for _, r in selected.nlargest(3, "schools").iterrows():
        above = r["private"] > r["public"]
        near_right = r["x_plot"] > 3.6
        fig.add_annotation(
            x=r["x_plot"], y=r["y_plot"], text=r["Town"].split(" (")[0],
            showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=PAL.muted,
            ax=-46 if (above or near_right) else 40, ay=-28 if above else 30,
            font=dict(size=12, color=PAL.ink), bgcolor="rgba(0,0,0,0)",
        )

    ticks = [0, 1, 2, 5, 10, 20, 50, 100]
    axis = dict(
        range=[-0.3, top], tickvals=np.log1p(ticks), ticktext=[str(t) for t in ticks],
        showgrid=True, gridcolor=PAL.grid, gridwidth=1, zeroline=False,
        tickfont=dict(color=PAL.muted, size=12), title_font=dict(color=PAL.ink_2, size=13),
    )
    fig.update_xaxes(title_text="Public schools in the town (log scale)", **axis)
    fig.update_yaxes(title_text="Private schools in the town (log scale)", scaleanchor="x", scaleratio=1, **axis)
    fig.add_annotation(
        x=0.05, y=top - 0.12, text="More private than public", xanchor="left",
        showarrow=False, font=dict(size=12, color=PAL.muted),
    )
    fig.add_annotation(
        x=top - 0.05, y=0.05, text="More public than private", xanchor="right",
        showarrow=False, font=dict(size=12, color=PAL.muted),
    )
    base_layout(fig, height=640)
    fig.update_layout(margin=dict(l=8, r=16, t=16, b=8), hovermode="closest", hoverdistance=24)
    return fig


# --------------------------------------------------------------------------- #
# Page styles
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
.lede {{ font-size: 1.08rem; line-height: 1.6; color: {PAL.ink_2}; max-width: 62rem; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0.4rem 0 1.2rem; }}
.insight-row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 0.6rem; }}
@media (max-width: 900px) {{
  .kpi-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .insight-row {{ grid-template-columns: 1fr; }}
}}
.card {{ background: {PAL.card}; border: 1px solid {PAL.card_border}; border-radius: 10px; padding: 14px 18px; }}
.kpi .label {{ font-size: 0.9rem; color: {PAL.ink_2}; }}
.kpi .value {{ font-size: 2.1rem; font-weight: 600; line-height: 1.15; color: {PAL.ink}; margin: 2px 0; }}
.kpi .sub {{ font-size: 0.88rem; color: {PAL.muted}; }}
.insight .kicker {{ font-size: 0.8rem; letter-spacing: 0.04em; text-transform: uppercase; color: {PAL.muted}; }}
.insight .head {{ font-size: 1.12rem; font-weight: 600; color: {PAL.ink}; margin: 2px 0 6px; }}
.insight p {{ margin: 0; line-height: 1.55; color: {PAL.ink_2}; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }}
.ring {{ display: inline-block; border-radius: 50%; border: 2px solid {PAL.muted}; vertical-align: middle; margin: 0 4px 0 8px; }}
.dash {{ display: inline-block; width: 18px; border-top: 2px dashed; vertical-align: middle; margin-right: 6px; }}
.legend {{ font-size: 0.88rem; color: {PAL.ink_2}; line-height: 1.9; }}
.legend span.item {{ margin-right: 14px; white-space: nowrap; }}
.chart-title {{ font-size: 1.05rem; font-weight: 600; color: {PAL.ink}; margin-bottom: 0; }}
.chart-sub {{ font-size: 0.9rem; color: {PAL.muted}; margin-bottom: 0.3rem; }}
.summary {{ font-size: 1rem; line-height: 1.6; color: {PAL.ink_2}; }}
.summary b {{ color: {PAL.ink}; }}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Header and context
# --------------------------------------------------------------------------- #
st.title("As many private schools as public ones, but not in the same places")
st.caption(
    "Public and private education across Lebanon's towns and villages, 2023 · "
    "MSBA 325 Streamlit activity · Charbel Youssef El Hajj"
)
st.markdown(
    """
<p class="lede">This page uses the <b>Educational Resources – Lebanon 2023</b> dataset
(Impact Open Data, published through AUB's CODEC platform). Each row is a town or village
outside Beirut, with counts of public schools, private schools, universities and vocational
institutes, plus an index of how many residents each public school serves. It counts
<i>institutions</i>, not pupils. That distinction matters: in 2020–21, 64% of Lebanon's
1.05 million school students were enrolled in private schools (CRDP).</p>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="kpi-row">
  <div class="card kpi"><div class="label">Towns and villages</div>
    <div class="value">{fmt(N_TOWNS)}</div><div class="sub">all governorates except Beirut</div></div>
  <div class="card kpi"><div class="label">{dot(PAL.public)}Public schools</div>
    <div class="value">{fmt(N_PUBLIC)}</div><div class="sub">spread over {fmt(TOWNS_WITH_PUBLIC)} towns</div></div>
  <div class="card kpi"><div class="label">{dot(PAL.private)}Private schools</div>
    <div class="value">{fmt(N_PRIVATE)}</div><div class="sub">concentrated in {fmt(TOWNS_WITH_PRIVATE)} towns</div></div>
  <div class="card kpi"><div class="label">Towns with no school</div>
    <div class="value">{fmt(TOWNS_NO_SCHOOL)}</div><div class="sub">{pct(100 * TOWNS_NO_SCHOOL / N_TOWNS)} of all towns</div></div>
</div>
""",
    unsafe_allow_html=True,
)

most_private = districts.nlargest(3, "private_share")
most_public = districts.nsmallest(2, "private_share")
st.markdown(
    f"""
<div class="insight-row">
  <div class="card insight"><div class="kicker">Insight 1</div>
    <div class="head">Same number of schools, different reach</div>
    <p>Public and private schools are almost exactly tied ({fmt(N_PUBLIC)} vs {fmt(N_PRIVATE)}), but
    public schools are spread across {fmt(TOWNS_WITH_PUBLIC)} towns while private schools sit in
    {fmt(TOWNS_WITH_PRIVATE)}. The private share of schools runs from
    {pct_range(most_public['private_share'])} in {' and '.join(most_public['district'])} to
    {pct_range(most_private['private_share'])} in {', '.join(most_private['district'][:-1])} and
    {most_private['district'].iloc[-1]}.</p></div>
  <div class="card insight"><div class="kicker">Insight 2</div>
    <div class="head">Private schools lead where public schools are stretched</div>
    <p>In the {N_PRIVATE_LED} towns where private schools outnumber public ones, the median public
    school serves {fmt(MEDIAN_PRIVATE_LED)} residents, {MEDIAN_PRIVATE_LED / MEDIAN_PUBLIC_ONLY:.1f} times
    the {fmt(MEDIAN_PUBLIC_ONLY)} in the {N_PUBLIC_ONLY} towns that only have public schools. Beirut's
    southern suburbs show it most clearly: Bourj El-Brajneh has {bourj['private']} private and
    {bourj['public']} public schools, about {fmt(bourj['residents_per_public_school'])} residents per
    public school.</p></div>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# The two linked controls
# --------------------------------------------------------------------------- #
st.subheader("Explore: from Lebanon down to a district")
st.markdown(
    "Choose a governorate, then one of its districts. Both charts respond: the bar chart keeps "
    "every district visible for comparison, and the scatter zooms in on the chosen towns."
)

GOV_OPTIONS = [ALL_LEBANON, *sorted(governorates["governorate"])]


def reset_district() -> None:
    """Widget ① rebuilds widget ②'s options, so start ② again from 'All districts'."""
    st.session_state["district"] = ALL_DISTRICTS


c1, c2 = st.columns([2.3, 1], gap="large")
with c1:
    governorate = st.radio(
        "① Governorate",
        GOV_OPTIONS,
        key="governorate",
        horizontal=True,
        on_change=reset_district,
        help="Start from the national overview, then pick a region.",
    )
if governorate == ALL_LEBANON:
    district_options = sorted(districts["district"])
else:
    district_options = sorted(districts.loc[districts["governorate"] == governorate, "district"])


def district_label(name: str) -> str:
    if name == ALL_DISTRICTS:
        n = len(district_options)
        return f"All {n} districts" if governorate == ALL_LEBANON else f"All {n} districts in {governorate}"
    extra = f" ({district_governorate[name]})" if governorate == ALL_LEBANON else ""
    return f"{name}{extra} · {pct(district_share[name])} private"


with c2:
    district = st.selectbox(
        "② District (list depends on ①)",
        [ALL_DISTRICTS, *district_options],
        key="district",
        format_func=district_label,
        help="Each option shows the district's private share of schools.",
    )

if district != ALL_DISTRICTS:
    highlight = {district}
    scope_name = district
elif governorate != ALL_LEBANON:
    highlight = set(district_options)
    scope_name = governorate
else:
    highlight = set(districts["district"])
    scope_name = ALL_LEBANON

scope = towns[towns["district"].isin(highlight)]

# --------------------------------------------------------------------------- #
# A plain-language answer for the current selection
# --------------------------------------------------------------------------- #
s_public, s_private = int(scope["public"].sum()), int(scope["private"].sum())
s_schools = s_public + s_private
s_share = 100 * s_private / s_schools if s_schools else float("nan")
s_none = int((scope["schools"] == 0).sum())

if scope_name == ALL_LEBANON:
    top_d = districts.loc[districts["private_share"].idxmax()]
    low_d = districts.loc[districts["private_share"].idxmin()]
    summary = (
        f"<b>All Lebanon (outside Beirut)</b>: {fmt(s_schools)} schools in {fmt(len(scope))} towns, "
        f"{pct(s_share)} private. The most private-leaning district is <b>{top_d['district']}</b> "
        f"({pct(top_d['private_share'])}); the most public-leaning is <b>{low_d['district']}</b> "
        f"({pct(low_d['private_share'])})."
    )
elif district == ALL_DISTRICTS:
    rank = int(governorate_share.rank(ascending=False, method="min")[governorate])
    ds = districts[districts["governorate"] == governorate].sort_values("private_share", ascending=False)
    summary = (
        f"<b>{governorate}</b>: {fmt(s_schools)} schools in {fmt(len(scope))} towns, {pct(s_share)} private "
        f"({lean_word(s_share)}; {ordinal(rank)} of {len(governorates)} governorates for private share, "
        f"national average {pct(NATIONAL_SHARE)}). "
        f"{s_none} of its towns ({pct(100 * s_none / len(scope))}) have no school."
    )
    if len(ds) > 1:
        summary += (
            f" Across its districts the private share runs from {pct(ds.iloc[-1]['private_share'])} in "
            f"<b>{ds.iloc[-1]['district']}</b> to {pct(ds.iloc[0]['private_share'])} in "
            f"<b>{ds.iloc[0]['district']}</b>."
        )
else:
    rank = int(district_share.rank(ascending=False, method="min")[district])
    summary = (
        f"<b>{district}</b> ({district_governorate[district]}): {fmt(s_schools)} schools "
        f"in {fmt(len(scope))} towns, {s_public} public and {s_private} private, so {pct(s_share)} private "
        f"({lean_word(s_share)}; {ordinal(rank)} of {len(districts)} districts for private share). "
        f"{s_none} of its towns ({pct(100 * s_none / len(scope))}) have no school."
    )
    stretched = scope.dropna(subset=["residents_per_public_school"])
    if len(stretched):
        t = stretched.loc[stretched["residents_per_public_school"].idxmax()]
        summary += (
            f" Public schools are most stretched in <b>{t['Town']}</b> "
            f"(≈{fmt(t['residents_per_public_school'])} residents per public school, "
            f"{t['private']} private schools)."
        )
    if scope["district_inferred"].any():
        summary += " <i>District assigned from town names; see data notes.</i>"

flagged = scope[scope["counts_look_inconsistent"]]
if len(flagged):
    names = ", ".join(flagged.nlargest(2, "schools")["Town"])
    summary += (
        f"<br><span style='color:{PAL.muted}'>Includes {len(flagged)} town{'s' if len(flagged) > 1 else ''} "
        f"whose counts look high for their size ({names}); see data notes.</span>"
    )

st.markdown(f'<div class="card summary">{summary}</div>', unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------- #
# The two charts
# --------------------------------------------------------------------------- #
left, right = st.columns([1, 1.08], gap="large")
with left:
    st.markdown(
        '<div class="chart-title">Private share of schools, by district</div>'
        '<div class="chart-sub">Bars grow out of the 50% line. Number of schools in brackets. '
        "Districts outside your selection turn grey.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="legend"><span class="item">{dot(PAL.private)}Private majority</span>'
        f'<span class="item">{dot(PAL.public)}Public majority</span>'
        f'<span class="item">{dot(PAL.context_bar)}Outside selection</span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        district_bar_chart(highlight), theme=None, key="bars",
        config={"displayModeBar": False, "scrollZoom": False},
    )

with right:
    shown = int((scope["schools"] > 0).sum())
    st.markdown(
        '<div class="chart-title">Towns: public against private schools</div>'
        f'<div class="chart-sub">{shown} town{"s" if shown != 1 else ""} with at least one school in your '
        "selection, on top of the rest of Lebanon in grey. Hover a dot for details.</div>",
        unsafe_allow_html=True,
    )
    sizes = "".join(
        f'<span class="item"><span class="ring" style="width:{marker_size(r):.0f}px;'
        f'height:{marker_size(r):.0f}px"></span>{fmt(r)}</span>'
        for r in (2_500, 10_000, 40_000)
    )
    st.markdown(
        f'<div class="legend"><span class="item">{dot(PAL.private)}More private</span>'
        f'<span class="item">{dot(PAL.neutral)}Equal</span>'
        f'<span class="item">{dot(PAL.public)}More public</span>'
        f'<span class="item"><span class="dash" style="border-color:{PAL.muted}"></span>Equal numbers</span>'
        f'<span class="item"><span class="ring" style="width:9px;height:9px;border-color:{PAL.private}"></span>'
        "No public school</span><br>"
        f'<span class="item">Dot size = residents per public school:</span>{sizes}</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        town_scatter(highlight), theme=None, key="scatter",
        config={"displayModeBar": False, "scrollZoom": False},
    )

with st.expander(f"Table view: the {fmt(len(scope))} towns in this selection"):
    table = scope.sort_values(["schools", "Town"], ascending=[False, True])[[
        "Town", "district", "governorate", "public", "private", "universities", "vocational",
        "residents_per_public_school", "counts_look_inconsistent",
    ]]
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "district": "District",
            "governorate": "Governorate",
            "public": st.column_config.NumberColumn("Public schools"),
            "private": st.column_config.NumberColumn("Private schools"),
            "universities": st.column_config.NumberColumn("Universities"),
            "vocational": st.column_config.NumberColumn("Vocational institutes"),
            "residents_per_public_school": st.column_config.NumberColumn(
                "Residents per public school", format="localized",
                help="The source's 'public school coverage index'. Empty when the town has no public school.",
            ),
            "counts_look_inconsistent": st.column_config.CheckboxColumn(
                "Counts look high", help="5+ schools but fewer than 100 residents per school implied.",
            ),
        },
    )

# --------------------------------------------------------------------------- #
# Design justifications
# --------------------------------------------------------------------------- #
st.subheader("Design notes: why these two controls")
j1, j2 = st.columns(2, gap="large")
with j1:
    with st.expander("① Governorate: horizontal radio buttons", expanded=False):
        st.markdown(
            """
**User question.** *Is my region one where private schools dominate, and how does it compare
with the rest of Lebanon?*

**Why this widget.** There are only seven governorates plus a national view, so horizontal radio
buttons can show every option at once and always hold exactly one valid choice. I considered a
dropdown, but it hides the options behind a click and the reader loses sight of the regional
structure. I also considered a clickable map, which is the most natural control for geography,
but the dataset has no boundary files, and a map is slower to scan than eight labelled buttons
when the reader already knows which region they want.

**Course concept.** *Overview first, then zoom and filter* (Shneiderman's mantra): the page opens
on "All Lebanon" and the radio zooms into one region. It also *focuses attention* with a
preattentive attribute: the chosen governorate's districts keep their blue or orange while every
other district turns grey. Nothing is filtered away, so the grey bars keep providing *context*:
the reader still sees where the region sits in the national ranking.
"""
        )
with j2:
    with st.expander("② District: a dropdown whose options depend on ①", expanded=False):
        st.markdown(
            f"""
**User question.** *Inside this district, which towns lean private, and are they the towns where
each public school serves the most people?*

**Why this widget.** The dropdown is rebuilt from the governorate choice, so it only lists the one
to six districts that belong to it instead of all 25, and it can never produce an impossible
combination such as "South + Matn". Each option also shows the district's private share, so the
list previews the answer before you click. I considered a multiselect to compare several districts
at once, but colouring towns from many districts together brings back the clutter the drill-down is
meant to remove, and the bar chart already handles comparison between districts.

**Course concept.** *Reducing clutter* and *details on demand*. At the national level the scatter
colours {TOWNS_WITH_SCHOOL} towns; after drilling into one district it colours between
{MIN_TOWNS_DISTRICT} and {MAX_TOWNS_DISTRICT}, with the rest of Lebanon pushed back to light grey.
Hovering any town gives its exact counts and coverage, and the table view lists the same numbers
without hovering.
"""
        )

# --------------------------------------------------------------------------- #
# About the data
# --------------------------------------------------------------------------- #
with st.expander("About the data and how it was cleaned"):
    idx = towns["residents_per_public_school"]
    st.markdown(
        f"""
- **Source.** *Educational Resources – Lebanon 2023*, published by Impact Open Data and served
  through AUB's CODEC linked-data platform (linked.aub.edu.lb). {fmt(N_TOWNS)} towns and villages;
  Beirut is not in the file. Enrolment figure in the introduction: CRDP data for 2020–21, as
  reported by the U.S. International Trade Administration.
- **Two administrative levels in one column.** The `refArea` column names a district for most
  towns but only a governorate for {N_GOVERNORATE_ONLY} of them. For each governorate, those towns
  turned out to be the one district that never appears by name (Chouf, Koura, Rashaya, Jezzine,
  Nabatieh, Baalbek; Akkar is a single-district governorate). I assigned them to that district
  after checking town names ({N_INFERRED} towns in six districts), which gives a clean
  {len(governorates)}-governorate, {len(districts)}-district hierarchy.
- **Broken characters.** Names such as "ZahlÃ©" were UTF-8 text read with the wrong encoding;
  they are repaired by re-encoding (Zahlé, Miniyeh–Danniyeh).
- **Coverage index.** The source calls it "number of schools per citizen", but its values run from
  {fmt(idx.min())} to {fmt(idx.max())}, so it is really residents per public school. It is zero for
  every town without a public school and for {N_ZERO_INDEX_WITH_PUBLIC} towns that do have one;
  those zeros are treated as missing, not as perfect coverage.
- **Counts that look too high.** {N_FLAGGED} small towns report five or more schools but fewer
  than 100 residents per school (for example Abou Qamha in Hasbaya: 55 schools for about 200
  residents). Together they hold {FLAGGED_SCHOOLS} of the {fmt(N_PUBLIC + N_PRIVATE)} schools. They
  are kept, because the source reports them, but marked in the hover text and the table. Without
  them, Insight 2 still holds: {fmt(MEDIAN_PRIVATE_LED_CLEAN)} residents per public school in
  private-led towns against {fmt(MEDIAN_PUBLIC_ONLY_CLEAN)} in public-only towns.
- **What a count cannot tell you.** A school is a school here, whatever its size. The dataset has
  no enrolment, so "more private schools" does not mean "more pupils in private schools".
"""
    )

st.caption(
    "Built with Streamlit and Plotly. Charts evolve two figures from my Plotly assignment: the "
    "'top areas' bar chart (now at a consistent district level and split by sector) and the "
    "public-versus-private scatter (now with log axes, an equality line and coverage as dot size)."
)
