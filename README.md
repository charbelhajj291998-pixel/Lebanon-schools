# Lebanon's schools: public vs private

**Live app:** https://lebanon-schools-public-private.streamlit.app

MSBA 325 · Streamlit interactivity activity · Charbel Youssef El Hajj

An interactive Streamlit page built on the **Educational Resources – Lebanon 2023** dataset
(Impact Open Data, served through AUB's CODEC platform): 1,137 towns and villages outside
Beirut, with counts of public schools, private schools, universities and vocational institutes.

Nationally there are as many private schools as public ones (1,035 vs 1,038), but they are not
in the same places. The page asks where each sector dominates and whether private schools lead
where public schools are most stretched.

## The two linked controls

1. **Governorate** (radio buttons) → picks a region and rebuilds the district list.
2. **District** (dropdown whose options depend on 1) → drills into one district.

Both controls drive both charts:

- a bar chart of the private share of schools in each of the 25 districts (evolved from the
  "top areas" bar chart in my Plotly assignment), with the selection in colour and the rest in grey;
- a town-level scatter of public against private schools (evolved from the Plotly scatter), with
  dot size showing residents per public school.

The design justification for each control is on the page, under **Design notes**.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | The app: data cleaning, charts, widgets, page text |
| `dataset.csv` | The source data, unchanged |
| `requirements.txt` | Pinned package versions used for deployment |
| `.streamlit/config.toml` | Light theme and colours |

## Run it locally

Python 3.11 or newer:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Data cleaning in brief

- `refArea` mixes districts and governorates; the 448 towns labelled only by governorate are
  assigned to the one district of that governorate that never appears by name (checked on town names).
- Double-encoded names (`ZahlÃ©`) are repaired to UTF-8 (`Zahlé`).
- The "public school coverage index" is residents per public school (values 13 to 82,000);
  zeros are treated as missing.
- 14 small towns whose counts look too high for their size are kept but flagged.
