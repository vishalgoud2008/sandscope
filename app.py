import streamlit as st
import os
import shutil
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

from analyze import process_image


st.set_page_config(
    page_title="SandScope",
    page_icon="🌊",
    layout="wide"
)

DATA_FOLDER = "coastal_data"
os.makedirs(DATA_FOLDER, exist_ok=True)


REGIONS = [
    {"id": "region_01", "name": "Mumbai Coast", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "coast": "West Coast"},
    {"id": "region_02", "name": "Ratnagiri Coast", "state": "Maharashtra", "lat": 16.9944, "lon": 73.3000, "coast": "West Coast"},
    {"id": "region_03", "name": "Panaji Coast", "state": "Goa", "lat": 15.4909, "lon": 73.8278, "coast": "West Coast"},
    {"id": "region_04", "name": "Karwar Coast", "state": "Karnataka", "lat": 14.8121, "lon": 74.1297, "coast": "West Coast"},
    {"id": "region_05", "name": "Mangaluru Coast", "state": "Karnataka", "lat": 12.9141, "lon": 74.8560, "coast": "West Coast"},
    {"id": "region_06", "name": "Udupi Coast", "state": "Karnataka", "lat": 13.3409, "lon": 74.7421, "coast": "West Coast"},
    {"id": "region_07", "name": "Kochi Coast", "state": "Kerala", "lat": 9.9312, "lon": 76.2673, "coast": "West Coast"},
    {"id": "region_08", "name": "Thiruvananthapuram Coast", "state": "Kerala", "lat": 8.5241, "lon": 76.9366, "coast": "West Coast"},
    {"id": "region_09", "name": "Kanniyakumari Coast", "state": "Tamil Nadu", "lat": 8.0883, "lon": 77.5385, "coast": "Southern Coast"},
    {"id": "region_10", "name": "Rameswaram Coast", "state": "Tamil Nadu", "lat": 9.2876, "lon": 79.3129, "coast": "Southern Coast"},
    {"id": "region_11", "name": "Puducherry Coast", "state": "Puducherry", "lat": 11.9416, "lon": 79.8083, "coast": "East Coast"},
    {"id": "region_12", "name": "Nagapattinam Coast", "state": "Tamil Nadu", "lat": 10.7672, "lon": 79.8449, "coast": "East Coast"},
    {"id": "region_13", "name": "Chennai Coast", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "coast": "East Coast"},
    {"id": "region_14", "name": "Nellore Coast", "state": "Andhra Pradesh", "lat": 14.4426, "lon": 79.9865, "coast": "East Coast"},
    {"id": "region_15", "name": "Kakinada Coast", "state": "Andhra Pradesh", "lat": 16.9891, "lon": 82.2475, "coast": "East Coast"},
    {"id": "region_16", "name": "Visakhapatnam Coast", "state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185, "coast": "East Coast"},
    {"id": "region_17", "name": "Bhubaneswar Coastal Region", "state": "Odisha", "lat": 20.2961, "lon": 85.8245, "coast": "East Coast"},
    {"id": "region_18", "name": "Puri Coast", "state": "Odisha", "lat": 19.8135, "lon": 85.8312, "coast": "East Coast"},
    {"id": "region_19", "name": "Digha Coast", "state": "West Bengal", "lat": 21.6268, "lon": 87.5074, "coast": "East Coast"},
    {"id": "region_20", "name": "Kolkata Coastal Region", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "coast": "East Coast"},
]


def classify_sand(d50):
    if d50 < 0.125:
        return "Very Fine Sand", "NORMAL"
    elif d50 < 0.25:
        return "Fine Sand", "NORMAL"
    elif d50 < 0.50:
        return "Medium Sand", "NORMAL"
    elif d50 < 1.00:
        return "Coarse Sand", "WATCH"
    else:
        return "Very Coarse / Gravel-like", "ATTENTION"


def get_region(region_id):
    for region in REGIONS:
        if region["id"] == region_id:
            return region
    return REGIONS[0]


def get_region_folder(region_id):
    folder = os.path.join(DATA_FOLDER, region_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def get_samples(region_id):
    folder = get_region_folder(region_id)
    samples = []

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            samples.append({"name": name, "path": path})

    return sorted(samples, key=lambda x: x["name"])


def get_network_stats():
    total_sites = len(REGIONS)
    active_sites = 0
    total_samples = 0

    for region in REGIONS:
        region_samples = get_samples(region["id"])
        if region_samples:
            active_sites += 1
            total_samples += len(region_samples)

    return total_sites, active_sites, total_samples


def create_map():
    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=[r["lat"] for r in REGIONS],
            lon=[r["lon"] for r in REGIONS],
            mode="markers",
            customdata=[r["id"] for r in REGIONS],
            text=[
                f"<b>{r['name']}</b><br>{r['state']}<br>{r['coast']}<br><br>🖱️ Click to select"
                for r in REGIONS
            ],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=12, opacity=0.95, line=dict(width=2)),
        )
    )

    fig.update_geos(
        scope="asia",
        projection_type="mercator",
        center=dict(lat=18, lon=79),
        projection_scale=5,
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="aliceblue",
        showcountries=True,
        countrycolor="gray",
        showcoastlines=True,
        coastlinecolor="gray",
        coastlinewidth=1.5,
        showlakes=True,
        lakecolor="aliceblue",
        lonaxis=dict(range=[67, 92]),
        lataxis=dict(range=[5, 30]),
    )

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        clickmode="event+select",
        hoverlabel=dict(bgcolor="white", font_size=14),
    )

    return fig


st.title("🌊 SandScope")
st.subheader("Coastal Sand Intelligence Network")
st.write(
    "Explore India's coastal regions and analyze sand grain characteristics."
)

st.divider()

# =========================================================
# NETWORK OVERVIEW
# =========================================================

st.subheader("📊 Coastal Network Overview")

total_sites, active_sites, total_samples = get_network_stats()
pending_sites = total_sites - active_sites

n1, n2, n3, n4 = st.columns(4)

with n1:
    st.metric("Coastal Sites", total_sites)

with n2:
    st.metric("Total Samples", total_samples)

with n3:
    st.metric("Active Sites", active_sites)

with n4:
    st.metric("Awaiting Samples", pending_sites)

st.caption(
    "Network statistics are calculated from the samples currently saved "
    "in the coastal_data folder."
)

st.divider()

st.subheader("🇮🇳 Indian Coastal Sampling Network")
st.caption("Hover over a location for information. Click a location to select it.")

fig = create_map()

event = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="coastal_map",
)

if event is not None:
    if event.selection is not None:
        if len(event.selection.points) > 0:
            point = event.selection.points[0]
            region_id = point.get("customdata")

            if region_id:
                st.session_state["selected_region_id"] = region_id


if "selected_region_id" not in st.session_state:
    st.session_state["selected_region_id"] = REGIONS[0]["id"]


selected_region = get_region(
    st.session_state["selected_region_id"]
)

st.divider()

st.header("📍 " + selected_region["name"])

st.write("**State:** " + selected_region["state"])
st.write("**Coastal Zone:** " + selected_region["coast"])
st.write(
    "**Coordinates:** "
    + f'{selected_region["lat"]:.4f}, '
    + f'{selected_region["lon"]:.4f}'
)

samples = get_samples(selected_region["id"])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Samples Collected", len(samples))

with col2:
    st.metric(
        "Region ID",
        selected_region["id"].replace("region_", "R")
    )

with col3:
    st.metric(
        "Data Status",
        "Available" if samples else "No Data"
    )


st.divider()
st.subheader("📊 Previous Sand Data")

if not samples:
    st.info(
        "No previous sand-analysis data is available for this region."
    )
else:
    for sample in reversed(samples):

        summary_path = os.path.join(
            sample["path"],
            "summary.csv"
        )

        analyzed_path = os.path.join(
            sample["path"],
            "analyzed.jpg"
        )

        histogram_path = os.path.join(
            sample["path"],
            "grain_size_distribution.png"
        )

        with st.expander(sample["name"]):

            if os.path.exists(summary_path):
                try:
                    summary_df = pd.read_csv(summary_path)
                    st.dataframe(
                        summary_df,
                        use_container_width=True,
                        hide_index=True
                    )
                except Exception as error:
                    st.warning(
                        "Could not read summary: " + str(error)
                    )

            left, right = st.columns(2)

            with left:
                if os.path.exists(analyzed_path):
                    st.image(
                        analyzed_path,
                        caption="Detected grains",
                        use_container_width=True
                    )

            with right:
                if os.path.exists(histogram_path):
                    st.image(
                        histogram_path,
                        caption="Grain-size distribution",
                        use_container_width=True
                    )


st.divider()
st.subheader("📷 Add New Sand Sample")

st.write("Sample location:")
st.info(selected_region["name"])

uploaded_file = st.file_uploader(
    "Upload sand image",
    type=["jpg", "jpeg", "png"],
    key="sample_upload"
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded sand sample",
        use_container_width=True
    )

if uploaded_file is not None:

    if st.button(
        "🔬 Analyze & Save Sample",
        type="primary",
        use_container_width=True
    ):

        os.makedirs("images", exist_ok=True)

        temp_filename = "app_sample.png"
        temp_path = os.path.join("images", temp_filename)

        with open(temp_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        with st.spinner("Analyzing sand grains..."):

            try:
                result = process_image(temp_filename)

            except Exception as error:
                st.error("Analysis error: " + str(error))
                result = None

        if result is not None:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            sample_name = "sample_" + timestamp

            region_folder = get_region_folder(
                selected_region["id"]
            )

            sample_folder = os.path.join(
                region_folder,
                sample_name
            )

            os.makedirs(
                sample_folder,
                exist_ok=True
            )

            backend_folder = os.path.join(
                "results",
                "app_sample"
            )

            if os.path.exists(backend_folder):

                for filename in os.listdir(backend_folder):

                    source = os.path.join(
                        backend_folder,
                        filename
                    )

                    destination = os.path.join(
                        sample_folder,
                        filename
                    )

                    if os.path.isfile(source):
                        shutil.copy2(
                            source,
                            destination
                        )

            shutil.copy2(
                temp_path,
                os.path.join(
                    sample_folder,
                    "uploaded_image.png"
                )
            )

            location_data = {
                "region_id": selected_region["id"],
                "region_name": selected_region["name"],
                "state": selected_region["state"],
                "coastal_zone": selected_region["coast"],
                "latitude": selected_region["lat"],
                "longitude": selected_region["lon"],
                "date": datetime.now().isoformat(),
            }

            with open(
                os.path.join(
                    sample_folder,
                    "location.json"
                ),
                "w"
            ) as file:

                json.dump(
                    location_data,
                    file,
                    indent=4
                )

            st.success(
                "✅ Sample analyzed and saved successfully!"
            )

            st.divider()
            st.subheader("📊 Grain Analysis")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Detected Grains",
                    result["Grains"]
                )

            with c2:
                st.metric(
                    "Mean Diameter",
                    f'{result["Mean mm"]:.4f} mm'
                )

            with c3:
                st.metric(
                    "D50",
                    f'{result["D50 mm"]:.4f} mm'
                )

            c4, c5, c6 = st.columns(3)

            with c4:
                st.metric(
                    "D10",
                    f'{result["D10 mm"]:.4f} mm'
                )

            with c5:
                st.metric(
                    "D30",
                    f'{result["D30 mm"]:.4f} mm'
                )

            with c6:
                st.metric(
                    "D60",
                    f'{result["D60 mm"]:.4f} mm'
                )

            st.divider()
            st.subheader("🏖️ Sand Classification")

            d50_value = result["D50 mm"]

            sand_class, status = classify_sand(d50_value)

            class_col, status_col = st.columns(2)

            with class_col:
                st.metric(
                    "Sand Type",
                    sand_class
                )

            with status_col:

                if status == "NORMAL":
                    st.success("🟢 NORMAL")

                elif status == "WATCH":
                    st.warning("🟡 WATCH")

                else:
                    st.error("🔴 ATTENTION")

            st.caption(
                "This classification is based only on measured D50 "
                "and is intended as a sediment-size screening indicator."
            )

            analyzed_image = os.path.join(
                backend_folder,
                "analyzed.jpg"
            )

            histogram_image = os.path.join(
                backend_folder,
                "grain_size_distribution.png"
            )

            left, right = st.columns(2)

            with left:

                if os.path.exists(analyzed_image):

                    st.subheader("🔍 Detected Grains")

                    st.image(
                        analyzed_image,
                        use_container_width=True
                    )

            with right:

                if os.path.exists(histogram_image):

                    st.subheader("📈 Grain Distribution")

                    st.image(
                        histogram_image,
                        use_container_width=True
                    )

            st.success(
                "📍 Saved under " + selected_region["name"]
            )

        else:

            st.error("❌ Sand analysis failed.")

            st.warning(
                "Make sure the image contains the required reference square."
            )


st.divider()

st.caption(
    "SandScope • Coastal Sediment Observation Platform"
)
