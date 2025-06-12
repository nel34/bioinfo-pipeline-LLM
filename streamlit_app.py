import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from fltower.main_fltower import process_fcs_files, compile_summary_report
import zipfile
from PIL import Image
import glob

os.makedirs("results", exist_ok=True)

st.set_page_config(layout="wide")
st.markdown("<h1 style='font-size: 4rem; font-weight: bold;'>FLtower - Cytometry Analysis</h1>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 3rem;'></div>", unsafe_allow_html=True)

# Load default parameters
with open("test/reference_input/parameters.json", "r") as f:
    default_parameters = json.load(f)

# App layout
col1, spacer, col2 = st.columns([1, 0.3, 2])

custom_parameters = {}
results_dir = None

# -------------------- LEFT COLUMN --------------------
with col1:
    st.header("1. Upload a ZIP File with .FCS Files")
    uploaded_zip = st.file_uploader("Upload a .zip file containing your .FCS files", type="zip")

    st.header("3. Run Analysis")
    if st.button("Analyze Data") and uploaded_zip:
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "uploaded.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())

                extract_dir = os.path.join(temp_dir, "extracted")
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                fcs_files = [
                    os.path.join(extract_dir, f)
                    for f in os.listdir(extract_dir)
                    if f.lower().endswith(".fcs")
                ]

                if not fcs_files:
                    st.error("No .FCS files found in the ZIP archive.")
                else:
                    parameters_path = os.path.join(temp_dir, "parameters.json")
                    with open(parameters_path, "w") as f:
                        json.dump(custom_parameters, f, indent=4)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_dir = os.path.join("results", f"results_{timestamp}")
                    os.makedirs(results_dir, exist_ok=True)

                    input_dir = os.path.join(temp_dir, "input")
                    os.makedirs(input_dir)
                    for fcs_path in fcs_files:
                        with open(fcs_path, "rb") as source, open(os.path.join(input_dir, os.path.basename(fcs_path)), "wb") as dest:
                            dest.write(source.read())

                    scatter_dfs, hist_dfs, singlet_df, runtime = process_fcs_files(
                        input_dir, custom_parameters, results_dir
                    )
                    compile_summary_report(results_dir, custom_parameters)

                    st.success(f"Analysis completed in {runtime:.1f} seconds")

                    zip_results_path = os.path.join(results_dir, "complete_results.zip")
                    with zipfile.ZipFile(zip_results_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(results_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, results_dir)
                                zipf.write(file_path, arcname=arcname)

                    with open(zip_results_path, "rb") as f:
                        st.download_button(
                            "Download all results (.zip)",
                            f,
                            file_name="fltower_results.zip",
                            mime="application/zip"
                        )



# -------------------- RIGHT COLUMN --------------------
with col2:
    st.header("2. Configure Plots")
    selected_plots = st.multiselect(
        "Select the types of plots to include:",
        options=list(default_parameters.keys()),
        default=list(default_parameters.keys())
    )

    for key in selected_plots:
        with st.expander(f"Parameters for {key}", expanded=False):
            plot_config = default_parameters[key]

            if plot_config["type"] == "scatter":
                if "quadrant_gates" in plot_config:
                    st.markdown("**Quadrant Gates**")
                    qx = st.number_input(f"{key} - Quadrant X", value=plot_config["quadrant_gates"].get("x", 0), key=key + "_qx")
                    qy = st.number_input(f"{key} - Quadrant Y", value=plot_config["quadrant_gates"].get("y", 0), key=key + "_qy")
                    plot_config["quadrant_gates"]["x"] = qx
                    plot_config["quadrant_gates"]["y"] = qy
                else:
                    st.info("No editable parameters available for this scatter plot.")

            elif plot_config["type"] == "histogram":
                st.markdown("**Histogram Gates**")
                gates = plot_config.get("gates", [])
                new_gates = []
                for i, gate in enumerate(gates):
                    g0 = st.number_input(f"{key} - Gate {i+1} Min", value=gate[0], key=f"{key}_g{i}_min")
                    g1 = st.number_input(f"{key} - Gate {i+1} Max", value=gate[1], key=f"{key}_g{i}_max")
                    new_gates.append([g0, g1])
                plot_config["gates"] = new_gates

            custom_parameters[key] = plot_config
