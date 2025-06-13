import streamlit as st

st.set_page_config(layout="wide")

import os
import tempfile
import json
from datetime import datetime
from fltower.main_fltower import process_fcs_files, compile_summary_report
import zipfile
import re

# Main title centered
st.markdown("<h1 style='font-size: 4rem; margin-bottom: 2rem;'>FLtower - Cytometry Analysis</h1>", unsafe_allow_html=True)

# Create two columns with spacing in the middle
col1, spacer, col2 = st.columns([1, 0.2, 2])

# === Column 2: Graph selection and parameters ===
with col2:
    st.markdown("<h2 style='font-size: 2.2rem;'>2. Graph Configuration</h2>", unsafe_allow_html=True)
    with open("test/reference_input/parameters.json", "r") as f:
        default_parameters = json.load(f)

    selected_plots = st.multiselect(
        "Select the types of plots to include:",
        options=list(default_parameters.keys()),
        default=list(default_parameters.keys())
    )
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

# === Dynamic parameter setup ===
custom_parameters = {}

with col2:
    for key in selected_plots:
        plot_config = default_parameters[key]
        with st.expander(f"📊 {key}", expanded=False):
            st.markdown(f"**Plot type: `{plot_config['type']}`**")
            if "x_param" in plot_config:
                plot_config["x_param"] = st.text_input(f"{key} - x_param", value=plot_config["x_param"], key=f"{key}_xparam")
            if plot_config["type"] == "scatter" and "y_param" in plot_config:
                plot_config["y_param"] = st.text_input(f"{key} - y_param", value=plot_config["y_param"], key=f"{key}_yparam")
            if "x_scale" in plot_config:
                plot_config["x_scale"] = st.selectbox(f"{key} - x_scale", options=["linear", "log"], index=["linear", "log"].index(plot_config["x_scale"]), key=f"{key}_xscale")
            if plot_config["type"] == "scatter" and "y_scale" in plot_config:
                plot_config["y_scale"] = st.selectbox(f"{key} - y_scale", options=["linear", "log"], index=["linear", "log"].index(plot_config["y_scale"]), key=f"{key}_yscale")
            if "xlim" in plot_config:
                plot_config["xlim"] = [
                    st.number_input(f"{key} - xlim min", value=plot_config["xlim"][0], key=f"{key}_xlim_min"),
                    st.number_input(f"{key} - xlim max", value=plot_config["xlim"][1], key=f"{key}_xlim_max")
                ]
            if "ylim" in plot_config:
                plot_config["ylim"] = [
                    st.number_input(f"{key} - ylim min", value=plot_config["ylim"][0], key=f"{key}_ylim_min"),
                    st.number_input(f"{key} - ylim max", value=plot_config["ylim"][1], key=f"{key}_ylim_max")
                ]
            if "cmap" in plot_config:
                plot_config["cmap"] = st.text_input(f"{key} - colormap", value=plot_config["cmap"], key=f"{key}_cmap")
            if "gridsize" in plot_config:
                plot_config["gridsize"] = st.number_input(f"{key} - gridsize", value=plot_config["gridsize"], key=f"{key}_gridsize")
            if "scatter_type" in plot_config:
                plot_config["scatter_type"] = st.selectbox(f"{key} - scatter_type", options=["density", "scatter"], index=["density", "scatter"].index(plot_config["scatter_type"]), key=f"{key}_stype")
            if "quadrant_gates" in plot_config:
                st.markdown("**Quadrant Gates**")
                plot_config["quadrant_gates"] = {
                    "x": st.number_input(f"{key} - Quadrant gate X", value=plot_config["quadrant_gates"]["x"], key=f"{key}_qx"),
                    "y": st.number_input(f"{key} - Quadrant gate Y", value=plot_config["quadrant_gates"]["y"], key=f"{key}_qy")
                }
            if "color" in plot_config:
                plot_config["color"] = st.text_input(f"{key} - color", value=plot_config["color"], key=f"{key}_color")
            if "kde" in plot_config:
                plot_config["kde"] = st.checkbox(f"{key} - KDE", value=plot_config["kde"], key=f"{key}_kde")
            if "gates" in plot_config:
                st.markdown("**Gates (interval)**")
                new_gates = []
                for i, gate in enumerate(plot_config["gates"]):
                    col1_g, col2_g = st.columns(2)
                    with col1_g:
                        gmin = st.number_input(f"{key} - Gate {i+1} min", value=gate[0], key=f"{key}_gmin_{i}")
                    with col2_g:
                        gmax = st.number_input(f"{key} - Gate {i+1} max", value=gate[1], key=f"{key}_gmax_{i}")
                    new_gates.append([gmin, gmax])
                plot_config["gates"] = new_gates
        custom_parameters[key] = plot_config

# === Column 1: File upload and run button ===
with col1:
    st.markdown("<h2 style='font-size: 2.2rem;'>1. Upload .zip containing your FCS files</h2>", unsafe_allow_html=True)
    uploaded_zip = st.file_uploader("Drop a .zip file", type="zip")

    st.markdown("<h2 style='font-size: 2.2rem;'>3. Run analysis</h2>", unsafe_allow_html=True)
    run_analysis = st.button("Run analysis")

    if run_analysis and uploaded_zip:
        with st.spinner("Processing data..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                input_dir = os.path.join(temp_dir, "input")
                os.makedirs(input_dir)
                uploaded_zip_path = os.path.join(temp_dir, "uploaded.zip")
                with open(uploaded_zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())

                with zipfile.ZipFile(uploaded_zip_path, "r") as zip_ref:
                    for member in zip_ref.infolist():
                        if (
                            not member.filename.lower().endswith(".fcs") or
                            "__macosx" in member.filename.lower() or
                            member.filename.startswith(".") or
                            member.is_dir()
                        ):
                            continue
                        filename = os.path.basename(member.filename)
                        target_path = os.path.join(input_dir, filename)
                        with zip_ref.open(member) as source, open(target_path, "wb") as target:
                            target.write(source.read())

                fcs_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".fcs")]
                if not fcs_files:
                    st.error("No .fcs files found in the .zip.")
                    st.stop()

                parameters_path = os.path.join(temp_dir, "parameters.json")
                with open(parameters_path, "w") as f:
                    json.dump(custom_parameters, f, indent=4)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = os.path.join(temp_dir, f"results_{timestamp}")
                os.makedirs(results_dir, exist_ok=True)

                scatter_dfs, hist_dfs, singlet_df, runtime = process_fcs_files(
                    input_dir, custom_parameters, results_dir
                )
                compile_summary_report(results_dir, custom_parameters)

                st.success(f"Analysis completed in {runtime:.1f} seconds")

                zip_path = os.path.join(results_dir, "full_results.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(results_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, results_dir)
                            zipf.write(file_path, arcname=arcname)

                with open(zip_path, "rb") as f:
                    st.download_button(
                        "📦 Download all results (.zip)",
                        f,
                        file_name="fltower_results.zip",
                        mime="application/zip"
                    )
