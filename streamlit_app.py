import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from fltower.main_fltower import process_fcs_files, compile_summary_report
from fltower.data_manager import save_parameters
import zipfile

st.title("FLtower - Cytometry Analysis")

# Upload ZIP file
st.header("1. Upload a ZIP File with .FCS Files")
uploaded_zip = st.file_uploader("Upload a .zip file containing your .FCS files", type="zip")

# Graph configuration
st.header("2. Configure Plots")
with open("test/reference_input/parameters.json", "r") as f:
    default_parameters = json.load(f)

selected_plots = st.multiselect(
    "Select the types of plots to include:",
    options=list(default_parameters.keys()),
    default=list(default_parameters.keys())
)

# Modify parameters for selected plots
custom_parameters = {}
for key in selected_plots:
    with st.expander(f" Parameters for {key}", expanded=False):
        plot_config = default_parameters[key]

        # Display main gates
        if plot_config["type"] == "scatter":
            if "quadrant_gates" not in plot_config:
                plot_config["quadrant_gates"] = {"x": 0, "y": 0}
            st.markdown("**Quadrant Gates**")
            qx = st.number_input(f"{key} - Quadrant X", value=plot_config["quadrant_gates"].get("x", 0), key=key + "_qx")
            qy = st.number_input(f"{key} - Quadrant Y", value=plot_config["quadrant_gates"].get("y", 0), key=key + "_qy")
            plot_config["quadrant_gates"]["x"] = qx
            plot_config["quadrant_gates"]["y"] = qy

        elif plot_config["type"] == "histogram":
            st.markdown("**Histogram Gates**")
            gates = plot_config.get("gates", [])
            new_gates = []
            for i, gate in enumerate(gates):
                g0 = st.number_input(f"{key} - Gate {i+1} Min", value=gate[0], key=f"{key}_g{i}_min")
                g1 = st.number_input(f"{key} - Gate {i+1} Max", value=gate[1], key=f"{key}_g{i}_max")
                new_gates.append([g0, g1])
            plot_config["gates"] = new_gates

        # Advanced settings toggle
        if st.toggle(f"Show advanced settings for {key}"):
            st.markdown("### 🔧 Advanced Settings")

            # Shared
            plot_config["x_param"] = st.text_input(f"{key} - x_param", value=plot_config.get("x_param", ""), key=key + "_x_adv")
            plot_config["x_scale"] = st.selectbox(f"{key} - x_scale", ["linear", "log"], index=["linear", "log"].index(plot_config.get("x_scale", "log")), key=key + "_xscale")

            if plot_config["type"] == "scatter":
                plot_config["y_param"] = st.text_input(f"{key} - y_param", value=plot_config.get("y_param", ""), key=key + "_y_adv")
                plot_config["y_scale"] = st.selectbox(f"{key} - y_scale", ["linear", "log"], index=["linear", "log"].index(plot_config.get("y_scale", "log")), key=key + "_yscale")

                plot_config["xlim"] = [
                    st.number_input(f"{key} - xlim min", value=plot_config.get("xlim", [0, 0])[0], key=key + "_xlim_min"),
                    st.number_input(f"{key} - xlim max", value=plot_config.get("xlim", [0, 0])[1], key=key + "_xlim_max")
                ]
                plot_config["ylim"] = [
                    st.number_input(f"{key} - ylim min", value=plot_config.get("ylim", [0, 0])[0], key=key + "_ylim_min"),
                    st.number_input(f"{key} - ylim max", value=plot_config.get("ylim", [0, 0])[1], key=key + "_ylim_max")
                ]
                plot_config["cmap"] = st.text_input(f"{key} - Color map", value=plot_config.get("cmap", ""), key=key + "_cmap")
                plot_config["gridsize"] = st.number_input(f"{key} - Grid size", value=plot_config.get("gridsize", 100), key=key + "_gridsize")
                plot_config["scatter_type"] = st.text_input(f"{key} - Scatter type", value=plot_config.get("scatter_type", ""), key=key + "_scattertype")

            elif plot_config["type"] == "histogram":
                plot_config["xlim"] = [
                    st.number_input(f"{key} - xlim min", value=plot_config.get("xlim", [0, 0])[0], key=key + "_xlim_min"),
                    st.number_input(f"{key} - xlim max", value=plot_config.get("xlim", [0, 0])[1], key=key + "_xlim_max")
                ]
                plot_config["color"] = st.text_input(f"{key} - Color", value=plot_config.get("color", ""), key=key + "_color")
                plot_config["kde"] = st.checkbox(f"{key} - Show KDE", value=plot_config.get("kde", True), key=key + "_kde")

            # Optional plots
            st.markdown("#### 📈 96-Well Plots")
            plot_config["96well_plots"] = plot_config.get("96well_plots", [])
            new_96well = []
            for i, p in enumerate(plot_config["96well_plots"]):
                metric = st.text_input(f"{key} - 96well Plot {i+1} - Metric", value=p.get("metric", ""), key=f"{key}_96_{i}_metric")
                title = st.text_input(f"{key} - 96well Plot {i+1} - Title", value=p.get("title", ""), key=f"{key}_96_{i}_title")
                new_96well.append({"metric": metric, "title": title})
            plot_config["96well_plots"] = new_96well

            st.markdown("#### 📊 Triplicate Plots")
            plot_config["triplicate_plots"] = plot_config.get("triplicate_plots", [])
            new_trip = []
            for i, p in enumerate(plot_config["triplicate_plots"]):
                metric = st.text_input(f"{key} - Triplicate Plot {i+1} - Metric", value=p.get("metric", ""), key=f"{key}_trip_{i}_metric")
                title = st.text_input(f"{key} - Triplicate Plot {i+1} - Title", value=p.get("title", ""), key=f"{key}_trip_{i}_title")
                new_trip.append({"metric": metric, "title": title})
            plot_config["triplicate_plots"] = new_trip

        custom_parameters[key] = plot_config



# Run analysis
st.header("3. Run Analysis")
if st.button("Analyze Data") and uploaded_zip:
    with st.spinner("Processing..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save and extract the uploaded ZIP
            zip_path = os.path.join(temp_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())

            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Filter .fcs files
            fcs_files = [
                os.path.join(extract_dir, f)
                for f in os.listdir(extract_dir)
                if f.lower().endswith(".fcs")
            ]

            if not fcs_files:
                st.error("No .FCS files found in the ZIP archive.")
            else:
                # Save parameters
                parameters_path = os.path.join(temp_dir, "parameters.json")
                with open(parameters_path, "w") as f:
                    json.dump(custom_parameters, f, indent=4)

                # Prepare results folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = os.path.join(temp_dir, f"results_{timestamp}")
                os.makedirs(results_dir, exist_ok=True)

                # Copy FCS files into a new input folder
                input_dir = os.path.join(temp_dir, "input")
                os.makedirs(input_dir)
                for fcs_path in fcs_files:
                    with open(fcs_path, "rb") as source, open(os.path.join(input_dir, os.path.basename(fcs_path)), "wb") as dest:
                        dest.write(source.read())

                # Run the analysis
                scatter_dfs, hist_dfs, singlet_df, runtime = process_fcs_files(
                    input_dir, custom_parameters, results_dir
                )
                compile_summary_report(results_dir, custom_parameters)

                st.success(f"Analysis completed in {runtime:.1f} seconds")

                # Create ZIP of results
                zip_results_path = os.path.join(results_dir, "complete_results.zip")
                with zipfile.ZipFile(zip_results_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(results_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, results_dir)
                            zipf.write(file_path, arcname=arcname)

                # Download button
                with open(zip_results_path, "rb") as f:
                    st.download_button(
                        "Download all results (.zip)",
                        f,
                        file_name="fltower_results.zip",
                        mime="application/zip"
                    )
