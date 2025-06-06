import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from fltower.main_fltower import process_fcs_files, compile_summary_report
from fltower.data_manager import save_parameters

st.title("FLtower - Analyse de Cytométrie")

# Upload FCS files
st.header("1. Upload des fichiers .FCS")
uploaded_files = st.file_uploader("Sélectionnez vos fichiers FCS", type="fcs", accept_multiple_files=True)

# Paramètres dynamiques
st.header("2. Configuration des graphiques")
with open("parameters.json", "r") as f:
    default_parameters = json.load(f)

selected_plots = st.multiselect(
    "Choisir les types de graphes à inclure :",
    options=list(default_parameters.keys()),
    default=list(default_parameters.keys())
)

# Permet de modifier les paramètres (optionnellement)
custom_parameters = {}
for key in selected_plots:
    st.subheader(f"Paramètres pour {key}")
    plot_config = default_parameters[key]
    x_param = st.text_input(f"{key} - x_param", plot_config.get("x_param", ""), key + "_x")
    y_param = st.text_input(f"{key} - y_param", plot_config.get("y_param", ""), key + "_y") if plot_config["type"] == "scatter" else None
    plot_config["x_param"] = x_param
    if y_param:
        plot_config["y_param"] = y_param
    custom_parameters[key] = plot_config

# Lancer l’analyse
st.header("3. Lancer l’analyse")
if st.button("Analyser les données") and uploaded_files:
    with st.spinner("Traitement en cours..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            os.makedirs(input_dir)
            for file in uploaded_files:
                with open(os.path.join(input_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())

            # Sauver les paramètres
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

            st.success(f"Analyse terminée en {runtime:.1f} secondes")

            # Télécharger le rapport
            pdf_path = os.path.join(results_dir, "summary_report.pdf")
            with open(pdf_path, "rb") as f:
                st.download_button("Télécharger le rapport PDF", f, file_name="summary_report.pdf")
