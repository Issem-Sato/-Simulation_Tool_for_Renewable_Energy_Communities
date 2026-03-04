@echo off
REM === Crea cartelle principali ===
mkdir src
mkdir src\cer_core
mkdir src\cer_core\vision
mkdir src\cer_core\pv
mkdir src\cer_core\energy
mkdir src\cer_core\finance
mkdir src\cer_app
mkdir src\cer_app\pages
mkdir data
mkdir data\raw data\interim data\processed data\results

REM === File package markers ===
type nul > src\cer_core\__init__.py
type nul > src\cer_core\vision\__init__.py
type nul > src\cer_core\pv\__init__.py
type nul > src\cer_core\energy\__init__.py
type nul > src\cer_core\finance\__init__.py
type nul > src\cer_app\__init__.py

REM === Stubs basilari ===
echo # config placeholder > src\cer_core\config.py

echo """UNet inference stub""" > src\cer_core\vision\inference.py
echo """tiling utils stub""" > src\cer_core\vision\tiling.py
echo """postprocess utils stub""" > src\cer_core\vision\postprocess.py

echo """geometry stub""" > src\cer_core\pv\geometry.py
echo """sizing stub""" > src\cer_core\pv\sizing.py
echo """production stub""" > src\cer_core\pv\production.py

echo """matching stub""" > src\cer_core\energy\matching.py
echo """metrics stub""" > src\cer_core\energy\metrics.py

echo """cashflow stub""" > src\cer_core\finance\cashflow.py
echo """tariff stub""" > src\cer_core\finance\tariff.py

REM === App Streamlit multipagina (senza emoji nei nomi file: più comodo su Windows) ===
(
echo import streamlit as st
echo st.set_page_config(page_title="CER Simulator", layout="wide")
echo st.title("CER Simulator - Home")
echo st.markdown("Seleziona una pagina dal menu di sinistra.")
) > src\cer_app\app.py

(
echo import streamlit as st
echo st.title("1 - Segmentazione Tetti")
echo st.info("Carica un^immagine e avvia la segmentazione (stub).")
) > "src\cer_app\pages\1_Segmentazione.py"

(
echo import streamlit as st
echo st.title("2 - Dimensionamento PV")
echo st.info("Calcolo Wp da area utile (stub).")
) > "src\cer_app\pages\2_Dimensionamento_PV.py"

(
echo import streamlit as st
echo st.title("3 - Produzione e Consumi")
echo st.info("Simulazione produzione con pvlib e matching consumi (stub).")
) > "src\cer_app\pages\3_Produzione_Consumi.py"

(
echo import streamlit as st
echo st.title("4 - Economia e Report")
echo st.info("NPV/IRR e esportazione report (stub).")
) > "src\cer_app\pages\4_Economia_Report.py"

echo Struttura creata.
