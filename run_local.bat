@echo off
echo ====================================================================
echo Starting Real-Time Anomaly Detection on Streaming Sensor Data
echo ====================================================================

if not exist .venv (
    echo [1/4] Creating Python virtual environment...
    python -m venv .venv
)

echo [2/4] Activating virtual environment and installing dependencies...
call .venv\Scripts\activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo [3/4] Calibrating PyTorch Autoencoder ^& Isolation Forest Models...
python -m src.models

echo [4/4] Launching Streamlit Fault-Injection ^& Telemetry Sandbox...
streamlit run app.py
pause
