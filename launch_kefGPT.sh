set -e
cd "$(dirname "$0")"
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
fi
echo "Activating virtual environment..."
source venv/bin/activate
echo "Installing dependencies (if not installed)..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Launching KefGPT..."
streamlit run app.py
echo "Press Ctrl+C to exit."