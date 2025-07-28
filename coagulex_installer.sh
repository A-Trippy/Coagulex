#!/bin/bash

echo "🚀 Starting Coagulex Installer..."

# ==== Step 1: System Update ====
echo "🔄 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# ==== Step 2: Install Python & Tools ====
echo "🐍 Installing Python 3, pip, venv, and git..."
sudo apt-get install -y python3 python3-pip python3-venv git

# ==== Step 3: Clone the Coagulex GitHub Repo ====
echo "📥 Cloning Coagulex project from GitHub..."
git clone https://github.com/A-Trippy/coagulex-tracker.git ~/coagulex-tracker || {
    echo "⚠️ Repo already exists. Pulling latest changes..."
    cd ~/coagulex-tracker && git pull
}
cd ~/coagulex-tracker

# ==== Step 4: Create and Activate Virtual Environment ====
echo "📁 Creating virtual environment..."
python3 -m venv ~/coagulex-env
source ~/coagulex-env/bin/activate

# ==== Step 5: Install Python Dependencies ====
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# ==== Step 6: Install Imaging Source Drivers ====
if [ -f "install-imaging-driver.sh" ]; then
    echo "📸 Installing Imaging Source camera drivers..."
    chmod +x install-imaging-driver.sh
    sudo ./install-imaging-driver.sh
else
    echo "⚠️ No Imaging Source driver script found. Skipping camera setup."
fi

# ==== Step 7: Launch App ====
echo "🎬 Launching Coagulex application..."
python3 singleCam.py

echo "✅ Setup complete!"
