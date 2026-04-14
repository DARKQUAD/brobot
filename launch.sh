#!/usr/bin/env bash
# ============================================================
#  Qwen3 Pipeline - GUI  (Debian / Ubuntu / Linux)
# ============================================================
set -e

echo "========================================"
echo " Qwen3 Pipeline - GUI  (Debian/Linux)"
echo "========================================"
echo

if ! command -v python3 &>/dev/null; then
    echo "[ERREUR] python3 introuvable."
    echo "Installez : sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

PYTHON=$(command -v python3)
echo "[OK] Python : $($PYTHON --version)"

VENV_DIR="$(dirname "$0")/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/6] Création du virtualenv .venv/ ..."
    $PYTHON -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"

echo "[2/6] Mise à jour pip..."
$PIP install --upgrade pip --quiet

echo "[3/6] Dépendances de base..."
$PIP install customtkinter Pillow sounddevice soundfile numpy scipy --quiet

echo "[4/6] PyTorch (CPU)..."
$PIP install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

echo "[5/6] Transformers + llama-cpp-python + qwen-tts..."
$PIP install transformers accelerate --quiet
$PIP install llama-cpp-python --quiet
$PIP install qwen-tts --quiet

echo
echo "[6/6] Démarrage..."
echo "(1er lancement : téléchargement automatique des modèles ~1.8 Go)"
echo "========================================"
$PY "$(dirname "$0")/app.py"
