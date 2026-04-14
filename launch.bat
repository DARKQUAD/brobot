@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set TF_ENABLE_ONEDNN_OPTS=0
echo ========================================
echo  Qwen3 Pipeline - GUI  (Windows + CUDA)
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python introuvable. Installez Python 3.10+ depuis python.org
    pause & exit /b 1
)

echo [0/7] Nettoyage des conflits PyTorch existants...
pip uninstall torchvision torchaudio torch -y --quiet 2>nul
echo     Anciens paquets PyTorch supprimes.

echo [1/7] Dependances de base...
pip install customtkinter Pillow sounddevice soundfile numpy scipy --quiet

echo [2/7] PyTorch CUDA 12.8 (RTX 5060 / Blackwell)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
if errorlevel 1 (
    echo [AVERTISSEMENT] Index CUDA 12.8 indisponible, essai CUDA 12.4...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
    if errorlevel 1 (
        echo [AVERTISSEMENT] Fallback CPU uniquement...
        pip install torch torchvision torchaudio --quiet
    )
)

echo [3/7] Transformers - ASR Qwen3-ASR...
pip install "transformers>=4.40.0" accelerate huggingface_hub --quiet

echo [4/7] LLM GGUF (llama-cpp-python)...
python -c "from llama_cpp import Llama; print('  llama-cpp-python deja OK - conserve')" 2>nul
if not errorlevel 1 goto llama_ok
echo   llama-cpp-python absent ou casse, tentative de reinstallation...
pip uninstall llama-cpp-python -y --quiet 2>nul
python -c "import shutil,site; [shutil.rmtree(p+'/llama_cpp',True) for p in site.getsitepackages()]" 2>nul
pip install llama-cpp-python --prefer-binary --quiet
if errorlevel 1 (
    echo   [AVERTISSEMENT] Impossible de compiler llama-cpp-python sans Visual Studio Build Tools.
    echo   Telechargez-les ici : https://aka.ms/vs/17/release/vs_BuildTools.exe
    echo   Selectionnez "Developpement Desktop en C++" puis relancez ce script.
    echo   Le LLM sera desactive jusqu'a l'installation du compilateur.
)
:llama_ok

echo [5/7] TTS Qwen3 (qwen-tts)...
pip install qwen-tts --quiet

echo [6/7] Verification GPU + packages...
python -c "import torch; cuda=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if cuda else 'AUCUN'; print('  torch', torch.__version__, '| CUDA:', cuda, '| GPU:', name)"
python -c "import torchvision; print('  torchvision', torchvision.__version__, '- OK')" 2>nul || echo   torchvision absent
python -c "import transformers; print('  transformers', transformers.__version__, '- OK')" 2>nul || echo   [AVERTISSEMENT] transformers absent
python -c "from qwen_tts import Qwen3TTSModel; print('  qwen-tts - OK')" 2>nul || echo   [AVERTISSEMENT] qwen-tts absent
python -c "from llama_cpp import Llama; print('  llama-cpp - OK')" 2>nul || echo   [AVERTISSEMENT] llama-cpp absent

echo.
echo [7/7] Test import ASR complet...
python -c "import torch; exec(\"try:\n from transformers import AutoModelForSpeechSeq2Seq as M\nexcept ImportError:\n from transformers import AutoModelForSpeech2Seq as M\nprint('  ASR imports OK - classe:', M.__name__)\")"
if errorlevel 1 (
    echo   ^^^ ERREUR import ASR - voir ci-dessus.
    echo   Solution : pip install --upgrade transformers
)

echo.
echo ========================================
echo  Demarrage...
echo  (1er lancement : telechargement auto
echo   des modeles Qwen3 ~1.8 Go total)
echo ========================================
python app.py
pause
