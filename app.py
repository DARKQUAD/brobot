#!/usr/bin/env python3
"""
Qwen3 Pipeline — Interface Graphique  (Windows / Debian / macOS)
GUI : Qwen3-ASR  →  Qwen3.5-LLM (GGUF)  →  Qwen3-TTS

Modèles :
  ASR  : Qwen/Qwen3-ASR-0.6B               (transformers, téléchargé auto)
  LLM  : Qwen_Qwen3.5-4B-Q5_K_M.gguf      (llama-cpp-python, local GGUF)
  TTS  : Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice  (qwen-tts, téléchargé auto)

Installation :
    pip install customtkinter pillow sounddevice soundfile numpy scipy huggingface-hub
    pip install torch torchaudio transformers
    pip install llama-cpp-python
    pip install qwen-tts
"""

import os, sys, platform, time, threading, queue, subprocess
from pathlib import Path
from datetime import datetime

import numpy as np

# ── GUI ───────────────────────────────────────────────────────────────────────
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

# ── Audio ─────────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False

# ── LLM backend (llama-cpp-python) ───────────────────────────────────────────
LLAMA_IMPORT_ERROR = ""
try:
    from llama_cpp import Llama
    LLAMA_OK = True
except Exception as _llama_exc:   # ImportError ou RuntimeError (llama.dll manquant)
    LLAMA_OK = False
    LLAMA_IMPORT_ERROR = f"{type(_llama_exc).__name__}: {_llama_exc}"

# ── ASR backend (transformers + Qwen3-ASR) ───────────────────────────────────
# transformers ≥4.46 renamed AutoModelForSpeech2Seq → AutoModelForSpeechSeq2Seq
ASR_IMPORT_ERROR = ""
ASR_MODEL_CLASS = None
try:
    import torch
    try:
        from transformers import AutoModelForSpeechSeq2Seq as _ASRModel, AutoProcessor
    except ImportError:
        from transformers import AutoModelForSpeech2Seq as _ASRModel, AutoProcessor
    ASR_MODEL_CLASS = _ASRModel
    ASR_OK = True
except Exception as _asr_exc:   # ImportError OU OSError (DLL manquante / incompatible)
    ASR_OK = False
    ASR_IMPORT_ERROR = f"{type(_asr_exc).__name__}: {_asr_exc}"

# ── TTS backend (qwen-tts) ───────────────────────────────────────────────────
try:
    from qwen_tts import Qwen3TTSModel
    QWEN_TTS_OK = True
except Exception:
    QWEN_TTS_OK = False

# ── Device ───────────────────────────────────────────────────────────────────
TORCH_IMPORT_ERROR = ""
try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = _torch.bfloat16 if DEVICE == "cuda" else _torch.float32
    TORCH_OK = True
except Exception as _torch_exc:   # catch OSError si torchvision ou autre DLL incompatible
    DEVICE = "cpu"
    TORCH_DTYPE = None
    TORCH_OK = False
    TORCH_IMPORT_ERROR = f"{type(_torch_exc).__name__}: {_torch_exc}"

# ══════════════════════════════════════════════════════════════════════════════
# CHEMINS & IDENTIFIANTS MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).parent.resolve()

# ── Téléchargement auto du GGUF depuis Hugging Face ──────────────────────────
LLM_HF_REPO    = "Qwen/Qwen3.5-4B-GGUF"
LLM_HF_FILE    = "qwen3.5-4b-q5_k_m.gguf"
LLM_MODEL_PATH = ROOT / "models" / LLM_HF_FILE

ASR_MODEL_ID   = "Qwen/Qwen3-ASR-0.6B"
TTS_MODEL_ID   = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def _download_gguf():
    """Télécharge le GGUF depuis Hugging Face Hub si absent."""
    if LLM_MODEL_PATH.exists():
        return LLM_MODEL_PATH

    LLM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub manquant : pip install huggingface-hub"
        )

    print(f"⬇️  Téléchargement de {LLM_HF_REPO}/{LLM_HF_FILE} ...")
    downloaded = hf_hub_download(
        repo_id=LLM_HF_REPO,
        filename=LLM_HF_FILE,
        local_dir=str(LLM_MODEL_PATH.parent),
        local_dir_use_symlinks=False,
    )
    print(f"✅ Modèle sauvegardé : {downloaded}")
    return Path(downloaded)


# ══════════════════════════════════════════════════════════════════════════════
# BACKENDS
# ══════════════════════════════════════════════════════════════════════════════
class ModelState:
    UNLOADED = "⬜ Non chargé"
    LOADING  = "🔄 Chargement…"
    READY    = "✅ Prêt"
    ERROR    = "❌ Erreur"


class LLMBackend:
    """Qwen3.5-4B GGUF via llama-cpp-python."""
    def __init__(self):
        self.llm     = None
        self.state   = ModelState.UNLOADED
        self.error   = ""
        self.history = []

    def load(self, n_ctx=4096, n_threads=None, n_gpu_layers=None):
        self.state = ModelState.LOADING
        if not LLAMA_OK:
            self.state = ModelState.ERROR
            self.error = LLAMA_IMPORT_ERROR or "llama-cpp-python manquant : pip install llama-cpp-python"
            return False

        # ── Télécharger le GGUF si absent ──
        try:
            _download_gguf()
        except Exception as e:
            self.state = ModelState.ERROR
            self.error = f"Téléchargement GGUF échoué : {e}"
            return False

        if not LLM_MODEL_PATH.exists():
            self.state = ModelState.ERROR
            self.error = f"GGUF introuvable après téléchargement : {LLM_MODEL_PATH}"
            return False

        if n_gpu_layers is None:
            n_gpu_layers = -1 if DEVICE == "cuda" else 0
        try:
            self.llm = Llama(
                model_path=str(LLM_MODEL_PATH),
                n_ctx=n_ctx,
                n_threads=n_threads or N_THREADS,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            self.history = []
            self.state = ModelState.READY
            return True
        except Exception as e:
            self.error = str(e)
            self.state = ModelState.ERROR
            return False

    def transcribe(self, audio_path: str, language: str = "French") -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("ASR non chargé.")
        lang_map = {
            "French": "fr", "English": "en", "Chinese": "zh",
            "Japanese": "ja", "German": "de", "Spanish": "es",
            "Korean": "ko", "Russian": "ru", "Italian": "it",
        }
        lang_code = lang_map.get(language, language.lower()[:2])

        audio_array, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        if sample_rate != 16_000:
            try:
                import torchaudio
                import torch
                wf = torch.tensor(audio_array).unsqueeze(0)
                audio_array = torchaudio.transforms.Resample(sample_rate, 16_000)(wf).squeeze(0).numpy()
                sample_rate = 16_000
            except ImportError:
                pass

        import torch
        inputs = self.processor(
            audio_array, sampling_rate=sample_rate,
            return_tensors="pt", language=lang_code,
        ).to(DEVICE)

        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class TTSBackend:
    """Qwen3-TTS-0.6B-CustomVoice via package qwen-tts."""
    def __init__(self):
        self.model  = None
        self.state  = ModelState.UNLOADED
        self.error  = ""

    def load(self):
        self.state = ModelState.LOADING
        if not QWEN_TTS_OK:
            self.state = ModelState.ERROR
            self.error = "qwen-tts manquant : pip install qwen-tts"
            return False
        try:
            import torch
            dtype      = TORCH_DTYPE or torch.float32
            device_map = DEVICE if DEVICE == "cuda" else "cpu"
            self.model = Qwen3TTSModel.from_pretrained(
                TTS_MODEL_ID,
                device_map=device_map,
                dtype=dtype,
            )
            self.state = ModelState.READY
            return True
        except Exception as e:
            self.error = str(e)
            self.state = ModelState.ERROR
            return False

    def synthesize(self, text: str, output_path: str,
                   speaker: str = "Vivian", language: str = "French") -> str:
        if self.state != ModelState.READY:
            raise RuntimeError("TTS non prêt.")

        lang_map = {
            "French": "French", "English": "English", "Chinese": "Chinese",
            "Japanese": "Japanese", "German": "German", "Spanish": "Spanish",
            "Korean": "Korean", "Russian": "Russian", "Italian": "Italian",
        }
        tts_lang = lang_map.get(language, "English")

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=tts_lang,
            speaker=speaker.capitalize(),
        )
        sf.write(output_path, wavs[0], sr)
        return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ENREGISTREMENT AUDIO
# ══════════════════════════════════════════════════════════════════════════════
class AudioRecorder:
    def __init__(self, device_idx=None, samplerate=16000):
        self.device_idx = device_idx
        self.samplerate = samplerate
        self._recording = False
        self._frames    = []

    def start(self):
        self._frames    = []
        self._recording = True

        def callback(indata, frames, time_info, status):
            if self._recording:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            device=self.device_idx, samplerate=self.samplerate,
            channels=1, dtype="float32", callback=callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        self._recording = False
        self._stream.stop()
        self._stream.close()
        if not self._frames:
            return np.zeros(0)
        return np.concatenate(self._frames, axis=0).squeeze()

    def save(self, path: str) -> str:
        audio = self.stop()
        sf.write(path, audio, self.samplerate, subtype="PCM_16")
        return path


# ══════════════════════════════════════════════════════════════════════════════
# COULEURS
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "bg":       "#0f1117",
    "sidebar":  "#161b22",
    "card":     "#1c2333",
    "accent":   "#58a6ff",
    "accent2":  "#7c3aed",
    "user_msg": "#1e3a5f",
    "bot_msg":  "#1c2333",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
    "border":   "#30363d",
    "green":    "#3fb950",
    "red":      "#f85149",
    "orange":   "#d29922",
    "record":   "#da3633",
}


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Qwen3 Pipeline  •  ASR → LLM → TTS")
        self.geometry("1180x780")
        self.minsize(900, 600)
        self.configure(fg_color=COLORS["bg"])

        self.llm = LLMBackend()
        self.asr = ASRBackend()
        self.tts = TTSBackend()

        self.attached_image: str | None = None
        self.recording  = False
        self.recorder   = None
        self._job_queue = queue.Queue()
        self._tmp_dir   = ROOT / "_tmp"
        self._tmp_dir.mkdir(exist_ok=True)

        self._build_layout()
        self._refresh_devices()
        self._poll_queue()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=270, fg_color=COLORS["sidebar"], corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(99, weight=1)

        ctk.CTkLabel(sb, text="⚡ Qwen3 Pipeline",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=COLORS["accent"]).grid(
            row=0, column=0, pady=(18, 6), padx=16, sticky="w")

        # ── Statut modèles ────────────────────────────────────────────────────
        ctk.CTkLabel(sb, text="MODÈLES",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=COLORS["subtext"]).grid(
            row=1, column=0, pady=(10, 2), padx=16, sticky="w")

        self._model_cards = {}
        os_info  = platform.system()
        gpu_info = f"GPU:{DEVICE.upper()}"
        for key, label in [
            ("asr", f"ASR  Qwen3-ASR-0.6B  ({gpu_info})"),
            ("llm", f"LLM  Qwen3.5-4B GGUF  ({os_info})"),
            ("tts", f"TTS  Qwen3-TTS-0.6B  ({gpu_info})"),
        ]:
            card = self._make_model_card(sb, key, label)
            card.grid(row={"asr":2,"llm":3,"tts":4}[key],
                      column=0, padx=12, pady=3, sticky="ew")

        ctk.CTkButton(sb, text="⬇  Charger tous les modèles",
                      fg_color=COLORS["accent2"], hover_color="#6d28d9",
                      command=self._load_all_models).grid(
            row=5, column=0, padx=12, pady=(8,4), sticky="ew")

        ctk.CTkFrame(sb, height=1, fg_color=COLORS["border"]).grid(
            row=6, column=0, padx=12, pady=8, sticky="ew")

        # ── Appareils audio ───────────────────────────────────────────────────
        ctk.CTkLabel(sb, text="APPAREILS",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=COLORS["subtext"]).grid(
            row=7, column=0, pady=(2,2), padx=16, sticky="w")

        ctk.CTkLabel(sb, text="🎙  Microphone", text_color=COLORS["text"]).grid(
            row=8, column=0, padx=16, sticky="w")
        self.mic_var  = ctk.StringVar(value="Défaut")
        self.mic_menu = ctk.CTkOptionMenu(sb, variable=self.mic_var, values=["Défaut"],
                                          fg_color=COLORS["card"],
                                          button_color=COLORS["border"])
        self.mic_menu.grid(row=9, column=0, padx=12, pady=(0,4), sticky="ew")

        ctk.CTkLabel(sb, text="🔊  Sortie audio", text_color=COLORS["text"]).grid(
            row=10, column=0, padx=16, sticky="w")
        self.out_var  = ctk.StringVar(value="Défaut")
        self.out_menu = ctk.CTkOptionMenu(sb, variable=self.out_var, values=["Défaut"],
                                          fg_color=COLORS["card"],
                                          button_color=COLORS["border"])
        self.out_menu.grid(row=11, column=0, padx=12, pady=(0,4), sticky="ew")

        ctk.CTkButton(sb, text="↺ Rafraîchir", height=28,
                      fg_color=COLORS["card"], hover_color=COLORS["border"],
                      command=self._refresh_devices).grid(
            row=12, column=0, padx=12, pady=(0,4), sticky="ew")

        ctk.CTkFrame(sb, height=1, fg_color=COLORS["border"]).grid(
            row=13, column=0, padx=12, pady=8, sticky="ew")

        # ── Configuration ─────────────────────────────────────────────────────
        ctk.CTkLabel(sb, text="CONFIGURATION",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=COLORS["subtext"]).grid(
            row=14, column=0, pady=(2,2), padx=16, sticky="w")

        ctk.CTkLabel(sb, text="🌐  Langue", text_color=COLORS["text"]).grid(
            row=15, column=0, padx=16, sticky="w")
        self.lang_var = ctk.StringVar(value="French")
        ctk.CTkOptionMenu(sb, variable=self.lang_var,
                          values=["French","English","Chinese","Japanese",
                                  "German","Spanish","Korean","Russian","Italian"],
                          fg_color=COLORS["card"],
                          button_color=COLORS["border"]).grid(
            row=16, column=0, padx=12, pady=(0,4), sticky="ew")

        ctk.CTkLabel(sb, text="🎤  Voix TTS", text_color=COLORS["text"]).grid(
            row=17, column=0, padx=16, sticky="w")
        self.spk_var = ctk.StringVar(value="Vivian")
        ctk.CTkOptionMenu(sb, variable=self.spk_var,
                          values=["Serena","Vivian","Ryan","Aiden",
                                  "Uncle_fu","Ono_anna","Sohee","Eric","Dylan"],
                          fg_color=COLORS["card"],
                          button_color=COLORS["border"]).grid(
            row=18, column=0, padx=12, pady=(0,4), sticky="ew")

        ctk.CTkFrame(sb, height=1, fg_color=COLORS["border"]).grid(
            row=19, column=0, padx=12, pady=8, sticky="ew")

        # ── Tests ──────────────────────────────────────────────────────────────
        ctk.CTkLabel(sb, text="TEST PIPELINE",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=COLORS["subtext"]).grid(
            row=20, column=0, pady=(2,2), padx=16, sticky="w")

        ctk.CTkButton(sb, text="▶  Test LLM seul", height=32,
                      fg_color=COLORS["card"], hover_color=COLORS["border"],
                      command=lambda: self._test_component("llm")).grid(
            row=21, column=0, padx=12, pady=2, sticky="ew")
        ctk.CTkButton(sb, text="▶  Test TTS seul", height=32,
                      fg_color=COLORS["card"], hover_color=COLORS["border"],
                      command=lambda: self._test_component("tts")).grid(
            row=22, column=0, padx=12, pady=2, sticky="ew")
        ctk.CTkButton(sb, text="🔁  Test Pipeline complet", height=36,
                      fg_color=COLORS["accent2"], hover_color="#6d28d9",
                      command=self._test_full_pipeline).grid(
            row=23, column=0, padx=12, pady=(2,4), sticky="ew")

        ctk.CTkButton(sb, text="🗑  Vider le chat", height=28,
                      fg_color=COLORS["card"], hover_color=COLORS["record"],
                      command=self._clear_chat).grid(
            row=99, column=0, padx=12, pady=(4,12), sticky="ew")

    def _make_model_card(self, parent, key, label):
        frame = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=8)
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text=label,
                     font=ctk.CTkFont(size=11),
                     text_color=COLORS["text"]).grid(
            row=0, column=0, padx=10, pady=(8,2), sticky="w")
        status_lbl = ctk.CTkLabel(frame, text=ModelState.UNLOADED,
                                  font=ctk.CTkFont(size=10),
                                  text_color=COLORS["subtext"])
        status_lbl.grid(row=1, column=0, padx=10, pady=(0,8), sticky="w")
        self._model_cards[key] = status_lbl
        return frame

    # ── Main ──────────────────────────────────────────────────────────────────
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.status_bar = ctk.CTkLabel(
            main, text="Prêt.", height=28, corner_radius=0,
            fg_color=COLORS["card"], text_color=COLORS["subtext"],
            font=ctk.CTkFont(size=11), anchor="w")
        self.status_bar.grid(row=0, column=0, sticky="ew")

        chat_frame = ctk.CTkFrame(main, fg_color=COLORS["bg"])
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(8,0))
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_scroll = ctk.CTkScrollableFrame(
            chat_frame, fg_color=COLORS["bg"],
            scrollbar_button_color=COLORS["border"])
        self.chat_scroll.grid(row=0, column=0, sticky="nsew")
        self.chat_scroll.grid_columnconfigure(0, weight=1)
        self._chat_row = 0

        # Image preview
        self.img_preview_frame = ctk.CTkFrame(main, fg_color=COLORS["card"],
                                               height=80, corner_radius=8)
        self.img_preview_frame.grid_columnconfigure(0, weight=1)
        self.img_preview_label = ctk.CTkLabel(self.img_preview_frame, text="")
        self.img_preview_label.grid(row=0, column=0, padx=8, pady=4, sticky="w")
        ctk.CTkButton(self.img_preview_frame, text="✕ Retirer", width=80, height=24,
                      fg_color=COLORS["record"], hover_color="#991b1b",
                      command=self._clear_image).grid(row=0, column=1, padx=8, pady=4)

        # Input bar
        input_frame = ctk.CTkFrame(main, fg_color=COLORS["card"],
                                   corner_radius=12, height=64)
        input_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=12)
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_propagate(False)

        self.rec_btn = ctk.CTkButton(
            input_frame, text="🎙", width=44, height=44, corner_radius=22,
            fg_color=COLORS["card"], hover_color=COLORS["record"],
            font=ctk.CTkFont(size=18))
        self.rec_btn.grid(row=0, column=1, padx=(8,4), pady=10)
        self.rec_btn.bind("<ButtonPress-1>",   self._start_record)
        self.rec_btn.bind("<ButtonRelease-1>", self._stop_record)

        self.text_input = ctk.CTkEntry(
            input_frame, placeholder_text="Écris un message…",
            fg_color=COLORS["bg"], border_color=COLORS["border"],
            text_color=COLORS["text"], font=ctk.CTkFont(size=13), height=44)
        self.text_input.grid(row=0, column=0, padx=(8,4), pady=10, sticky="ew")
        self.text_input.bind("<Return>", self._on_enter)

        ctk.CTkButton(
            input_frame, text="📎", width=44, height=44, corner_radius=22,
            fg_color=COLORS["card"], hover_color=COLORS["border"],
            font=ctk.CTkFont(size=18), command=self._attach_image).grid(
            row=0, column=2, padx=(0,4), pady=10)

        self.send_btn = ctk.CTkButton(
            input_frame, text="Envoyer ➤", width=100, height=44, corner_radius=10,
            fg_color=COLORS["accent"], hover_color="#1d4ed8",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._send_message)
        self.send_btn.grid(row=0, column=3, padx=(0,8), pady=10)

    # ── Messages chat ─────────────────────────────────────────────────────────
    def _add_message(self, role: str, text: str, image_path: str = None):
        is_user = role == "user"
        anchor  = "e" if is_user else "w"
        bg      = COLORS["user_msg"] if is_user else COLORS["bot_msg"]
        prefix  = "👤 Vous" if is_user else "🤖 Assistant"
        padx    = (80, 8) if is_user else (8, 80)

        bubble = ctk.CTkFrame(self.chat_scroll, fg_color=bg, corner_radius=12)
        bubble.grid(row=self._chat_row, column=0, sticky=anchor, padx=padx, pady=4)
        bubble.grid_columnconfigure(0, weight=1)
        self._chat_row += 1

        ctk.CTkLabel(bubble, text=prefix,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=COLORS["accent"] if is_user else COLORS["green"],
                     anchor="w").grid(row=0, column=0, padx=12, pady=(8,0), sticky="w")

        if image_path:
            try:
                img = Image.open(image_path)
                img.thumbnail((220, 160))
                ctk.CTkLabel(bubble, image=ctk.CTkImage(img, size=img.size), text="").grid(
                    row=1, column=0, padx=12, pady=4, sticky="w")
            except Exception:
                pass

        ctk.CTkLabel(bubble, text=text, wraplength=480, justify="left",
                     text_color=COLORS["text"],
                     font=ctk.CTkFont(size=13), anchor="w").grid(
            row=2, column=0, padx=12, pady=(2,8), sticky="w")

        ctk.CTkLabel(bubble, text=datetime.now().strftime("%H:%M"),
                     font=ctk.CTkFont(size=9), text_color=COLORS["subtext"],
                     anchor="e").grid(row=3, column=0, padx=12, pady=(0,6), sticky="e")

        self.chat_scroll._parent_canvas.after(
            50, lambda: self.chat_scroll._parent_canvas.yview_moveto(1.0))

    def _add_system_msg(self, text: str, color=None):
        color = color or COLORS["subtext"]
        lbl = ctk.CTkLabel(self.chat_scroll, text=f"— {text} —",
                           text_color=color, font=ctk.CTkFont(size=11), anchor="center")
        lbl.grid(row=self._chat_row, column=0, pady=4, padx=20, sticky="ew")
        self._chat_row += 1

    # ── Appareils ─────────────────────────────────────────────────────────────
    def _refresh_devices(self):
        if not AUDIO_OK:
            self.mic_menu.configure(values=["sounddevice manquant"])
            self.out_menu.configure(values=["sounddevice manquant"])
            return
        devices = sd.query_devices()
        inputs  = ["Défaut"] + [f"{i}: {d['name']}" for i, d in enumerate(devices)
                                 if d["max_input_channels"] > 0]
        outputs = ["Défaut"] + [f"{i}: {d['name']}" for i, d in enumerate(devices)
                                 if d["max_output_channels"] > 0]
        self.mic_menu.configure(values=inputs)
        self.out_menu.configure(values=outputs)
        self._set_status("Appareils audio rafraîchis.")

    def _get_mic_idx(self):
        v = self.mic_var.get()
        return None if v == "Défaut" else int(v.split(":")[0])

    def _get_out_idx(self):
        v = self.out_var.get()
        return None if v == "Défaut" else int(v.split(":")[0])

    # ── Chargement modèles ────────────────────────────────────────────────────
    def _update_model_status(self, key: str, backend):
        lbl = self._model_cards[key]
        state = backend.state
        color_map = {
            ModelState.UNLOADED: COLORS["subtext"],
            ModelState.LOADING:  COLORS["orange"],
            ModelState.READY:    COLORS["green"],
            ModelState.ERROR:    COLORS["red"],
        }
        lbl.configure(text=state, text_color=color_map.get(state, COLORS["subtext"]))

    def _load_model(self, key: str):
        backend = {"asr": self.asr, "llm": self.llm, "tts": self.tts}[key]
        self._model_cards[key].configure(text=ModelState.LOADING,
                                         text_color=COLORS["orange"])
        self._set_status(f"Chargement {key.upper()}…")

        def worker():
            ok  = backend.load()
            self.after(0, self._update_model_status, key, backend)
            msg = f"{key.upper()} prêt ✓" if ok else f"{key.upper()} erreur : {backend.error}"
            self.after(0, self._add_system_msg, msg,
                       COLORS["green"] if ok else COLORS["red"])
            self.after(0, self._set_status, msg)

        threading.Thread(target=worker, daemon=True).start()

    def _load_all_models(self):
        for key in ["asr", "llm", "tts"]:
            self._load_model(key)

    # ── Envoi message ─────────────────────────────────────────────────────────
    def _on_enter(self, event):
        self._send_message()

    def _send_message(self):
        text = self.text_input.get().strip()
        if not text and not self.attached_image:
            return
        self.text_input.delete(0, "end")
        self._add_message("user", text or "(image)", self.attached_image)
        image_path = self.attached_image
        self._clear_image()
        self._run_in_thread(self._process_message, text, image_path)

    def _process_message(self, text: str, image_path: str = None):
        self.after(0, self._set_status, "LLM en cours…")
        self.after(0, self.send_btn.configure, {"state": "disabled"})
        try:
            if self.llm.state != ModelState.READY:
                self.after(0, self._add_system_msg,
                           "LLM non chargé — charge d'abord les modèles.", COLORS["orange"])
                return
            reply = self.llm.chat(text or "(décrire l'image)", image_path)
            self.after(0, self._add_message, "assistant", reply)
            if self.tts.state == ModelState.READY:
                self.after(0, self._set_status, "TTS en cours…")
                wav = str(self._tmp_dir / f"tts_{int(time.time())}.wav")
                self.tts.synthesize(reply, wav,
                                    speaker=self.spk_var.get(),
                                    language=self.lang_var.get())
                self.after(0, self._play_audio, wav)
        except Exception as e:
            self.after(0, self._add_system_msg, f"Erreur : {e}", COLORS["red"])
        finally:
            self.after(0, self._set_status, "Prêt.")
            self.after(0, self.send_btn.configure, {"state": "normal"})

    # ── Enregistrement ────────────────────────────────────────────────────────
    def _start_record(self, event):
        if not AUDIO_OK:
            messagebox.showerror("Erreur", "sounddevice non installé.")
            return
        self.recording = True
        self.recorder  = AudioRecorder(device_idx=self._get_mic_idx())
        self.recorder.start()
        self.rec_btn.configure(fg_color=COLORS["record"])
        self._set_status("🔴 Enregistrement… relâchez pour arrêter.")

    def _stop_record(self, event):
        if not self.recording or not self.recorder:
            return
        self.recording = False
        self.rec_btn.configure(fg_color=COLORS["card"])
        audio = self.recorder.stop()
        if audio.size < 1000:
            self._set_status("Enregistrement trop court.")
            return
        wav_path = str(self._tmp_dir / f"rec_{int(time.time())}.wav")
        sf.write(wav_path, audio, self.recorder.samplerate, subtype="PCM_16")
        self._run_in_thread(self._process_audio, wav_path)

    def _process_audio(self, wav_path: str):
        self.after(0, self._set_status, "ASR en cours…")
        try:
            if self.asr.state != ModelState.READY:
                self.after(0, self._add_system_msg,
                           "ASR non chargé — charge d'abord les modèles.", COLORS["orange"])
                return
            text = self.asr.transcribe(wav_path, language=self.lang_var.get())
            self.after(0, self._add_message, "user", f"🎙 {text}")
            self._process_message(text)
        except Exception as e:
            self.after(0, self._add_system_msg, f"ASR Erreur : {e}", COLORS["red"])
        finally:
            if os.path.exists(wav_path):
                try: os.remove(wav_path)
                except: pass

    # ── Image ─────────────────────────────────────────────────────────────────
    def _attach_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif")])
        if not path:
            return
        self.attached_image = path
        try:
            img = Image.open(path)
            img.thumbnail((80, 60))
            self.img_preview_label.configure(
                image=ctk.CTkImage(img, size=img.size),
                text=f"  📎 {Path(path).name}", compound="left",
                text_color=COLORS["text"])
        except Exception:
            self.img_preview_label.configure(
                text=f"📎 {Path(path).name}", image=None)
        self.img_preview_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(0,4))

    def _clear_image(self):
        self.attached_image = None
        self.img_preview_frame.grid_remove()
        self.img_preview_label.configure(image=None, text="")

    # ── Lecture audio ─────────────────────────────────────────────────────────
    def _play_audio(self, path: str):
        if not AUDIO_OK:
            return
        def _play():
            try:
                data, sr = sf.read(path)
                sd.play(data, sr, device=self._get_out_idx())
                sd.wait()
            except Exception as e:
                self.after(0, self._set_status, f"Lecture erreur : {e}")
        threading.Thread(target=_play, daemon=True).start()

    # ── Tests ─────────────────────────────────────────────────────────────────
    def _test_component(self, key: str):
        if key == "llm":
            self._run_in_thread(self._test_llm)
        elif key == "tts":
            self._run_in_thread(self._test_tts)

    def _test_llm(self):
        self.after(0, self._add_system_msg, "Test LLM →", COLORS["accent"])
        self.after(0, self._set_status, "Test LLM en cours…")
        try:
            if self.llm.state != ModelState.READY:
                raise RuntimeError("LLM non chargé.")
            prompt = (f"<|im_start|>user\nDis bonjour en une phrase.<|im_end|>\n"
                      f"<|im_start|>assistant\n")
            resp   = self.llm.llm(prompt, max_tokens=64, echo=False)
            reply  = resp["choices"][0]["text"].strip()
            self.after(0, self._add_message, "assistant", f"[Test LLM] {reply}")
            self.after(0, self._set_status, "Test LLM OK ✓")
        except Exception as e:
            self.after(0, self._add_system_msg, f"Test LLM échec : {e}", COLORS["red"])
            self.after(0, self._set_status, "Test LLM échoué.")

    def _test_tts(self):
        self.after(0, self._add_system_msg, "Test TTS →", COLORS["accent"])
        self.after(0, self._set_status, "Test TTS en cours…")
        try:
            if self.tts.state != ModelState.READY:
                raise RuntimeError("TTS non chargé.")
            text = "Bonjour, le pipeline TTS fonctionne correctement."
            wav  = str(self._tmp_dir / "test_tts.wav")
            self.tts.synthesize(text, wav,
                                speaker=self.spk_var.get(),
                                language=self.lang_var.get())
            self.after(0, self._add_system_msg, f"TTS OK ✓ → {wav}", COLORS["green"])
            self.after(0, self._play_audio, wav)
            self.after(0, self._set_status, "Test TTS OK ✓")
        except Exception as e:
            self.after(0, self._add_system_msg, f"Test TTS échec : {e}", COLORS["red"])
            self.after(0, self._set_status, "Test TTS échoué.")

    def _test_full_pipeline(self):
        self._run_in_thread(self._run_full_pipeline_test)

    def _run_full_pipeline_test(self):
        self.after(0, self._add_system_msg,
                   "Test pipeline complet : ASR → LLM → TTS", COLORS["accent"])
        phrase = "Bonjour, donne-moi un fait intéressant sur l'IA en une phrase."
        self.after(0, self._add_message, "user", f"[Test] {phrase}")
        self._process_message(phrase)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _set_status(self, text: str):
        self.status_bar.configure(text=f"  {text}")

    def _clear_chat(self):
        for widget in self.chat_scroll.winfo_children():
            widget.destroy()
        self._chat_row = 0
        self.llm.reset_history()
        self._add_system_msg("Conversation réinitialisée.", COLORS["subtext"])

    def _run_in_thread(self, fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                self._job_queue.get_nowait()()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Supprimer les avertissements SoX sans rapport avec notre pipeline ──
    import warnings
    warnings.filterwarnings("ignore", message=".*SoX.*")
    warnings.filterwarnings("ignore", message=".*sox.*")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # silence oneDNN

    print(f"[INFO] OS     : {platform.system()} {platform.release()}")
    print(f"[INFO] Python : {sys.version.split()[0]}")
    print(f"[INFO] Device : {DEVICE.upper()}")
    print(f"[INFO] ASR    : {ASR_MODEL_ID}")
    print(f"[INFO] LLM    : {LLM_MODEL_PATH.name}")
    print(f"[INFO] TTS    : {TTS_MODEL_ID}")

    # ── Diagnostics détaillés ─────────────────────────────────────────────
    print("\n[DIAG] torch     :", "OK" if TORCH_OK  else f"ECHEC -> {TORCH_IMPORT_ERROR}")
    print("[DIAG] ASR       :", "OK" if ASR_OK    else f"ECHEC -> {ASR_IMPORT_ERROR}")
    print("[DIAG] llama-cpp :", "OK" if LLAMA_OK  else "non charge")
    print("[DIAG] qwen-tts  :", "OK" if QWEN_TTS_OK else "non charge")
    print("[DIAG] sounddev  :", "OK" if AUDIO_OK  else "non charge")
    print()

    missing = []
    if not AUDIO_OK:
        missing.append("pip install sounddevice soundfile")
    if not LLAMA_OK:
        missing.append("pip install llama-cpp-python")
    if not ASR_OK:
        missing.append("pip install torch torchaudio transformers")
    if not QWEN_TTS_OK:
        missing.append("pip install qwen-tts")

    if missing:
        print("[AVERTISSEMENT] Paquets potentiellement manquants :")
        for pkg in missing:
            print(f"  {pkg}")
        print()

    app = App()
    app.mainloop()
