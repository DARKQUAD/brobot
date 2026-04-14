#!/usr/bin/env python3
"""
Pipeline Qwen3 ASR → Qwen3.5 LLM → Qwen3 TTS  (Windows / Debian / macOS)
--------------------------------------------------------------------------
Modèles :
  ASR  : Qwen/Qwen3-ASR-0.6B          via transformers (téléchargé auto HF)
  LLM  : Qwen_Qwen3.5-4B-Q5_K_M.gguf  via llama-cpp-python (local GGUF)
  TTS  : Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice  via qwen-tts (téléchargé auto HF)

Installation :
    pip install torch torchaudio transformers soundfile sounddevice
    pip install llama-cpp-python
    pip install qwen-tts

Usage :
    python pipeline.py --text "Bonjour, comment vas-tu ?"
    python pipeline.py --audio question.wav --output reponse.wav
    python pipeline.py --interactive
"""

import argparse
import os
import platform
import sys
import time
import wave
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────
# Paramètres globaux
# ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()

# Modèle LLM local (GGUF)
LLM_MODEL_PATH = str(SCRIPT_DIR / "Qwen_Qwen3.5-4B-Q5_K_M.gguf")

# Identifiants HuggingFace (téléchargement automatique au premier lancement)
ASR_MODEL_ID  = "Qwen/Qwen3-ASR-0.6B"
TTS_MODEL_ID  = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

DEFAULT_LANGUAGE = "French"
DEFAULT_SPEAKER  = "Serena"      # serena, vivian, ryan, aiden, eric, dylan, …
DEFAULT_OUTPUT   = "output_tts.wav"
N_THREADS        = max(4, os.cpu_count() or 4)

SYSTEM_PROMPT = (
    "Tu es un assistant vocal utile et concis. "
    "Réponds toujours dans la même langue que l'utilisateur. "
    "Tes réponses doivent être courtes et fluides pour être converties en parole."
)

# ── Détection automatique GPU/CPU ─────────────────────────────────
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
except Exception:    # ImportError OU OSError (DLL torchvision incompatible, etc.)
    DEVICE = "cpu"
    TORCH_DTYPE = None


def _info(msg: str):
    print(f"[INFO] {msg}")


def _check_import(pkg_name: str, install_cmd: str):
    """Lève une ImportError claire si un paquet est manquant."""
    try:
        __import__(pkg_name)
    except ImportError:
        raise ImportError(
            f"Paquet '{pkg_name}' introuvable.\n"
            f"Installez-le avec : {install_cmd}"
        )


# ══════════════════════════════════════════════
# Étape 1 : ASR — Qwen3-ASR-0.6B (transformers)
# ══════════════════════════════════════════════
def run_asr(audio_path: str, language: str = DEFAULT_LANGUAGE) -> str:
    """
    Transcrit un fichier audio WAV en texte avec Qwen3-ASR-0.6B.
    Utilise la bibliothèque HuggingFace `transformers`.
    Le modèle est téléchargé automatiquement au premier lancement (~300 Mo).
    """
    print(f"\n[ASR] Transcription : {audio_path}")
    t0 = time.time()

    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

    _check_import("torch",        "pip install torch torchaudio")
    _check_import("transformers", "pip install transformers")
    _check_import("soundfile",    "pip install soundfile")

    import torch
    import soundfile as sf
    # transformers ≥4.46 renamed AutoModelForSpeech2Seq → AutoModelForSpeechSeq2Seq
    try:
        from transformers import AutoModelForSpeechSeq2Seq as _ASRModel, AutoProcessor
    except ImportError:
        from transformers import AutoModelForSpeech2Seq as _ASRModel, AutoProcessor

    lang_map = {
        "French": "fr", "English": "en", "Chinese": "zh",
        "Japanese": "ja", "German": "de", "Spanish": "es",
        "Korean": "ko", "Russian": "ru", "Italian": "it",
    }
    lang_code = lang_map.get(language, language.lower()[:2])

    # Chargement du modèle (mis en cache après le premier téléchargement)
    _info(f"Chargement Qwen3-ASR depuis {ASR_MODEL_ID} (1ère fois : téléchargement ~300 Mo)…")
    processor = AutoProcessor.from_pretrained(ASR_MODEL_ID, trust_remote_code=True)
    model = _ASRModel.from_pretrained(
        ASR_MODEL_ID,
        torch_dtype=TORCH_DTYPE or torch.float32,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()

    # Lecture audio et mise dans le bon format (16 kHz, mono)
    audio_array, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Rééchantillonnage à 16 kHz si nécessaire
    if sample_rate != 16_000:
        try:
            import torchaudio
            waveform = torch.tensor(audio_array).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, 16_000)
            audio_array = resampler(waveform).squeeze(0).numpy()
            sample_rate = 16_000
        except ImportError:
            _info("torchaudio absent — rééchantillonnage ignoré (qualité réduite)")

    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        language=lang_code,
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"[ASR] ✓ ({time.time()-t0:.1f}s) → \"{text}\"")
    return text


# ══════════════════════════════════════════════
# Étape 2 : LLM — Qwen3.5-4B GGUF (llama-cpp)
# ══════════════════════════════════════════════
def run_llm(
    user_text: str,
    history: list | None = None,
    max_tokens: int = 512,
    n_ctx: int = 4096,
    n_gpu_layers: int = 0,
) -> str:
    """
    Génère une réponse avec Qwen3.5-4B GGUF via llama-cpp-python.
    Le modèle GGUF doit être présent localement dans le même dossier.
    """
    print("\n[LLM] Génération de la réponse…")
    t0 = time.time()

    _check_import("llama_cpp", "pip install llama-cpp-python")

    if not Path(LLM_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Modèle LLM introuvable : {LLM_MODEL_PATH}\n"
            f"Placez le fichier GGUF dans : {SCRIPT_DIR}"
        )

    from llama_cpp import Llama

    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=n_ctx,
        n_threads=N_THREADS,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_text})

    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    output = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )
    reply = output["choices"][0]["text"].strip()
    print(f"[LLM] ✓ ({time.time()-t0:.1f}s) → \"{reply}\"")
    return reply


# ══════════════════════════════════════════════
# Étape 3 : TTS — Qwen3-TTS-0.6B (qwen-tts)
# ══════════════════════════════════════════════
def run_tts(
    text: str,
    output_path: str,
    speaker: str = DEFAULT_SPEAKER,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """
    Synthétise 'text' en WAV avec Qwen3-TTS-0.6B-CustomVoice.
    Utilise le package officiel `qwen-tts` (pip install qwen-tts).
    Le modèle est téléchargé automatiquement au premier lancement (~1.5 Go).
    """
    print(f"\n[TTS] Synthèse vocale… (voix : {speaker}, langue : {language})")
    t0 = time.time()

    _check_import("qwen_tts",   "pip install qwen-tts")
    _check_import("torch",      "pip install torch")
    _check_import("soundfile",  "pip install soundfile")

    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    lang_map = {
        "French": "French", "English": "English", "Chinese": "Chinese",
        "Japanese": "Japanese", "German": "German", "Spanish": "Spanish",
        "Korean": "Korean", "Russian": "Russian", "Italian": "Italian",
    }
    tts_language = lang_map.get(language, "English")

    _info(f"Chargement Qwen3-TTS depuis {TTS_MODEL_ID} (1ère fois : téléchargement ~1.5 Go)…")

    device_map = DEVICE if DEVICE == "cuda" else "cpu"
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    model = Qwen3TTSModel.from_pretrained(
        TTS_MODEL_ID,
        device_map=device_map,
        dtype=dtype,
    )

    wavs, sr = model.generate_custom_voice(
        text=text,
        language=tts_language,
        speaker=speaker.capitalize(),
    )

    sf.write(output_path, wavs[0], sr)
    print(f"[TTS] ✓ ({time.time()-t0:.1f}s) → {output_path}")
    return output_path


# ══════════════════════════════════════════════
# Pipeline complet
# ══════════════════════════════════════════════
def run_pipeline(
    audio_path: str | None = None,
    text_input: str | None = None,
    output_path: str = DEFAULT_OUTPUT,
    speaker: str = DEFAULT_SPEAKER,
    language: str = DEFAULT_LANGUAGE,
    n_gpu_layers: int = 0,
) -> dict:
    results: dict = {}
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  Pipeline  Qwen3-ASR → Qwen3.5-LLM → Qwen3-TTS")
    print(f"  OS : {platform.system()}  |  GPU : {DEVICE.upper()}")
    print(sep)

    # 1. ASR
    if audio_path:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio introuvable : {audio_path}")
        transcription = run_asr(audio_path, language=language)
    elif text_input:
        transcription = text_input
        print(f"\n[ASR] Entrée texte directe : \"{transcription}\"")
    else:
        raise ValueError("Fournissez --audio ou --text.")
    results["transcription"] = transcription

    # 2. LLM
    llm_reply = run_llm(transcription, n_gpu_layers=n_gpu_layers)
    results["llm_reply"] = llm_reply

    # 3. TTS
    wav_path = run_tts(llm_reply, output_path, speaker=speaker, language=language)
    results["output_wav"] = wav_path

    print(f"\n{sep}")
    print("  Pipeline terminé avec succès ✓")
    print(f"  Audio de sortie : {wav_path}")
    print(sep)
    return results


# ══════════════════════════════════════════════
# Mode interactif
# ══════════════════════════════════════════════
def interactive_mode(
    speaker: str = DEFAULT_SPEAKER,
    language: str = DEFAULT_LANGUAGE,
    n_gpu_layers: int = 0,
):
    """Boucle interactive : micro → ASR → LLM → TTS → lecture."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("[ERREUR] pip install sounddevice soundfile")
        sys.exit(1)

    RECORD_SR = 16_000
    history: list = []
    tmp_dir = SCRIPT_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\n[Interactif] Assistant Qwen3  ({platform.system()})")
    print("[Interactif] Tapez 'q' + Entrée pour quitter, Entrée pour parler.\n")

    session = 0
    while True:
        cmd = input(">>> Appuyez sur Entrée pour parler (q + Entrée pour quitter) : ").strip().lower()
        if cmd == "q":
            print("[Interactif] Au revoir !")
            break

        DURATION = 5
        print(f"[Interactif] Enregistrement {DURATION}s… Parlez !")
        rec = sd.rec(int(DURATION * RECORD_SR), samplerate=RECORD_SR, channels=1, dtype="float32")
        sd.wait()
        print("[Interactif] Enregistrement terminé.")

        tmp_wav = str(tmp_dir / f"rec_{session}.wav")
        sf.write(tmp_wav, rec, RECORD_SR, subtype="PCM_16")
        session += 1

        try:
            text = run_asr(tmp_wav, language=language)
            print(f"[Interactif] Vous avez dit : \"{text}\"")

            reply = run_llm(text, history=history, n_gpu_layers=n_gpu_layers)
            history.append({"role": "user",      "content": text})
            history.append({"role": "assistant", "content": reply})
            if len(history) > 20:
                history = history[-20:]

            out_wav = str(tmp_dir / f"tts_{session}.wav")
            run_tts(reply, out_wav, speaker=speaker, language=language)

            data, sr = sf.read(out_wav)
            print("[Interactif] Lecture…")
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print(f"[ERREUR] {e}")
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)


# ══════════════════════════════════════════════
# Point d'entrée CLI
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Qwen3-ASR → Qwen3.5-LLM → Qwen3-TTS (cross-platform)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--audio",       "-a", help="Fichier audio WAV/MP3 d'entrée")
    group.add_argument("--text",        "-t", help="Texte direct (bypasse ASR)")
    group.add_argument("--interactive", "-i", action="store_true", help="Mode microphone")

    parser.add_argument("--output",    "-o", default=DEFAULT_OUTPUT,
                        help=f"Fichier WAV de sortie (défaut : {DEFAULT_OUTPUT})")
    parser.add_argument("--speaker",   "-s", default=DEFAULT_SPEAKER,
                        help=f"Voix TTS CustomVoice (défaut : {DEFAULT_SPEAKER})")
    parser.add_argument("--language",  "-l", default=DEFAULT_LANGUAGE,
                        help=f"Langue (défaut : {DEFAULT_LANGUAGE})")
    parser.add_argument("--gpu-layers","-g", type=int, default=0,
                        help="Couches GPU llama-cpp (0=CPU, -1=tout)")

    args = parser.parse_args()

    print(f"[INFO] OS     : {platform.system()} {platform.release()}")
    print(f"[INFO] Python : {sys.version.split()[0]}")
    print(f"[INFO] Device : {DEVICE.upper()}")
    print(f"[INFO] ASR    : {ASR_MODEL_ID}")
    print(f"[INFO] LLM    : {Path(LLM_MODEL_PATH).name}")
    print(f"[INFO] TTS    : {TTS_MODEL_ID}")

    if args.interactive:
        interactive_mode(speaker=args.speaker, language=args.language,
                         n_gpu_layers=args.gpu_layers)
    elif args.audio or args.text:
        run_pipeline(
            audio_path=args.audio,
            text_input=args.text,
            output_path=args.output,
            speaker=args.speaker,
            language=args.language,
            n_gpu_layers=args.gpu_layers,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
