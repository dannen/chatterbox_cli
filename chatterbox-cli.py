#!/usr/bin/env python3

import argparse
import os
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from datetime import datetime
import re
import textwrap
import random
import numpy as np
import sys
from contextlib import contextmanager
import warnings

# Suppress the UserWarning from perth about pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

class TermColors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

@contextmanager
def suppress_stderr():
    """A context manager that redirects the low-level stderr file descriptor to devnull."""
    original_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, original_stderr_fd)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stderr_fd)

@contextmanager
def null_context():
    """A dummy context manager that does nothing."""
    yield

DEFAULT_MAX_LENGTH = 450

def split_text_into_sentences(text, max_len):
    """
    Splits text into chunks that are strictly under max_len, respecting sentence
    and word boundaries wherever possible.
    """
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    
    processed_parts = []
    for sentence in sentences:
        if len(sentence) > max_len:
            processed_parts.extend(textwrap.wrap(sentence, max_len, break_long_words=False, replace_whitespace=False))
        else:
            processed_parts.append(sentence)

    chunks = []
    current_chunk_sentences = []
    for part in processed_parts:
        if current_chunk_sentences and len(" ".join(current_chunk_sentences)) + 1 + len(part) > max_len:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [part]
        else:
            current_chunk_sentences.append(part)

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return [c for c in chunks if c]


def generate_output_filename(base="output", ext=".wav", ex=None, cfg=None, seed=None):
    now = datetime.now().strftime("%Y%m%d%M%S")
    tag = ""
    if ex is not None and cfg is not None:
        e_tag = f"e{int(round(ex * 100))}"
        c_tag = f"c{int(round(cfg * 100))}"
        tag = f"_{e_tag}_{c_tag}"
    
    seed_tag = f"_s{seed}" if seed is not None else ""
    
    return f"{base}_{now}{tag}{seed_tag}{ext}"

def main():
    parser = argparse.ArgumentParser(description="Generate TTS with Chatterbox.")
    # ... (all arguments remain the same) ...
    parser.add_argument('-f', '--file', type=str, help='Single reference WAV file for voice cloning')
    parser.add_argument('--batch-dir', action='store_true',
                        help='Loop through all .wav files in ./source and apply the script to each')
    parser.add_argument('-t', '--text', type=str, help='Text to synthesize (ignored if --script is used)')
    parser.add_argument('--script', type=str, help='Path to a .txt file to use as the input text instead of -t')
    parser.add_argument('-o', '--output', type=str, help='Base output filename (no extension; .wav will be added)')
    parser.add_argument('--exaggeration', type=float, default=0.5, help='Emotion exaggeration (default: 0.5)')
    parser.add_argument('--cfg', type=float, default=0.5, help='CFG weight for pacing (default: 0.5)')
    parser.add_argument('--repeat', '-r', type=int, default=1, help='Number of times to repeat synthesis (default: 1)')
    parser.add_argument('--chaos', type=float, default=0.0,
                        help='Randomize exaggeration and cfg by ±value × 0.1 (e.g., --chaos 2 = ±0.2)')
    parser.add_argument('--pause-duration', type=float, default=65, help='Duration of pause in milliseconds to insert between audio chunks.')
    parser.add_argument('--max-length', type=int, default=DEFAULT_MAX_LENGTH, help=f'Max character length of a text chunk. Values > 475 may cause issues. (default: {DEFAULT_MAX_LENGTH})')
    parser.add_argument('--repetition-penalty', type=float, default=1.2, help='Penalty for repeating tokens. Higher values discourage repetition. (Default: 1.2)')
    parser.add_argument('--temperature', type=float, default=0.8, help='Controls randomness. Lower values are more deterministic. (Default: 0.8)')
    parser.add_argument('--top-p', type=float, default=1.0, help='Nucleus sampling probability. (Default: 1.0)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU use')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible generation.')
    parser.add_argument('--random-seed', action='store_true', help='Generate a random 32-bit seed and use it for this run.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, which shows all warnings and progress bars.')
    parser.add_argument('--silent', action='store_true', help='Suppress all console output except fatal errors.')

    args = parser.parse_args()

    if args.silent:
        def log(*args, **kwargs):
            pass
    else:
        log = print

    if args.debug:
        suppress_context_factory = null_context
        log(f"{TermColors.BLUE}[INFO]{TermColors.RESET} Debug mode enabled. All warnings will be displayed.")
    else:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*LlamaModel is using LlamaSdpaAttention.*")
        warnings.filterwarnings("ignore", message=".*We detected that you are passing `past_key_values`.*")
        suppress_context_factory = suppress_stderr
    
    text_input = ""
    if args.script:
        try:
            with open(args.script, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                text_input = " ".join(lines)
        except FileNotFoundError:
            __builtins__.print(f"{TermColors.RED}[ERROR]{TermColors.RESET} Script file not found: {args.script}")
            return
    elif args.text:
        text_input = args.text.strip()

    if not text_input:
        __builtins__.print(f"{TermColors.RED}[ERROR]{TermColors.RESET} No input text provided. Use -t or --script.")
        return

    device = "cpu" if args.cpu else "cuda"
    
    log(f"{TermColors.BLUE}[INFO]{TermColors.RESET} Loading model on {device}...")
    with suppress_context_factory():
        model = ChatterboxTTS.from_pretrained(device=device)

    text_chunks = split_text_into_sentences(text_input, args.max_length)

    if args.batch_dir:
        source_dir = "./source"
        if not os.path.isdir(source_dir):
            __builtins__.print(f"{TermColors.RED}[ERROR]{TermColors.RESET} Source directory for batch mode not found: {source_dir}")
            return
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        voice_files = sorted([
            os.path.join(source_dir, f)
            for f in os.listdir(source_dir)
            if f.lower().endswith(".wav")
        ])
    else:
        voice_files = [args.file]

    for run in range(args.repeat):
        # <<< START MODIFIED BLOCK: Seeding logic is now inside the repeat loop
        
        # A copy of the seed argument is made to be potentially modified for this run
        current_run_seed = args.seed

        if args.random_seed:
            current_run_seed = random.randint(0, 2**32 - 1)
            log(f"{TermColors.BLUE}[INFO]{TermColors.RESET} Run {run + 1}: Generated random seed: {current_run_seed}")
        
        if current_run_seed is not None:
            # If we are on run 1, print the main info message.
            if run == 0:
                 log(f"{TermColors.BLUE}[INFO]{TermColors.RESET} Using random seed: {current_run_seed} and enabling deterministic mode.")
            
            random.seed(current_run_seed)
            np.random.seed(current_run_seed)
            torch.manual_seed(current_run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_run_seed)
            
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        # <<< END MODIFIED BLOCK

        for voice_path in voice_files:
            voice_label = os.path.basename(voice_path) if voice_path else "none"
            log(f"\n{TermColors.BLUE}[INFO]{TermColors.RESET} Run {run + 1} - Voice: {voice_label}")

            if voice_path:
                log(f"{TermColors.BLUE}[INFO]{TermColors.RESET} Conditioning voice from {voice_label}...")
                with suppress_context_factory():
                    model.prepare_conditionals(voice_path, exaggeration=args.exaggeration)

            log(f"{TermColors.BLUE}[INFO]{TermColors.RESET} Generating audio for {len(text_chunks)} chunk(s)...")

            if args.chaos:
                delta = args.chaos * 0.1
                ex = max(0.0, min(args.exaggeration + random.uniform(-delta, delta), 1.5))
                cfg = max(0.0001, min(args.cfg + random.uniform(-delta, delta), 1.0))
            else:
                ex = args.exaggeration
                cfg = args.cfg

            all_audio = []

            for idx, chunk in enumerate(text_chunks):
                log(f"  ↳ {TermColors.GREEN}Chunk{TermColors.RESET} {idx + 1} [{len(chunk)} chars]: \"{TermColors.YELLOW}{chunk}{TermColors.RESET}\"")

                kwargs = {
                    "text": chunk,
                    "exaggeration": ex,
                    "cfg_weight": cfg,
                    "seed": current_run_seed, # Use the seed for the current run
                    "repetition_penalty": args.repetition_penalty,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
                
                with suppress_context_factory():
                    audio = model.generate(**kwargs)
                all_audio.append(audio)
                
                if args.pause_duration > 0 and idx < len(text_chunks) - 1:
                    pause_in_seconds = args.pause_duration / 1000
                    pause_samples = int(pause_in_seconds * model.sr)
                    pause_tensor = torch.zeros((1, pause_samples))
                    all_audio.append(pause_tensor)

            combined = torch.cat(all_audio, dim=1)

            if args.output:
                base, _ = os.path.splitext(args.output)
            elif voice_path:
                voice_base = os.path.splitext(os.path.basename(voice_path))[0]
                base = os.path.join("./output", voice_base)
            else:
                base = "output"

            output_path = generate_output_filename(base, ex=ex, cfg=cfg, seed=current_run_seed) # Use the seed for the current run
            
            peak_value = torch.max(torch.abs(combined))
            if peak_value > 1.0:
                log(f"{TermColors.BLUE}[INFO]{TermColors.RESET} Audio peak is {peak_value:.4f}. Normalizing to prevent clipping.")
                audio_to_save = combined / peak_value
            else:
                audio_to_save = combined
            
            ta.save(output_path, audio_to_save, model.sr)
            log(f"{TermColors.GREEN}[DONE]{TermColors.RESET} Output saved to {output_path}")

if __name__ == '__main__':
    main()
