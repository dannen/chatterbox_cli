# Command-Line Frontend for Chatterbox TTS

This script provides a powerful and highly-configurable command-line interface for `resemble-ai/chatterbox`, a state-of-the-art text-to-speech model. It has been enhanced with numerous features for batch processing, audio quality control, and fully deterministic, reproducible output.

This tool was designed for users who need to generate high-quality audio from long texts in a reliable and tunable manner, going beyond basic text-to-speech functionality.

## Features

- **Text & Script Input**: Generate audio from a simple text string or an entire text file.
- **Voice Cloning**: Clone a voice from a single reference WAV file.
- **Batch Processing**: Loop through a directory of reference voices to generate audio with each one.
- **Intelligent Text Chunking**: Automatically splits long texts into manageable chunks, respecting sentence boundaries and preventing model overloads.
- **Reproducible Generation**: Achieve bit-for-bit identical audio output using a fixed seed and deterministic settings.
- **Random Seed Generation**: Easily create new, unique audio generations with a randomly generated seed that is printed for future use.
- **Advanced Generation Tuning**: Fine-tune the audio output with controls for:
    - Emotion/Prosody (`--exaggeration`)
    - Pacing (`--cfg`)
    - Repetition (`--repetition-penalty`)
    - Randomness (`--temperature`, `--top-p`)
    - Creative variations (`--chaos`)
- **Audio Quality Control**:
    - Automatic peak normalization to prevent audio clipping.
    - Configurable pauses between text chunks for natural pacing.
- **Enhanced User Experience**:
    - Color-coded console output for readability.
    - Silent mode (`--silent`) for clean integration into pipelines.
    - Debug mode (`--debug`) to show all library warnings and progress bars.

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/dannen/chatterbox_cli.git
    cd chatterbox_cli
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    This script relies on `chatterbox-tts` and its dependencies. Install them via pip.
    ```bash
    pip install chatterbox-tts torch torchaudio numpy
    ```
    *Note: For GPU acceleration, ensure you have a compatible NVIDIA driver and CUDA toolkit installed before installing PyTorch. Visit the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct installation command for your system.*

3. Install Dependencies

    This script relies on a specific fork of chatterbox-tts and must be installed directly from the GitHub repository.

    This single command will install the correct version of the library along with all of its required dependencies, such as torch, torchaudio, and numpy.
    ```
    pip install git+https://github.com/dannen/chatterbox.git
    ```
    Note: For GPU acceleration, ensure you have a compatible NVIDIA driver and CUDA toolkit installed before installing PyTorch. The command above will install the PyTorch version specified by the library, but if you run into CUDA issues, you may need to visit the PyTorch website to install a version that precisely matches your system's CUDA drivers.


## Usage

The script is run from the command line, providing various arguments to control its behavior.

### Command-Line Arguments

| Flag                    | Default | Description                                                                                |
| ----------------------- | ------- | ------------------------------------------------------------------------------------------ |
| **Input/Output** |         |                                                                                            |
| `-f, --file`            | `None`  | Path to a single reference WAV file for voice cloning.                                       |
| `--batch-dir`           | `False` | Loop through all `.wav` files in a `./source` directory.                                     |
| `-t, --text`            | `None`  | A string of text to synthesize.                                                            |
| `--script`              | `None`  | Path to a `.txt` file to use as the input text.                                            |
| `-o, --output`          | `None`  | The base name for the output audio file (e.g., `my-audio`).                                |
| **Generation Tuning** |         |                                                                                            |
| `--exaggeration`        | `0.5`   | Controls the emotional exaggeration of the voice.                                          |
| `--cfg`                 | `0.5`   | Classifier-Free Guidance weight, affects pacing and style.                                 |
| `--repetition-penalty`  | `1.2`   | Penalty for repeating words. Higher values (>1.0) discourage repetition.                   |
| `--temperature`         | `0.8`   | Controls randomness. Lower values are more deterministic.                                  |
| `--top-p`               | `1.0`   | Nucleus sampling probability.                                                              |
| `--chaos`               | `0.0`   | Applies a random delta to `--exaggeration` and `--cfg` for varied runs.                      |
| **Script Behavior** |         |                                                                                            |
| `--max-length`          | `450`   | Max character length of a text chunk. Values > 475 may cause issues.                     |
| `--pause-duration`      | `65`    | Duration of silence in milliseconds to insert between audio chunks.                          |
| `--repeat`              | `1`     | Number of times to repeat the entire generation process.                                   |
| **Reproducibility** |         |                                                                                            |
| `--seed`                | `None`  | A specific integer seed for reproducible generation.                                       |
| `--random-seed`         | `False` | Generate a random 32-bit seed for this run and print it.                                   |
| **System & Logging** |         |                                                                                            |
| `--cpu`                 | `False` | Force the script to use the CPU instead of the GPU.                                        |
| `--debug`               | `False` | Disable all warning suppression to show library warnings and progress bars.                  |
| `--silent`              | `False` | Suppress all informational console output for a quiet run.                                 |

### Examples

**1. Basic Generation with Voice Clone**
```
python3 chatterbox-cli.py -f ./source/my_voice.wav --script ./my_book.txt -o my_book_audio
```

**2. Highly Reproducible Generation with Fine-Tuning
This command uses a specific seed and tweaks the generation parameters for a specific style.
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 chatterbox-cli.py \
  --seed 12345 \
  -f ./source/my_voice.wav \
  --script ./my_book.txt \
  --exaggeration 0.7 \
  --cfg 0.2 \
  --repetition-penalty 1.5 \
  --max-length 450
```

**3. Creative Run with a Random Seed
This generates a new, unique output and prints the seed used so you can reproduce it later if you like the result.
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 chatterbox-cli.py \
  --random-seed \
  -f ./source/my_voice.wav \
  --script ./my_book.txt \
  --chaos 2.0
```

**4. Batch Processing
This will generate an audio file for every .wav file found in the ./source directory.
```
# First, create a source directory and add your voice files
mkdir source
cp my_voice1.wav my_voice2.wav ./source/

# Run the script in batch mode
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 chatterbox-cli.py --batch-dir --script ./my_book.txt
```

A Note on Full Reproducibility

To achieve 100% bit-for-bit identical audio output between runs, three conditions must be met:

    * Use the --seed flag with the same integer every time.
    * Enable PyTorch's deterministic algorithms (this is handled automatically when a seed is provided).
    * Set the CUBLAS_WORKSPACE_CONFIG environment variable before running the script, as shown in the examples. This is required to make certain GPU calculations deterministic.


Audio Normalization to Prevent Clipping

Expressive or exaggerated vocal performances can sometimes generate audio waveforms that exceed the maximum possible level, resulting in harsh digital clipping in the final audio file.

To prevent this, the script includes an automatic peak normalization step that runs just before the audio is saved. It works by:

    Scanning the entire generated audio clip to find its single loudest point (the peak value).
    If this peak exceeds the clipping threshold (i.e., its value is greater than 1.0), the entire clip's volume is scaled down proportionally so that the peak is at a safe maximum level.

This process ensures a clean, distortion-free output file while preserving the original dynamic range of the vocal performance. You will see an [INFO] message in the console (Audio peak is... Normalizing to prevent clipping.) whenever this normalization is applied.

License: 
    MIT License.
