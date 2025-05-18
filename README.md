# Parakeet-Windows-GUI

A GUI application for transcribing audio files using Parakeet TDT and CTC models, with support for CUDA acceleration and timestamp generation.

![App Screenshot](screenshot.png) *Example screenshot of the application*

## Features

- **Two Model Options**:
  - TDT (Token and Duration Transducer) model for high-quality transcription with word-level timestamps
  - CTC model for faster transcription (without timestamp support)
  
- **Output Formats**:
  - Plain text
  - SRT subtitles with timestamps
  
- **Memory Management**:
  - Automatic chunking of large audio files
  - Configurable memory thresholds
  
- **Performance**:
  - CUDA GPU acceleration support
  - Progress tracking during transcription
  
- **User-Friendly**:
  - File browsing interface
  - Copy to clipboard functionality
  - Save transcriptions to file

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/PasiKoodaa/Parakeet-Windows-GUI
   cd Parakeet-Windows-GUI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the ASR models and place them in the `models/ASR` directory:
   - `nemo-parakeet_tdt_ctc_110m.onnx`
   - `parakeet-tdt-0.6b-v2_decoder.onnx`
   - `parakeet-tdt-0.6b-v2_encoder.onnx`
   - `parakeet-tdt-0.6b-v2_joiner.onnx`
   - `parakeet-tdt-0.6b-v2_model_config.yaml`
   - `parakeet-tdt_ctc-110m_model_config.yaml`
   - `phomenizer_en.onnx`

## Usage

Run the application:
```bash
python app.py
```

### Command Line Options
```bash
python app.py --help
```

Options:
- `--gui`: Launch GUI (default if no other options specified)
- `--model`: Model type (`tdt` or `ctc`)
- `--input`: Input audio file
- `--output`: Output text file
- `--cuda`: Use CUDA if available
- `--timestamps`: Include timestamps (TDT model only)
- `--level`: Timestamp level (`word`, `segment`, `char`)
- `--format`: Output format (`text` or `srt`)

## Credits

This project is based on code from [dnhkng/GLaDOS](https://github.com/dnhkng/GLaDOS), which ported the Parakeet models to Windows. Most of the core ASR functionality is derived from their work.
