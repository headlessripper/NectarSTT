# 🍯 NectarSTT  
**Nectar Speech-to-Text Engine**

NectarSTT (Nectar Speech To Text) is a Python-based speech recognition engine designed for real-time, offline-capable voice input. It is built to be modular, extensible, and suitable for AI assistants, automation systems, and accessibility tools.

This project focuses on **accurate speech recognition**, **low latency**, and **tight integration with AI pipelines**.

---

## ✨ Features

- 🎙️ Real-time speech-to-text
- 🧠 Modular engine design (easy to extend)
- ⚡ Optimized for low latency
- 🔌 Designed to integrate with AI / assistant systems
- 🖥️ Cross-platform (Windows, Linux)
- 🧩 Compatible with TTS pipelines (Piper / eSpeak NG)

---

## 🛠️ Installation

### 0️⃣ Direct install
```
pip install nectarstt
```

- `OR if you want code`

### 1️⃣ Clone the repository
```bash
git clone https://github.com/headlessripper/NectarSTT.git
cd NectarSTT
```
---

### 2️⃣ Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate 

### 3️⃣ Install dependencies
pip install -r requirements.txt

---

## 📁 Usage Examples

### Direct use after installation
- `to use the GUI for NectarSTT run:`
    ``` 
    NectarSTT
    ```  
---
- `use in code:`
### TTS:
```python
from nectarstt import piper_tts

piper_tts("This is top-level API usage.", "out.wav")

```

---

### STT:
```python
from nectarstt import record_until_silence

text = record_until_silence()

```

> ⚠️ **Note:** Before you use download the Main-Engine from [Engine](https://github.com/headlessripper/NectarSTT/releases/download/1.0/Main-Engine.zip) then extract it to the root folder of your project.

---

## 📦 Models & Assets

Due to GitHub size limits, speech models and voice data are zipped into **Main-Engine.zip**.  
This archive contains:

- `Main-Engine/Model/`
- `Main-Engine/TTS-Engine/`
- `Main-Engine/STT-Engine/`
- `Main-Engine/Images/`
- `Main-Engine/Sound/`
- `Main-Engine/Source/`

Extract `Main-Engine.zip` into the project directory before running NectarSTT.

> 💡 A setup script or model downloader may be added in future releases.

---

## 🚀 Roadmap / Ideas

- Automated model download & setup script
- Extended TTS engine support
- Additional language models
- Optional Futher GPU acceleration (where supported)
- Enhanced logging and debugging tools

---

## 🤝 Contributing

Contributions are welcome!

You can:

- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests

Please open an issue to discuss major changes before starting work.

---

## 📜 License
Use a Custom License
---

## ⭐ Support

If you find **NectarSTT** useful:

- ⭐ Star the repository
- 🐞 Report issues
- 💬 Share feedback and ideas

---

**Built with ❤️ in Python for high-quality, low-latency STT And TTS Engine**
