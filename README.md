# Automated-Plateform-Contract_Management_Streamlit
Streamlit platform for OCR, benchmarking, TOC extraction via NLP and chatbot.

## System Dependencies

Before running the app, install required system packages:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
# 🌊 Automated Platform - Contract Management Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive platform for OCR processing, contract analysis, geospatial risk mapping, and fraud detection for reinsurance contracts.

## 🎯 Features

- **🔍 Advanced OCR Processing**: Multiple OCR engines (EasyOCR, Tesseract, Mistral OCR)
- **📊 Contract Analysis**: Automated legal compliance checking against Moroccan insurance regulations
- **🗺️ Geospatial Risk Mapping**: Interactive maps with risk zones and contract locations
- **📈 Descriptive Statistics**: Comprehensive data analysis and visualization
- **🎯 Fraud Detection**: AI-powered anomaly detection system
- **⚖️ Legal Compliance**: Automated verification against Code des Assurances
- **📄 Reporting System**: Generate comprehensive compliance reports

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Tesseract OCR (for Tesseract engine)
- System dependencies (see Installation)

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/Boutaina19/Automated-Plateform-Contract_Management_Streamlit.git
cd Automated-Plateform-Contract_Management_Streamlit
```

#### 2. Install system dependencies

**For Ubuntu/Debian:**
```bash
chmod +x setup.sh
sudo ./setup.sh
```

**Or manually:**
```bash
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-ara
```

**For macOS:**
```bash
brew install tesseract tesseract-lang
```

**For Windows:**
- Download Tesseract from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- Install to `C:\Tesseract-OCR\`

#### 3. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

#### 5. Download spaCy models

```bash
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
```

#### 6. Set up environment variables

Create a `.env` file:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
TESSERACT_PATH=/usr/bin/tesseract  # Adjust for your system
```

### Running the Application

```bash
streamlit run Streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🐳 Docker Deployment

### Build and run with Docker

```bash
docker build -t contract-management-app .
docker run -p 8501:8501 contract-management-app
```

### Using Docker Compose

```bash
docker-compose up
```

## 📦 Project Structure

```
Automated-Plateform-Contract_Management_Streamlit/
├── Streamlit_app.py          # Main application file
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── setup.sh                   # System setup script
├── .gitignore                # Git ignore rules
├── README.md                  # This file
├── logo.png                   # Application logo
├── users.json                 # User authentication data
├── .env.example              # Environment variables template
└── docs/                      # Documentation
    ├── user_guide.md
    ├── api_reference.md
    └── deployment.md
```

## 🔐 Security Notes

- Never commit `users.json` with real passwords
- Store API keys in `.env` file (not in code)
- Use strong passwords (minimum 8 characters)
- Enable 2FA for production deployments

## 🛠️ Configuration

Edit `CONFIG` dictionary in `Streamlit_app.py`:

```python
CONFIG = {
    "logo_path": "logo.png",
    "users_file": "users.json",
    "tesseract_path": "/usr/bin/tesseract",
    "default_mistral_api_key": "your_api_key"
}
```

## 📊 Usage Examples

### 1. OCR Processing

```python
# Upload a contract PDF
# Select OCR engine (EasyOCR, Tesseract, or Mistral)
# Choose language
# Extract text
```

### 2. Legal Compliance Check

```python
# Upload contract and legal reference
# System automatically checks compliance
# View detailed compliance report
# Export results
```

### 3. Geospatial Analysis

```python
# Import contract data with GPS coordinates
# Visualize risk zones on interactive map
# Perform hotspot analysis
# Calculate distances and areas
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Boutaina19** - *Initial work* - [GitHub](https://github.com/Boutaina19)

## 🙏 Acknowledgments

- Streamlit for the amazing framework
- Mistral AI for OCR capabilities
- OpenStreetMap for mapping data
- All contributors and users

## 📧 Contact

- GitHub: [@Boutaina19](https://github.com/Boutaina19)
- Project Link: [https://github.com/Boutaina19/Automated-Plateform-Contract_Management_Streamlit](https://github.com/Boutaina19/Automated-Plateform-Contract_Management_Streamlit)

## 🐛 Known Issues

- EasyOCR performance issues with large PDFs (>50 pages)
- Tesseract requires manual installation on Windows
- Some geospatial features require additional system libraries

## 🗺️ Roadmap

- [ ] Add multi-user collaboration features
- [ ] Implement real-time contract monitoring
- [ ] Add blockchain integration for contract verification
- [ ] Mobile app version
- [ ] API endpoints for integration
- [ ] Advanced ML models for fraud detection

---

**⭐ If you find this project useful, please consider giving it a star!**
