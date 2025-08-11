# nlp-resume-matcher
# 🧠 NLP Resume-to-Job Matcher

**🔍 An intelligent resume-job matching system using NLP and vector similarity**

---

## ✨ Key Features
- Automatic matching of resumes to job opportunities
- Utilizes **spaCy's language model** for semantic understanding
- Calculates similarity using **Euclidean distance** between keyword vectors
- Simple **Streamlit** web interface

## 🛠️ Technologies Used
- Python 3
- spaCy (en_core_web_lg)
- Streamlit
- NumPy

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/nlp-resume-matcher.git
cd nlp-resume-matcher

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 4. Run the application
streamlit run matcher_app.py
