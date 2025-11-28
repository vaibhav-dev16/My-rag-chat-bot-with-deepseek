using the python version 3.10

# Setup Instructions

## 1. Install Poetry
```bash
pip install poetry
```
## 2. Install Dependencies
```bash
poetry install
```
## 3. Python Version Requirement
Ensure your Python version is above 3.10 and below 3.13.
To use Python 3.10 with Poetry:
```bash
poetry env use 3.10
```
## 4. Environment Variables
```bash
cp .env.example .env
```
## 5. Activate Virtual Environment 
Windows:
```bash
.venv\Scripts\activate
```
Linux / macOS:
```bash
source .venv/bin/activate
```
## 6. Run Streamlit App
```bash
streamlit run rag_app/main.py
```
