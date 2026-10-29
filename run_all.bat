@echo off
REM Full pipeline on Windows (run from the project folder with the virtual environment active).
REM DistilBERT is the slow step: about 5 min on a GPU, much longer on CPU (see README).
cd /d "%~dp0"
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('stopwords', 'wordnet', 'omw-1.4', 'punkt_tab')]" || exit /b 1
python prepare_data.py || exit /b 1
python train_classical.py || exit /b 1
python train_distilbert.py --max-train 12000 || exit /b 1
python make_figures.py || exit /b 1
echo Demo: streamlit run app.py
