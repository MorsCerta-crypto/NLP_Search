Write-Host Creating virtual environment
python -m venv venv
Write-Host Activating virtual environment
.\venv\Scripts\Activate.ps1
Write-Host Installing required packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\setup.py install
Write-Host Installing spacy language package
python -m spacy download de_core_news_lg
Write-Host Successfully set up virtual evironment