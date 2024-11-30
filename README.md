# README for KB Article Analysis Tool

This repository provides a Python-based tool to analyze Windows Update KB articles, predict vulnerabilities using machine learning, and generate a detailed HTML report. The tool integrates data extraction, processing, prediction, and sentiment analysis to help users manage and understand their Windows environment's updates and potential risks.

Features
Retrieve Windows Version: Identifies the installed Windows version for compatibility checks.
Installed Updates Extraction: Lists installed KB updates using PowerShell.
CSV Comparison: Compares installed KB numbers with a provided CSV file of all KB updates.
Future Vulnerability Prediction: Uses a pre-trained Random Forest model to predict future vulnerability scores.
Web Scraping: Retrieves user feedback and article metadata from Microsoft Answers for KB updates.
Sentiment Analysis: Analyzes user feedback sentiment for KB articles using a pre-trained logistic regression model.
HTML Reporting: Generates a detailed and interactive HTML report, including tables, graphs, and pie charts.
Prerequisites
Python 3.8 or later
Required Python libraries (install via requirements.txt):
pandas
numpy
matplotlib
seaborn
selenium
scikit-learn
joblib
webdriver_manager
Chrome WebDriver: Automatically managed by webdriver_manager.
PowerShell: Necessary for extracting installed KB updates.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/kb-article-analysis-tool.git
cd kb-article-analysis-tool
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Add the required CSV file (combined_csv.csv) with KB articles to the root directory.

Add the machine learning model files:

vulnerability_model.pkl: For vulnerability prediction.
logistic_regression_model.pkl and tfidf_vectorizer.pkl: For sentiment analysis.
Usage
Run the script:

bash
Copy code
python kb_analysis_tool.py
The tool will:

Retrieve your Windows version and installed KB updates.
Compare installed KB numbers with those in the CSV file.
Predict future vulnerabilities based on a machine learning model.
Scrape user feedback for relevant KB articles.
Perform sentiment analysis on the scraped data.
Generate an HTML report (kb_report.html) in the root directory.
Directory Structure
plaintext
Copy code
├── combined_csv.csv              # CSV containing KB articles
├── kb_analysis_tool.py           # Main Python script
├── vulnerability_model.pkl       # Model for vulnerability prediction
├── logistic_regression_model.pkl # Model for sentiment analysis
├── tfidf_vectorizer.pkl          # TF-IDF vectorizer for text processing
├── requirements.txt              # Dependencies
├── kb_articles.txt               # Extracted KB articles
├── *.txt                         # Scraped feedback files (generated)
├── *.png                         # Sentiment pie charts (generated)
├── kb_report.html                # Generated HTML report
Example Workflow
Extract Updates: Lists all installed KB updates using PowerShell.
Analyze Updates: Compares installed KB updates with combined_csv.csv to find matches.
Predict Vulnerabilities: Forecasts future vulnerability scores for upcoming months.
Web Scraping: Retrieves titles and user feedback for relevant KB updates from Microsoft Answers.
Sentiment Analysis: Evaluates user feedback to determine overall sentiment (positive/negative).
HTML Report: Creates a comprehensive, interactive report with detailed insights.
