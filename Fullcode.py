import subprocess
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time

def get_windows_version():
    try:
        process = subprocess.Popen(["wmic", "os", "get", "Caption"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result, error = process.communicate()

        if process.returncode != 0:
            print("Failed to retrieve Windows version:", error.decode())
            return None

        version_info = result.decode().splitlines()

        version_info = [line.strip() for line in version_info if line.strip()]

        if len(version_info) > 1:
            windows_version = version_info[1]
        else:
            windows_version = "Unknown"

        return windows_version

    except Exception as e:
        print("An error occurred while retrieving the Windows version:", str(e))
        return None

def extract_clean_windows_version(full_version):
    if "Windows 10" in full_version:
        return "Windows 10"
    elif "Windows 11" in full_version:
        return "Windows 11"
    else:
        return None

def get_installed_updates():
    try:
        process = subprocess.Popen(["powershell", "-Command", "Get-HotFix | Select-Object -Property HotFixID"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result, error = process.communicate()

        if process.returncode != 0:
            print("Failed to retrieve updates:", error.decode())
            return None

        updates = result.decode()
        lines = updates.split('\n')
        kb_numbers = []
        for line in lines:
            if 'KB' in line:
                kb_number = line.split('KB')[1].strip()
                kb_numbers.append(kb_number)

        return kb_numbers

    except Exception as e:
        print("An error occurred:", str(e))
        return None

def compare_with_csv(kb_numbers, csv_path):
    try:
        if not os.path.exists(csv_path):
            print("CSV file does not exist:", csv_path)
            return None

        df = pd.read_csv(csv_path)

        df['Article'] = df['Article'].dropna().astype(str).str.replace('KB', '')
        
        matching_kb_numbers = set(kb_numbers).intersection(df['Article'])

        return df, matching_kb_numbers
    except Exception as e:
        print("An error occurred while processing the CSV file:", str(e))
        return None, None

def list_latest_kb_articles(df, most_recent_kb_date, windows_version):
    try:
        df['Release date'] = pd.to_datetime(df['Release date'])

        filtered_df = df[df['Release date'] >= pd.to_datetime(most_recent_kb_date)]

        if filtered_df.empty:
            print(f"No updates found in the CSV file on or after {most_recent_kb_date}.")
            return None

        relevant_df = filtered_df[filtered_df['Product'].str.contains(windows_version, case=False, na=False)]

        if relevant_df.empty:
            print(f"No relevant updates found for {windows_version}.")
            return None

        relevant_df = relevant_df.sort_values('Release date', ascending=False)

        return relevant_df
    except Exception as e:
        print("An error occurred while filtering the CSV data:", str(e))
        return None

def predict_vulnerabilities(model, future_dates):
    future_data = pd.DataFrame({
        'release_year': [date.year for date in future_dates],
        'release_month': [date.month for date in future_dates]
    })

    predictions = model.predict(future_data)
    prediction_df = future_data.copy()
    prediction_df['Predicted Base Score'] = predictions

    return prediction_df

def save_kb_articles_to_text(latest_kb_articles_df, output_text_path):
    try:
        if latest_kb_articles_df is not None and not latest_kb_articles_df.empty:
            kb_articles = latest_kb_articles_df['Article'].dropna().astype(str).tolist()

            kb_articles = list(set(kb_articles))

            kb_articles = [f"KB{kb}" for kb in kb_articles if kb.isdigit()]

            with open(output_text_path, 'w') as file:
                for kb in kb_articles:
                    file.write(f"{kb}\n")

            print(f"KB articles saved to {output_text_path}")
        else:
            print("No KB articles to save.")
    except Exception as e:
        print(f"An error occurred while saving KB articles to text: {str(e)}")

def perform_web_scraping_and_save_to_files(kb_articles):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    base_url = "https://answers.microsoft.com/en-us"

    for search_term in kb_articles:
        driver.get(base_url)
        time.sleep(3)
        file_name = f"{search_term}.txt"
        with open(file_name, "w") as file:
            try:
                search_input = driver.find_element(By.ID, "search-input-text")
            except Exception as e:
                print(f"Search input not found for {search_term}: {e}")
                continue

            search_input.clear()
            search_input.send_keys(search_term)

            try:
                search_button = driver.find_element(By.ID, "search-input-button")
                search_button.click()
            except Exception as e:
                print(f"Search button not found or click failed for {search_term}: {e}")
                continue

            time.sleep(5)

            def extract_titles():
                try:
                    results = driver.find_elements(By.CSS_SELECTOR, 'a.c-hyperlink[data-bi-id="thread-link"]')
                    for result in results:
                        try:
                            title = result.get_attribute('title')
                            file.write(f"Title: {title}\n")
                        except Exception as e:
                            print(f"Error extracting details from a result for {search_term}: {e}")
                except Exception as e:
                    print(f"Search results not found for {search_term}: {e}")

            extract_titles()

            for page in range(7):
                try:
                    next_button = driver.find_element(By.CSS_SELECTOR, 'a[data-bi-id="showNextPage"]')
                    next_button.click()
                    time.sleep(5)
                    extract_titles()
                except Exception as e:
                    print(f"No more pages or error navigating to the next page (page {page + 2}) for {search_term}: {e}")
                    break

    driver.quit()

def predict_sentiment(reviews):
    model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    reviews_tfidf = vectorizer.transform(reviews)
    predictions = model.predict(reviews_tfidf)
    return predictions

def perform_sentiment_analysis_for_each_kb(kb_articles):
    sentiment_results = {}
    for kb_article in kb_articles:
        file_name = f"{kb_article}.txt"
        try:
            with open(file_name, "r") as file:
                titles = file.readlines()
        except FileNotFoundError:
            print(f"File {file_name} not found.")
            continue

        titles = [title.split("Title: ")[-1].strip() for title in titles if "Title:" in title]
        
        if not titles:
            print(f"No titles found for {kb_article}, skipping sentiment analysis.")
            continue

        titles_series = pd.Series(titles)
        if titles_series.empty:
            print(f"No valid data to perform sentiment analysis for {kb_article}, skipping.")
            continue
        
        predictions = predict_sentiment(titles_series)
        sentiment_map = {1: 'positive', 0: 'negative'}
        predicted_sentiments = [sentiment_map[pred] for pred in predictions]

        total = len(predicted_sentiments)
        positive_count = predicted_sentiments.count('positive')
        negative_count = predicted_sentiments.count('negative')
        positive_percentage = (positive_count / total) * 100
        negative_percentage = (negative_count / total) * 100

        plt.figure(figsize=(6, 6))
        plt.pie([positive_percentage, negative_percentage], labels=['Positive', 'Negative'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        plt.title(f'Sentiment Analysis for {kb_article}')
        pie_chart_path = f"{kb_article}_sentiment_pie_chart.png"
        plt.savefig(pie_chart_path)
        plt.close()

        sentiment_results[kb_article] = {
            'positive': positive_percentage,
            'negative': negative_percentage,
            'pie_chart': pie_chart_path
        }

    return sentiment_results

def generate_html_report(df, matching_kb_numbers, most_recent_kb, kb_date, prediction_df, sentiment_analysis_results, output_path, latest_kb_articles_df, windows_version):
    html_report = f"""
    <html>
    <head>
        <title>KB Articles Report</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }}
            .container {{
                width: 90%;
                margin: 20px auto;
                background-color: #ffffff;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                border-radius: 8px;
            }}
            h1, h2 {{
                text-align: center;
                color: #333;
            }}
            h1 {{
                font-size: 22px;
                margin-bottom: 10px;
            }}
            h2 {{
                font-size: 24px;
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
            }}
            th {{
                background-color: #f9f9f9;
            }}
            .severity-header {{
                background-color: #f1f1f1;
                font-size: 19px;
                padding: 10px;
                cursor: pointer;
                margin-top: 10px;
                text-align: center;
                border: 1px solid #ddd;
            }}
            .scrollable-table {{
                max-height: 300px;
                overflow-y: auto;
                margin-bottom: 20px;
                display: none;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            img {{
                display: block;
                margin: 20px auto;
                max-width: 100%;
                height: auto;
            }}
        </style>
        <script>
            function toggleTable(id) {{
                var table = document.getElementById(id);
                if (table.style.display === "none") {{
                    table.style.display = "block";
                }} else {{
                    table.style.display = "none";
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Windows Version: {windows_version}</h1>
            <h1>Most Recent Matching KB Number: {most_recent_kb}</h1>
            <h2>Release Date of Most Recent KB Number: {kb_date}</h2>
    """

    if latest_kb_articles_df is not None:
        latest_kb_articles_df['Severity Level'] = pd.Categorical(
            latest_kb_articles_df['Max Severity'], 
            categories=["Critical", "Important", "Moderate", "Low"], 
            ordered=True
        )
        grouped = latest_kb_articles_df.groupby('Max Severity')

        html_report += f"<h2>Latest KB Articles Released After the Most Recent KB Update (Filtered for {windows_version})</h2>"

        for severity, group in grouped:
            if group.empty:
                continue

            table_id = f"latest_{severity.replace(' ', '_')}_table"
            html_report += f"<div class='severity-header' onclick='toggleTable(\"{table_id}\")'>{severity}</div>"
            html_report += f"<div class='scrollable-table' id='{table_id}'>"
            html_report += group.to_html(index=False, classes='highlight')
            html_report += "</div>"

    html_report += """
            <h2>Vulnerability Prediction Analysis</h2>
            <p>This section provides a predictive analysis of potential future vulnerabilities based on historical data. The predictions are made using a machine learning model trained on past vulnerability data.</p>
    """
    html_report += prediction_df.to_html(index=False, classes='highlight')

    html_report += "<h2>Sentiment Analysis for KB Articles</h2>"
    for kb_article, sentiments in sentiment_analysis_results.items():
        html_report += f"<h3>{kb_article}</h3>"
        html_report += f"<p>Positive Sentiment: {sentiments['positive']:.2f}%</p>"
        html_report += f"<p>Negative Sentiment: {sentiments['negative']:.2f}%</p>"
        html_report += f"<img src='{sentiments['pie_chart']}' alt='Sentiment Pie Chart for {kb_article}'>"

    with open(output_path, 'w') as file:
        file.write(html_report)

    plt.figure(figsize=(12, 8))
    prediction_df['release_month'] = pd.Categorical(prediction_df['release_month'], 
                                                    categories=[8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7], 
                                                    ordered=True)
    prediction_df = prediction_df.sort_values('release_month')
    sns.barplot(x='release_month', y='Predicted Base Score', data=prediction_df, palette="viridis")

    max_score = prediction_df['Predicted Base Score'].max()
    max_score_month = prediction_df.loc[prediction_df['Predicted Base Score'].idxmax(), 'release_month']
    plt.axhline(max_score, color='red', linestyle='--')
    plt.text(max_score_month - 1, max_score + 0.1, f'Highest Score: {max_score:.2f}', color='red')
    plt.title('Predicted Vulnerability Base Score Distribution by Month')
    plt.xlabel('Month')
    plt.ylabel('Predicted Base Score')
    plt.xticks(range(12), ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul' ])
    plt.tight_layout()
    plt.savefig('predicted_vulnerability_base_score_by_month.png')
    plt.close()
    
    with open(output_path, 'a') as file:
        file.write('<h2>Predicted Vulnerability Base Score Distribution by Month</h2>')
        file.write('<img src="predicted_vulnerability_base_score_by_month.png" alt="Predicted Vulnerability Base Score Distribution by Month">')
        file.write('</div></body></html>')

if __name__ == "__main__":
    model_path = 'vulnerability_model.pkl'  
    model = joblib.load(model_path)

    full_windows_version = get_windows_version()
    print(f"Full Windows Version: {full_windows_version}")

    windows_version = extract_clean_windows_version(full_windows_version)
    print(f"Cleaned Windows Version for Comparison: {windows_version}")

    if windows_version is None:
        print("Unsupported Windows version.")
        exit()

    kb_numbers = get_installed_updates()

    if kb_numbers:
        csv_path = 'combined_csv.csv' 
        df, matching_kb_numbers = compare_with_csv(kb_numbers, csv_path)

        if matching_kb_numbers is not None and len(matching_kb_numbers) > 0:
            most_recent_kb = max(matching_kb_numbers, key=lambda x: int(x))
            
            kb_date = df.loc[df['Article'] == most_recent_kb, 'Release date'].values[0]
            print("Most Recent Matching KB Number:", most_recent_kb)
            print("Release Date of Most Recent KB Number:", kb_date)
            
            future_dates = pd.date_range(start=pd.to_datetime('now'), periods=12, freq='ME')
            prediction_df = predict_vulnerabilities(model, future_dates)
            
            latest_kb_articles_df = list_latest_kb_articles(df, kb_date, windows_version)
            
            output_text_path = 'kb_articles.txt'
            save_kb_articles_to_text(latest_kb_articles_df, output_text_path)
            
            with open(output_text_path, 'r') as file:
                kb_articles = [line.strip() for line in file.readlines()]

            perform_web_scraping_and_save_to_files(kb_articles)

            sentiment_analysis_results = perform_sentiment_analysis_for_each_kb(kb_articles)

            output_path = 'kb_report.html'
            generate_html_report(df, matching_kb_numbers, most_recent_kb, kb_date, prediction_df, 
                                 sentiment_analysis_results, output_path, 
                                 latest_kb_articles_df, windows_version)
            print(f"HTML report generated and saved to {output_path}")
        else:
            print("No matching KB numbers found or an error occurred during comparison.")
    else:
        print("No updates found or an error occurred.")
