import pandas as pd

csv_files = ['2016.csv', '2017.csv', '2018.csv','2019.csv','2020.csv','2021.csv','2022.csv','2023.csv','2024.csv'] 
dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)
combined_csv = pd.concat(dataframes, ignore_index=True)
combined_csv.to_csv('combined_csv.csv', index=False)
