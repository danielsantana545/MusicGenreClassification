import pandas as pd

# Define column names if they are not present in the CSV file
# Since you have 7 columns, and the file paths are in the 6th column, you can define dummy names for the other columns
column_names = ['col1', 'col2', 'col3', 'col4', 'col5', 'file_path', 'col7']

# Load the CSV file without headers
df = pd.read_csv('D:\\MusicGenreClassification\\ismir04_genre\\metadata\\evaluation\\tracklist.csv', header=None, names=column_names)

# Check the first few rows of the dataframe to understand its structure
print(df.head())

# If 'file_path' column is not of object (string) type, convert it to string
if df['file_path'].dtype != 'object':
    df['file_path'] = df['file_path'].astype(str)

# Replace '.mp3' with '.wav' in the 'file_path' column
df['file_path'] = df['file_path'].str.replace('.mp3', '.wav', regex=False)

# Save the updated dataframe back to CSV
df.to_csv('D:\\MusicGenreClassification\\ismir04_genre\\metadata\\evaluation\\tracklistwav.csv', index=False, header=False)
