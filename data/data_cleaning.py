import pandas as pd

def load_data(file_path):
    """
    Load the sleep data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def clean_data(data):
    """
    Clean the sleep data by handling missing values and any necessary preprocessing steps.

    Args:
        data (pd.DataFrame): The raw sleep data.

    Returns:
        pd.DataFrame: The cleaned sleep data.
    """

    data_cleaned = data.dropna()


    data_cleaned = data_cleaned.drop_duplicates()

    return data_cleaned

def save_cleaned_data(data, output_path):
    """
    Save the cleaned sleep data to a specified CSV file.

    Args:
        data (pd.DataFrame): The cleaned data.
        output_path (str): The path to save the cleaned data.
    """
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    """
    Load, clean, and save the dataset for sleep analysis.
    """
    input_file = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\student_sleep_patterns.csv"
    output_file = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_cleaned.csv"

    data = load_data(input_file)
    cleaned_data = clean_data(data)
    save_cleaned_data(cleaned_data, output_file)
    print(f"Cleaned data saved to {output_file}")
