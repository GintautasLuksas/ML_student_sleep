import pandas as pd

def load_data(file_path):
    """Load the sleep data from a CSV file."""
    return pd.read_csv(file_path)

def normalize_data(data):
    """Normalize the sleep data."""
    # Drop Student_ID
    data = data.drop(columns=['Student_ID'])

    # Normalize Age (18 to 25)
    data['Age'] = ((data['Age'] - 18) / (25 - 18)).round(3)

    # Normalize University_Year (1st Year: 0, 2nd Year: 0.33, 3rd Year: 0.67, 4th Year: 1)
    year_mapping = {'1st Year': 0, '2nd Year': 0.33, '3rd Year': 0.67, '4th Year': 1}
    data['University_Year'] = data['University_Year'].map(year_mapping).round(3)

    # One-hot encoding for Gender
    data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

    # Normalize Sleep_Duration, Study_Hours, Screen_Time (using max values)
    data['Sleep_Duration'] = (data['Sleep_Duration'] / 24).round(3)  # Assuming max is 24 hours
    data['Study_Hours'] = (data['Study_Hours'] / 24).round(3)  # Assuming max is 24 hours
    data['Screen_Time'] = (data['Screen_Time'] / 24).round(3)  # Assuming max is 24 hours

    # Normalize Caffeine_Intake and Physical_Activity (assuming a reasonable max)
    data['Caffeine_Intake'] = (data['Caffeine_Intake'] / 10).round(3)  # Assuming max is 10 cups
    data['Physical_Activity'] = (data['Physical_Activity'] / 120).round(3)  # Assuming max is 120 minutes

    # Create Sleep_Quality binary target
    data['Sleep_Quality'] = data['Sleep_Quality'].apply(lambda x: 1 if x > 5 else 0)

    # Calculate Sleep Hours on Weekdays
    data['Weekday_Sleep_Start'] = data['Weekday_Sleep_Start'].apply(lambda x: int(x))
    data['Weekday_Sleep_End'] = data['Weekday_Sleep_End'].apply(lambda x: int(x))
    data['Sleep_Hours_Weekdays'] = (data['Weekday_Sleep_End'] - data['Weekday_Sleep_Start']).clip(lower=0) / 24
    data['Sleep_Hours_Weekdays'] = data['Sleep_Hours_Weekdays'].round(3)  # Normalize and round

    # Calculate Sleep Hours on Weekends
    data['Weekend_Sleep_Start'] = data['Weekend_Sleep_Start'].apply(lambda x: int(x))
    data['Weekend_Sleep_End'] = data['Weekend_Sleep_End'].apply(lambda x: int(x))
    data['Sleep_Hours_Weekends'] = (data['Weekend_Sleep_End'] - data['Weekend_Sleep_Start']).clip(lower=0) / 24
    data['Sleep_Hours_Weekends'] = data['Sleep_Hours_Weekends'].round(3)  # Normalize and round

    # Normalize Sleep Start time to binary
    data['Sleep_Start_Binary_Weekdays'] = data['Weekday_Sleep_Start'].apply(lambda x: 1 if 19 <= x <= 23 else 0)
    data['Sleep_Start_Binary_Weekends'] = data['Weekend_Sleep_Start'].apply(lambda x: 1 if 19 <= x <= 23 else 0)

    # Drop original sleep start and end columns
    data = data.drop(columns=['Weekday_Sleep_Start', 'Weekday_Sleep_End', 'Weekend_Sleep_Start', 'Weekend_Sleep_End'])

    return data


def save_normalized_data(data, output_path):
    """Save the normalized sleep data to a specified CSV file."""
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    """Load, normalize, and save the dataset for sleep analysis."""
    input_file = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_cleaned.csv"
    output_file = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_processed.csv"

    data = load_data(input_file)
    normalized_data = normalize_data(data)
    save_normalized_data(normalized_data, output_file)
    print(f"Normalized data saved to {output_file}")
