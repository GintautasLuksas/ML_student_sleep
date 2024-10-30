import pandas as pd

def calculate_sleep_hours(start, end):
    """
    Calculate sleep hours from start and end times.

    Args:
        start (int): The sleep start time in hours (0-23).
        end (int): The sleep end time in hours (0-23).

    Returns:
        float: Calculated sleep hours, capped at a maximum of 8 hours for regular sleep or 16 for overnight sleep.
    """
    if end < start:
        hours = (end + 24) - start
        return min(hours, 16)
    else:
        hours = end - start
        return min(hours, 8)

def normalize_data(data):
    """
    Normalize the sleep data by scaling numerical features, encoding categorical features,
    and calculating sleep hours on weekdays and weekends in hours.

    Args:
        data (pd.DataFrame): Input data to be normalized.

    Returns:
        pd.DataFrame: Normalized data with the following transformations:
            - Student_ID column dropped.
            - Age normalized to a range of 0 to 1.
            - University_Year normalized to specific numeric values.
            - Gender one-hot encoded.
            - Sleep_Duration, Study_Hours, Screen_Time normalized to a range of 0 to 1.
            - Caffeine_Intake and Physical_Activity normalized to specific ranges.
            - Sleep_Quality converted to a binary target.
            - Sleep hours calculated for weekdays and weekends.
            - Sleep start binary features created.
            - Original sleep start and end columns dropped.
    """
    data = data.drop(columns=['Student_ID'])
    data['Age'] = ((data['Age'] - 18) / (25 - 18)).round(3)
    year_mapping = {'1st Year': 0, '2nd Year': 0.33, '3rd Year': 0.67, '4th Year': 1}
    data['University_Year'] = data['University_Year'].map(year_mapping).round(3)
    data = pd.get_dummies(data, columns=['Gender'], drop_first=True)
    data['Sleep_Duration'] = (data['Sleep_Duration'] / 24).round(3)
    data['Study_Hours'] = (data['Study_Hours'] / 24).round(3)
    data['Screen_Time'] = (data['Screen_Time'] / 24).round(3)
    data['Caffeine_Intake'] = (data['Caffeine_Intake'] / 10).round(3)
    data['Physical_Activity'] = (data['Physical_Activity'] / 120).round(3)
    data['Sleep_Quality'] = data['Sleep_Quality'].apply(lambda x: 1 if x > 5 else 0)
    data['Sleep_Hours_Weekdays'] = data.apply(
        lambda row: calculate_sleep_hours(row['Weekday_Sleep_Start'], row['Weekday_Sleep_End']),
        axis=1
    ).round(3)
    data['Sleep_Hours_Weekends'] = data.apply(
        lambda row: calculate_sleep_hours(row['Weekend_Sleep_Start'], row['Weekend_Sleep_End']),
        axis=1
    ).round(3)
    data['Sleep_Start_Binary_Weekdays'] = data['Weekday_Sleep_Start'].apply(lambda x: 1 if 19 <= x <= 23 else 0)
    data['Sleep_Start_Binary_Weekends'] = data['Weekend_Sleep_Start'].apply(lambda x: 1 if 19 <= x <= 23 else 0)
    data = data.drop(columns=['Weekday_Sleep_Start', 'Weekday_Sleep_End', 'Weekend_Sleep_Start', 'Weekend_Sleep_End'])

    return data
