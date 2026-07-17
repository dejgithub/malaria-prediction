"""
Data Generator for Historical Malaria Cases
Generates 15 years of realistic malaria data with environmental factors
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

np.random.seed(42)


def generate_malaria_data():
    """
    Generate 15 years of monthly malaria case data with environmental factors
    Simulates real-world patterns: seasonal variation, yearly trends, environmental factors
    """
    years = 15
    months_per_year = 12
    total_months = years * months_per_year

    # Base year (2010)
    base_year = 2010

    # Generate time index
    dates = [
        datetime(base_year, 1, 1) + timedelta(days=30 * i) for i in range(total_months)
    ]
    years_idx = [(base_year + i // 12) for i in range(total_months)]
    months_idx = [(i % 12 + 1) for i in range(total_months)]

    # Seasonal pattern: higher cases in rainy season (May-October in tropical regions)
    seasonal_pattern = np.array(
        [0.3, 0.35, 0.5, 0.7, 0.9, 1.0, 0.95, 0.85, 0.75, 0.6, 0.4, 0.3]
    )

    # Base cases with yearly trend (assuming decreasing trend due to interventions)
    base_cases = 5000
    yearly_decrease_rate = 0.03  # 3% decrease per year

    # Generate malaria cases
    malaria_cases = []
    for i in range(total_months):
        year = years_idx[i]
        month = months_idx[i]

        # Seasonal factor
        seasonal_factor = seasonal_pattern[month - 1]

        # Yearly trend (decreasing)
        year_factor = (1 - yearly_decrease_rate) ** (year - base_year)

        # Random variation
        random_factor = np.random.uniform(0.85, 1.15)

        # Long-term cycles (El Nino effects)
        cycle_factor = 1 + 0.15 * np.sin(2 * np.pi * (year - base_year) / 7)

        # Calculate cases
        cases = (
            base_cases * seasonal_factor * year_factor * cycle_factor * random_factor
        )
        malaria_cases.append(int(cases))

    # Environmental factors (simulated)
    rainfall = []
    temperature = []
    humidity = []

    for i in range(total_months):
        month = months_idx[i]

        # Rainfall (mm) - peaks in monsoon
        base_rainfall = [50, 60, 80, 120, 180, 200, 180, 160, 140, 100, 70, 55]
        rain = base_rainfall[month - 1] + np.random.uniform(-30, 30)
        rainfall.append(max(0, rain))

        # Temperature (Celsius) - tropical region
        base_temp = [26, 27, 28, 29, 30, 29, 28, 28, 29, 28, 27, 26]
        temp = base_temp[month - 1] + np.random.uniform(-1, 1)
        temperature.append(temp)

        # Humidity (%)
        base_humidity = [70, 72, 75, 80, 85, 88, 86, 84, 82, 78, 74, 71]
        humid = base_humidity[month - 1] + np.random.uniform(-5, 5)
        humidity.append(min(100, max(50, humid)))

    # Create DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "year": years_idx,
            "month": months_idx,
            "malaria_cases": malaria_cases,
            "rainfall_mm": [round(x, 1) for x in rainfall],
            "temperature_celsius": [round(x, 1) for x in temperature],
            "humidity_percent": [round(x, 1) for x in humidity],
        }
    )

    return df


def save_data(df, filepath):
    """Save data to CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    return filepath


if __name__ == "__main__":
    # Generate data
    print("Generating 15 years of historical malaria data...")
    df = generate_malaria_data()

    # Save data
    data_path = save_data(df, "data/malaria_data.csv")

    # Display summary
    print(f"\nData Summary:")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total cases: {df['malaria_cases'].sum():,}")
    print(f"Average monthly cases: {df['malaria_cases'].mean():.0f}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\nLast 5 rows:")
    print(df.tail().to_string())
