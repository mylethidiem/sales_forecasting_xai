import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(2025)


def generate_store_data():
    """Generate store data"""

    # Define provinces and stores
    provinces = ["Hanoi", "Ho Chi Minh City"]

    stores = [
        # Hanoi stores
        {"id": 1, "name": "Hoan Kiem Market", "province": "Hanoi"},
        {"id": 2, "name": "Ba Dinh Supermarket", "province": "Hanoi"},
        {"id": 3, "name": "Dong Da Mall", "province": "Hanoi"},
        {"id": 4, "name": "Tay Ho Store", "province": "Hanoi"},
        {"id": 5, "name": "Long Bien Shop", "province": "Hanoi"},
        # Ho Chi Minh City stores
        {"id": 6, "name": "District 1 Market", "province": "Ho Chi Minh City"},
        {"id": 7, "name": "Ben Thanh Store", "province": "Ho Chi Minh City"},
        {"id": 8, "name": "Saigon Supermarket", "province": "Ho Chi Minh City"},
        {"id": 9, "name": "Phu Nhuan Shop", "province": "Ho Chi Minh City"},
        {"id": 10, "name": "Binh Thanh Market", "province": "Ho Chi Minh City"},
    ]

    return provinces, stores


def generate_item_data():
    """Generate item data"""

    # Define categories and items
    categories = [
        "Staples",
        "Dairy & Frozen",
        "Beverages & Snacks",
        "Household & Personal Care",
        "Baby & Health",
    ]

    items = [
        # Staples
        {
            "id": 1,
            "name": "Rice",
            "category": "Staples",
            "base_price": 20.0,
            "base_sales": 15,
            "volatility": 0.3,
        },
        {
            "id": 2,
            "name": "Noodles",
            "category": "Staples",
            "base_price": 15.0,
            "base_sales": 12,
            "volatility": 0.25,
        },
        {
            "id": 3,
            "name": "Bread",
            "category": "Staples",
            "base_price": 10.0,
            "base_sales": 20,
            "volatility": 0.4,
        },
        {
            "id": 4,
            "name": "Flour",
            "category": "Staples",
            "base_price": 12.0,
            "base_sales": 8,
            "volatility": 0.2,
        },
        {
            "id": 5,
            "name": "Cooking Oil",
            "category": "Staples",
            "base_price": 25.0,
            "base_sales": 10,
            "volatility": 0.15,
        },
        {
            "id": 6,
            "name": "Sugar",
            "category": "Staples",
            "base_price": 8.0,
            "base_sales": 7,
            "volatility": 0.1,
        },
        # Dairy & Frozen
        {
            "id": 7,
            "name": "Milk",
            "category": "Dairy & Frozen",
            "base_price": 18.0,
            "base_sales": 30,
            "volatility": 0.35,
        },
        {
            "id": 8,
            "name": "Cheese",
            "category": "Dairy & Frozen",
            "base_price": 35.0,
            "base_sales": 12,
            "volatility": 0.3,
        },
        {
            "id": 9,
            "name": "Yogurt",
            "category": "Dairy & Frozen",
            "base_price": 12.0,
            "base_sales": 25,
            "volatility": 0.4,
        },
        {
            "id": 10,
            "name": "Ice Cream",
            "category": "Dairy & Frozen",
            "base_price": 30.0,
            "base_sales": 15,
            "volatility": 0.5,
        },
        {
            "id": 11,
            "name": "Frozen Vegetables",
            "category": "Dairy & Frozen",
            "base_price": 22.0,
            "base_sales": 10,
            "volatility": 0.25,
        },
        # Beverages & Snacks
        {
            "id": 12,
            "name": "Soda",
            "category": "Beverages & Snacks",
            "base_price": 15.0,
            "base_sales": 40,
            "volatility": 0.45,
        },
        {
            "id": 13,
            "name": "Juice",
            "category": "Beverages & Snacks",
            "base_price": 20.0,
            "base_sales": 30,
            "volatility": 0.4,
        },
        {
            "id": 14,
            "name": "Water",
            "category": "Beverages & Snacks",
            "base_price": 10.0,
            "base_sales": 50,
            "volatility": 0.3,
        },
        {
            "id": 15,
            "name": "Coffee",
            "category": "Beverages & Snacks",
            "base_price": 45.0,
            "base_sales": 20,
            "volatility": 0.25,
        },
        {
            "id": 16,
            "name": "Tea",
            "category": "Beverages & Snacks",
            "base_price": 35.0,
            "base_sales": 15,
            "volatility": 0.2,
        },
        {
            "id": 17,
            "name": "Chips",
            "category": "Beverages & Snacks",
            "base_price": 12.0,
            "base_sales": 35,
            "volatility": 0.45,
        },
        {
            "id": 18,
            "name": "Cookies",
            "category": "Beverages & Snacks",
            "base_price": 18.0,
            "base_sales": 30,
            "volatility": 0.4,
        },
        {
            "id": 19,
            "name": "Chocolate",
            "category": "Beverages & Snacks",
            "base_price": 22.0,
            "base_sales": 25,
            "volatility": 0.35,
        },
        # Household & Personal Care
        {
            "id": 20,
            "name": "Soap",
            "category": "Household & Personal Care",
            "base_price": 8.0,
            "base_sales": 20,
            "volatility": 0.2,
        },
        {
            "id": 21,
            "name": "Shampoo",
            "category": "Household & Personal Care",
            "base_price": 25.0,
            "base_sales": 15,
            "volatility": 0.25,
        },
        {
            "id": 22,
            "name": "Toothpaste",
            "category": "Household & Personal Care",
            "base_price": 15.0,
            "base_sales": 18,
            "volatility": 0.15,
        },
        {
            "id": 23,
            "name": "Laundry Detergent",
            "category": "Household & Personal Care",
            "base_price": 40.0,
            "base_sales": 12,
            "volatility": 0.2,
        },
        {
            "id": 24,
            "name": "Paper Towels",
            "category": "Household & Personal Care",
            "base_price": 20.0,
            "base_sales": 14,
            "volatility": 0.3,
        },
        {
            "id": 25,
            "name": "Toilet Paper",
            "category": "Household & Personal Care",
            "base_price": 25.0,
            "base_sales": 16,
            "volatility": 0.25,
        },
        {
            "id": 26,
            "name": "Trash Bags",
            "category": "Household & Personal Care",
            "base_price": 18.0,
            "base_sales": 10,
            "volatility": 0.15,
        },
        {
            "id": 27,
            "name": "Dishwashing Liquid",
            "category": "Household & Personal Care",
            "base_price": 15.0,
            "base_sales": 11,
            "volatility": 0.2,
        },
        {
            "id": 28,
            "name": "All-Purpose Cleaner",
            "category": "Household & Personal Care",
            "base_price": 22.0,
            "base_sales": 9,
            "volatility": 0.15,
        },
        # Baby & Health
        {
            "id": 29,
            "name": "Diapers",
            "category": "Baby & Health",
            "base_price": 45.0,
            "base_sales": 25,
            "volatility": 0.3,
        },
        {
            "id": 30,
            "name": "Baby Food",
            "category": "Baby & Health",
            "base_price": 20.0,
            "base_sales": 15,
            "volatility": 0.25,
        },
        {
            "id": 31,
            "name": "Baby Wipes",
            "category": "Baby & Health",
            "base_price": 15.0,
            "base_sales": 20,
            "volatility": 0.2,
        },
        {
            "id": 32,
            "name": "Pain Relievers",
            "category": "Baby & Health",
            "base_price": 30.0,
            "base_sales": 10,
            "volatility": 0.15,
        },
        {
            "id": 33,
            "name": "Vitamins",
            "category": "Baby & Health",
            "base_price": 40.0,
            "base_sales": 8,
            "volatility": 0.2,
        },
        {
            "id": 34,
            "name": "Cold & Flu Medicine",
            "category": "Baby & Health",
            "base_price": 35.0,
            "base_sales": 7,
            "volatility": 0.4,
        },
        {
            "id": 35,
            "name": "First Aid Kit",
            "category": "Baby & Health",
            "base_price": 50.0,
            "base_sales": 5,
            "volatility": 0.1,
        },
    ]

    return categories, items


def calculate_daily_sales(date, store, item, weather_data=None):
    """
    Calculate daily sales based on various factors.
    Returns an integer value for sales quantity.
    """
    # Base sales for this item
    base_sales = item["base_sales"]

    # Store factor (some stores have higher sales)
    store_factor = 0.8 + (store["id"] % 10) / 10  # 0.8 to 1.7

    # Day of week factor (weekend boost)
    day_of_week = date.weekday()  # 0 = Monday, 6 = Sunday
    weekday_factor = 1.0
    if day_of_week >= 5:  # Weekend
        weekday_factor = 1.3

    # Monthly seasonality
    month = date.month
    # Higher sales in December (holidays), lower in February
    month_factor = 1.0 + 0.3 * (month == 12) - 0.1 * (month == 2)

    # Quarterly business cycle
    quarter = (month - 1) // 3 + 1
    quarter_factor = 1.0 + 0.05 * (quarter - 2.5)  # Q3-Q4 slightly higher

    # Holiday effects
    holiday_factor = 1.0
    # Vietnamese New Year (Tet) - usually in late January or early February
    if (month == 1 and date.day >= 27) or (month == 2 and date.day <= 5):
        holiday_factor = 1.5
    # National Day (September 2)
    elif month == 9 and date.day == 2:
        holiday_factor = 1.3
    # Year-end shopping
    elif month == 12 and date.day >= 20:
        holiday_factor = 1.4

    # Weather effects if weather data is provided
    weather_factor = 1.0
    if weather_data is not None:
        # Find weather for this date and province
        date_str = date.strftime("%Y-%m-%d")
        province = store["province"]
        day_weather = weather_data.get((date_str, province))

        if day_weather:
            temp = day_weather["temperature"]
            humidity = day_weather["humidity"]

            # Temperature effects differ by item category
            if item["category"] == "Beverages & Snacks":
                # More beverages sold in hot weather
                if temp > 28:
                    weather_factor *= 1.3
                elif temp < 18:
                    weather_factor *= 0.9
            elif item["category"] == "Dairy & Frozen":
                # More ice cream in hot weather
                if temp > 28:
                    weather_factor *= 1.4
                elif temp < 18:
                    weather_factor *= 0.8

            # Rain effect (approximated by high humidity)
            if humidity > 80:
                # People buy more when staying indoors
                if item["category"] in [
                    "Beverages & Snacks",
                    "Household & Personal Care",
                ]:
                    weather_factor *= 1.2

    # Year-over-year growth (for 2017 data)
    yoy_growth = 1.0
    if date.year == 2017:
        # 5% general growth with some category variations
        category_growth = {
            "Staples": 1.03,
            "Dairy & Frozen": 1.05,
            "Beverages & Snacks": 1.08,
            "Household & Personal Care": 1.05,
            "Baby & Health": 1.07,
        }
        yoy_growth = category_growth.get(item["category"], 1.05)

    # Random variation
    random_factor = np.random.normal(1.0, item["volatility"])

    # Calculate final sales
    sales = (
        base_sales
        * store_factor
        * weekday_factor
        * month_factor
        * quarter_factor
        * holiday_factor
        * weather_factor
        * yoy_growth
        * random_factor
    )

    # Ensure minimum sales and convert to integer
    sales = max(
        1, int(round(sales))
    )  # Minimum sales of 1 unit, rounded to nearest integer

    return sales


def generate_weather_data(start_date, end_date, provinces):
    """Generate synthetic weather data"""

    # Define base temperatures and humidity for each province
    province_weather = {
        "Hanoi": {
            "base_temp": {
                1: 16,
                2: 17,
                3: 20,
                4: 24,
                5: 28,
                6: 30,
                7: 30,
                8: 29,
                9: 28,
                10: 25,
                11: 21,
                12: 18,
            },
            "temp_variation": 3.5,
            "base_humidity": {
                1: 80,
                2: 83,
                3: 85,
                4: 85,
                5: 80,
                6: 80,
                7: 83,
                8: 85,
                9: 83,
                10: 78,
                11: 75,
                12: 77,
            },
            "humidity_variation": 10,
            "seasons": {
                1: "winter",
                2: "winter",
                3: "spring",
                4: "spring",
                5: "summer",
                6: "summer",
                7: "summer",
                8: "summer",
                9: "fall",
                10: "fall",
                11: "fall",
                12: "winter",
            },
        },
        "Ho Chi Minh City": {
            "base_temp": {
                1: 26,
                2: 27,
                3: 28,
                4: 29,
                5: 29,
                6: 28,
                7: 28,
                8: 28,
                9: 28,
                10: 27,
                11: 27,
                12: 26,
            },
            "temp_variation": 2.0,
            "base_humidity": {
                1: 70,
                2: 70,
                3: 70,
                4: 75,
                5: 80,
                6: 83,
                7: 85,
                8: 85,
                9: 88,
                10: 85,
                11: 80,
                12: 75,
            },
            "humidity_variation": 8,
            "seasons": {
                1: "dry",
                2: "dry",
                3: "dry",
                4: "dry",
                5: "wet",
                6: "wet",
                7: "wet",
                8: "wet",
                9: "wet",
                10: "wet",
                11: "wet",
                12: "dry",
            },
        },
    }

    # Create date range
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)

    # Generate weather data
    weather_data = []
    weather_dict = {}  # For lookup during sales calculation

    for date in date_list:
        month = date.month
        for province in provinces:
            # Get base values for this province and month
            base_temp = province_weather[province]["base_temp"][month]
            temp_variation = province_weather[province]["temp_variation"]
            base_humidity = province_weather[province]["base_humidity"][month]
            humidity_variation = province_weather[province]["humidity_variation"]
            season = province_weather[province]["seasons"][month]

            # Add random variation
            temperature = base_temp + np.random.uniform(-temp_variation, temp_variation)
            humidity = base_humidity + np.random.uniform(
                -humidity_variation, humidity_variation
            )

            # Round to one decimal place
            temperature = round(temperature, 1)
            humidity = round(humidity, 1)

            # Ensure humidity is within realistic range
            humidity = max(40, min(95, humidity))

            # Add to weather data
            weather_data.append(
                {
                    "city": province,
                    "date": date.strftime("%Y-%m-%d"),
                    "temperature": temperature,
                    "humidity": humidity,
                    "season": season,
                }
            )

            # Add to lookup dictionary
            weather_dict[(date.strftime("%Y-%m-%d"), province)] = {
                "temperature": temperature,
                "humidity": humidity,
                "season": season,
            }

    return pd.DataFrame(weather_data), weather_dict


def generate_sales_data(start_date, end_date, stores, items, weather_dict):
    """Generate synthetic sales data"""

    # Create date range
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)

    # Generate sales data
    sales_data = []

    # For each date, store, and item, calculate sales
    for date in date_list:
        for store in stores:
            # Not all stores carry all items
            # Use store_id to deterministically select items
            store_seed = store["id"] * 10
            np.random.seed(store_seed)

            # Select a subset of items for this store
            store_items = []
            for item in items:
                # 80% chance of carrying an item
                if np.random.random() < 0.8:
                    store_items.append(item)

            # Reset random seed
            np.random.seed(None)

            # Calculate sales for each item
            for item in store_items:
                # Calculate sales for this combination
                sales_value = calculate_daily_sales(date, store, item, weather_dict)

                # Add to sales data
                sales_data.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "province": store["province"],
                        "store_id": store["id"],
                        "store_name": store["name"],
                        "category": item["category"],
                        "item_id": item["id"],
                        "item_name": item["name"],
                        "sales": sales_value,
                    }
                )

    return pd.DataFrame(sales_data)


def add_outliers_and_nans(data, outlier_percentage=0.01, nan_percentage=0.1):
    """Add the nan values to data set"""
    # Copy the original data to avoid modifying the input directly
    modified_data = data.copy()

    # Calculate the number of rows to add outliers and NaN values
    num_rows = len(modified_data)
    num_outliers = int(num_rows * outlier_percentage / 100)
    num_nans = int(num_rows * nan_percentage / 100)

    # Add outliers to the 'sales' column
    np.random.seed(2025)
    outlier_indices = np.random.choice(num_rows, num_outliers, replace=False)
    modified_data.loc[
        outlier_indices, "sales"
    ] *= 3  # Increase sales by a factor to create outliers

    # Add NaN values to the 'sales' column
    nan_indices = np.random.choice(num_rows, num_nans, replace=False)
    modified_data.loc[nan_indices, "sales"] = np.nan

    return modified_data


def check_missing_values(df):
    """Check missing values"""
    df_nan = pd.DataFrame(
        {
            "counts": df.isna().sum(),
            "ratio (%)": np.round(df.isna().sum() / df.shape[0], 4) * 100,
        }
    )
    return df_nan


def main():
    """Main function to generate all data"""
    print("Generating synthetic data for Sales Forecasting with XAI project...")

    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Generate store and item data
    provinces, stores = generate_store_data()
    categories, items = generate_item_data()

    print(
        f"Created {len(stores)} stores and {len(items)} items across {len(categories)} categories"
    )

    # Define date ranges
    start_date_2016 = datetime(2016, 1, 1)
    end_date_2016 = datetime(2016, 12, 31)

    start_date_2017 = datetime(2017, 1, 1)
    end_date_2017 = datetime(2017, 12, 31)

    # Generate weather data for both years
    print("Generating weather data...")
    weather_df, weather_dict = generate_weather_data(
        start_date_2016, end_date_2017, provinces
    )

    # Save weather data
    weather_df.to_csv("data/weather_data.csv", index=False)
    print(f"Saved weather data with {len(weather_df)} records")

    # Generate 2016 sales data
    print("Generating 2016 sales data...")
    sales_2016 = generate_sales_data(
        start_date_2016, end_date_2016, stores, items, weather_dict
    )

    sales_2016 = add_outliers_and_nans(
        sales_2016, outlier_percentage=0.5, nan_percentage=1
    )

    # Save 2016 sales data
    sales_2016.to_csv("data/2016_sales.csv", index=False)
    print(f"Saved 2016 sales data with {len(sales_2016)} records")

    # Generate 2017 sales data
    print("Generating 2017 sales data...")
    sales_2017 = generate_sales_data(
        start_date_2017, end_date_2017, stores, items, weather_dict
    )

    sales_2017 = add_outliers_and_nans(
        sales_2017, outlier_percentage=0.5, nan_percentage=1
    )

    # Save 2017 sales data
    sales_2017.to_csv("data/2017_sales.csv", index=False)
    print(f"Saved 2017 sales data with {len(sales_2017)} records")

    # Print statistics
    print("\nData Generation Complete!")
    print(f"Total weather records: {len(weather_df)}")
    print(f"Total 2016 sales records: {len(sales_2016)}")
    print(f"Total 2017 sales records: {len(sales_2017)}")
    print(
        f"Total combined records: {len(weather_df) + len(sales_2016) + len(sales_2017)}"
    )

    print("\nSales Statistics:")
    print(f"2016 Average Sales: {sales_2016['sales'].mean():.2f} units")
    print(f"2016 Max Sales: {sales_2016['sales'].max()} units")
    print(f"2017 Average Sales: {sales_2017['sales'].mean():.2f} units")
    print(f"2017 Max Sales: {sales_2017['sales'].max()} units")
    print(f"Missing values: {check_missing_values(sales_2016)}")
    print(f"Missing values: {check_missing_values(sales_2017)}")

    print("\nFiles saved to data/ directory:")
    print("- data/weather_data.csv")
    print("- data/2016_sales.csv")
    print("- data/2017_sales.csv")


if __name__ == "__main__":
    main()
