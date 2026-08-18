"""Runs the complete climate data analysis."""

import os

from data_processing import calculate_slope
from data_processing import get_country_co2
from data_processing import get_world_co2
from data_processing import load_co2_data
from data_processing import load_nasa_data
from data_processing import load_weather_data
from data_processing import merge_country_co2_weather
from data_processing import merge_temperature_co2
from data_processing import summarize_weather_by_month
from data_processing import test_temperature_period_difference
from visualizations import plot_annual_temperature
from visualizations import plot_country_co2
from visualizations import plot_country_co2_weather
from visualizations import plot_monthly_weather
from visualizations import plot_seasonal_temperature
from visualizations import plot_temperature_co2
from visualizations import plot_temperature_precipitation
from visualizations import plot_world_co2


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, "data")
FIGURE_DIRECTORY = os.path.join(BASE_DIRECTORY, "figures")
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "outputs")

NASA_FILE = os.path.join(DATA_DIRECTORY, "NASA Global Means.csv")
WEATHER_FILE = os.path.join(
    DATA_DIRECTORY,
    "GlobalWeatherRepository.csv",
)
CO2_FILE = os.path.join(DATA_DIRECTORY, "OWID_CB_WIDEF.csv")

COUNTRIES = [
    "United States",
    "China",
    "India",
    "Russian Federation",
    "Japan",
]


def check_input_files():
    """Raises an error when one or more expected CSV files are missing."""
    expected_files = [NASA_FILE, WEATHER_FILE, CO2_FILE]
    missing = [
        file_path for file_path in expected_files
        if not os.path.exists(file_path)
    ]
    if missing:
        formatted = "\n".join(missing)
        raise FileNotFoundError(
            "Place the required CSV files in the data folder:\n"
            f"{formatted}"
        )


def create_output_directories():
    """Creates folders used for generated results."""
    os.makedirs(FIGURE_DIRECTORY, exist_ok=True)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def write_text_summary(
    nasa_data,
    weather_data,
    co2_data,
    merged_data,
    country_weather,
    output_file,
):
    """Writes major analysis facts and computed results to a text file."""
    early = nasa_data[
        nasa_data["Year"].between(1950, 1999)
    ]
    recent = nasa_data[
        nasa_data["Year"].between(2000, 2024)
    ]

    early_slope = calculate_slope(early, "Year", "J-D")
    recent_slope = calculate_slope(recent, "Year", "J-D")
    correlation = merged_data[
        ["J-D", "CO2_Emissions"]
    ].corr().iloc[0, 1]
    early_mean, recent_mean, statistic, p_value = (
        test_temperature_period_difference(nasa_data)
    )

    with open(output_file, "w", encoding="utf-8") as summary:
        summary.write("Climate Project Summary\n")
        summary.write("=======================\n\n")
        summary.write(f"NASA cleaned shape: {nasa_data.shape}\n")
        summary.write(
            f"Weather cleaned shape: {weather_data.shape}\n"
        )
        summary.write(f"CO2 long-format shape: {co2_data.shape}\n")
        summary.write(
            "NASA year range: "
            f"{nasa_data['Year'].min()}-"
            f"{nasa_data['Year'].max()}\n"
        )
        summary.write(
            "Weather observation range: "
            f"{weather_data['last_updated'].min()} to "
            f"{weather_data['last_updated'].max()}\n"
        )
        summary.write(
            "Temperature slope, 1950-1999: "
            f"{early_slope:.4f} degrees C per year\n"
        )
        summary.write(
            "Temperature slope, 2000-2024: "
            f"{recent_slope:.4f} degrees C per year\n"
        )
        summary.write(
            "Temperature-CO2 correlation, overlapping years: "
            f"{correlation:.4f}\n"
        )
        summary.write(
            "Mean temperature anomaly, 1950-1999: "
            f"{early_mean:.4f} degrees C\n"
        )
        summary.write(
            "Mean temperature anomaly, 2000-2024: "
            f"{recent_mean:.4f} degrees C\n"
        )
        summary.write(
            "One-sided Welch t-test statistic: "
            f"{statistic:.4f}\n"
        )
        summary.write(
            "One-sided Welch t-test p-value: "
            f"{p_value:.6g}\n"
        )
        summary.write(
            "Countries in CO2-weather merge: "
            f"{len(country_weather)}\n"
        )


def main():
    """Loads data, computes summaries, and saves all project figures."""
    check_input_files()
    create_output_directories()

    nasa_data = load_nasa_data(NASA_FILE)
    weather_data = load_weather_data(WEATHER_FILE)
    co2_data = load_co2_data(CO2_FILE)

    world_co2 = get_world_co2(co2_data)
    country_co2 = get_country_co2(co2_data, COUNTRIES)
    monthly_weather = summarize_weather_by_month(weather_data)
    merged_data = merge_temperature_co2(nasa_data, world_co2)
    country_weather = merge_country_co2_weather(
        co2_data,
        weather_data,
        COUNTRIES,
    )

    nasa_data.describe().to_csv(
        os.path.join(OUTPUT_DIRECTORY, "nasa_summary.csv")
    )
    weather_data.describe().to_csv(
        os.path.join(OUTPUT_DIRECTORY, "weather_summary.csv")
    )
    world_co2.describe().to_csv(
        os.path.join(OUTPUT_DIRECTORY, "world_co2_summary.csv")
    )
    monthly_weather.to_csv(
        os.path.join(OUTPUT_DIRECTORY, "monthly_weather.csv"),
        index=False,
    )
    merged_data.to_csv(
        os.path.join(
            OUTPUT_DIRECTORY,
            "temperature_co2_merged.csv",
        ),
        index=False,
    )
    country_weather.to_csv(
        os.path.join(
            OUTPUT_DIRECTORY,
            "country_co2_weather.csv",
        ),
        index=False,
    )

    write_text_summary(
        nasa_data,
        weather_data,
        co2_data,
        merged_data,
        country_weather,
        os.path.join(OUTPUT_DIRECTORY, "eda_summary.txt"),
    )

    plot_annual_temperature(
        nasa_data,
        os.path.join(
            FIGURE_DIRECTORY,
            "nasa_annual_temperature.png",
        ),
    )
    plot_seasonal_temperature(
        nasa_data,
        os.path.join(
            FIGURE_DIRECTORY,
            "nasa_seasonal_temperature.png",
        ),
    )
    plot_world_co2(
        world_co2,
        os.path.join(FIGURE_DIRECTORY, "co2_world.png"),
    )
    plot_country_co2(
        country_co2,
        os.path.join(FIGURE_DIRECTORY, "co2_countries.png"),
    )
    plot_monthly_weather(
        monthly_weather,
        os.path.join(
            FIGURE_DIRECTORY,
            "weather_monthly_temperature.png",
        ),
    )
    plot_temperature_precipitation(
        weather_data,
        os.path.join(
            FIGURE_DIRECTORY,
            "weather_temperature_precipitation.png",
        ),
    )
    plot_temperature_co2(
        merged_data,
        os.path.join(
            FIGURE_DIRECTORY,
            "combined_temperature_co2.png",
        ),
    )
    plot_country_co2_weather(
        country_weather,
        os.path.join(
            FIGURE_DIRECTORY,
            "country_co2_weather.png",
        ),
    )

    print("Analysis complete.")
    print(f"Figures saved in: {FIGURE_DIRECTORY}")
    print(f"Summary files saved in: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
