"""Plotting functions for the climate data analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np


SEASONS = ["DJF", "MAM", "JJA", "SON"]


def save_figure(output_file):
    """Saves the current figure and closes it."""
    parent = os.path.dirname(output_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_annual_temperature(nasa_data, output_file):
    """Plots annual global temperature anomalies over time."""
    data = nasa_data[["Year", "J-D"]].dropna().copy()
    rolling = data["J-D"].rolling(5, center=True).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(
        data["Year"],
        data["J-D"],
        linewidth=1,
        alpha=0.65,
        label="Annual anomaly",
    )
    plt.plot(
        data["Year"],
        rolling,
        linewidth=2.5,
        label="5-year moving average",
    )
    plt.title("Global Annual Temperature Anomalies")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (degrees C)")
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure(output_file)


def plot_seasonal_temperature(nasa_data, output_file):
    """Plots seasonal global temperature anomalies over time."""
    plt.figure(figsize=(10, 6))
    for season in SEASONS:
        if season in nasa_data.columns:
            data = nasa_data[["Year", season]].dropna()
            plt.plot(
                data["Year"],
                data[season],
                linewidth=1.2,
                label=season,
            )

    plt.title("Seasonal Global Temperature Anomalies")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (degrees C)")
    plt.grid(alpha=0.3)
    plt.legend(title="Season")
    save_figure(output_file)


def plot_world_co2(world_co2, output_file):
    """Plots annual world CO2 emissions."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        world_co2["Year"],
        world_co2["CO2_Emissions"],
        linewidth=2,
    )
    plt.title("World Annual CO2 Emissions")
    plt.xlabel("Year")
    plt.ylabel("CO2 Emissions (million tonnes)")
    plt.grid(alpha=0.3)
    save_figure(output_file)


def plot_country_co2(country_co2, output_file):
    """Plots annual CO2 emissions for selected countries."""
    plt.figure(figsize=(10, 6))
    grouped = country_co2.groupby("REF_AREA_LABEL")
    for country, data in grouped:
        plt.plot(
            data["Year"],
            data["CO2_Emissions"],
            linewidth=1.8,
            label=country,
        )

    plt.title("Annual CO2 Emissions for Selected Countries")
    plt.xlabel("Year")
    plt.ylabel("CO2 Emissions (million tonnes)")
    plt.grid(alpha=0.3)
    plt.legend(title="Country")
    save_figure(output_file)


def plot_monthly_weather(monthly_weather, output_file):
    """Plots monthly average temperatures for each available year."""
    plt.figure(figsize=(10, 6))
    grouped = monthly_weather.groupby("year")
    for year, data in grouped:
        plt.plot(
            data["month"],
            data["temperature_celsius"],
            marker="o",
            label=str(year),
        )

    plt.title("Monthly Average Temperature in Weather Observations")
    plt.xlabel("Month")
    plt.ylabel("Average Temperature (degrees C)")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.legend(title="Year")
    save_figure(output_file)


def plot_temperature_precipitation(weather_data, output_file):
    """Plots temperature against precipitation for a data sample."""
    sample_size = min(5000, len(weather_data))
    sample = weather_data.sample(
        n=sample_size,
        random_state=163,
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        sample["temperature_celsius"],
        sample["precip_mm"],
        alpha=0.25,
        s=14,
    )
    plt.title("Temperature and Precipitation in Weather Observations")
    plt.xlabel("Temperature (degrees C)")
    plt.ylabel("Precipitation (mm)")
    plt.grid(alpha=0.3)
    save_figure(output_file)


def plot_temperature_co2(merged_data, output_file):
    """Plots global temperature anomalies against world CO2 emissions."""
    x_values = merged_data["CO2_Emissions"]
    y_values = merged_data["J-D"]
    slope, intercept = np.polyfit(x_values, y_values, 1)
    trend = slope * x_values + intercept

    order = np.argsort(x_values.to_numpy())
    sorted_x = x_values.to_numpy()[order]
    sorted_trend = trend.to_numpy()[order]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.7)
    plt.plot(
        sorted_x,
        sorted_trend,
        linewidth=2,
        label="Linear trend",
    )
    plt.title("Global Temperature Anomaly vs. World CO2 Emissions")
    plt.xlabel("CO2 Emissions (million tonnes)")
    plt.ylabel("Temperature Anomaly (degrees C)")
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure(output_file)


def plot_country_co2_weather(country_weather, output_file):
    """Plots 2021 emissions against recent average temperature."""
    plt.figure(figsize=(10, 6))
    plt.scatter(
        country_weather["CO2_Emissions"],
        country_weather["temperature_celsius"],
        s=70,
    )

    for _, row in country_weather.iterrows():
        plt.annotate(
            row["REF_AREA_LABEL"],
            (
                row["CO2_Emissions"],
                row["temperature_celsius"],
            ),
        )

    plt.title("2021 CO2 Emissions vs. Recent Average Temperature")
    plt.xlabel("2021 CO2 Emissions (million tonnes)")
    plt.ylabel("2024-2025 Average Temperature (degrees C)")
    plt.grid(alpha=0.3)
    save_figure(output_file)
