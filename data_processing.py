"""Functions for loading and preparing the climate project datasets."""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


NASA_VALUE_COLUMNS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "J-D",
    "D-N",
    "DJF",
    "MAM",
    "JJA",
    "SON",
]

WEATHER_COLUMNS = [
    "country",
    "location_name",
    "last_updated",
    "temperature_celsius",
    "humidity",
    "precip_mm",
    "wind_kph",
]

WEATHER_COUNTRY_NAME_MAP = {
    "Russia": "Russian Federation",
    "United States of America": "United States",
    "USA United States of America": "United States",
}


def check_columns(data, required_columns, dataset_name):
    """Raises an error when a dataset is missing required columns."""
    missing = [
        column for column in required_columns if column not in data.columns
    ]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def clean_nasa_data(data):
    """Cleans a NASA global temperature DataFrame."""
    check_columns(data, ["Year", "J-D"], "NASA data")
    cleaned = data.copy()
    cleaned["Year"] = pd.to_numeric(cleaned["Year"], errors="coerce")

    available_columns = [
        column for column in NASA_VALUE_COLUMNS
        if column in cleaned.columns
    ]
    for column in available_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column].replace("***", np.nan),
            errors="coerce",
        )

    cleaned = cleaned.dropna(subset=["Year"])
    cleaned["Year"] = cleaned["Year"].astype(int)
    cleaned = cleaned.sort_values("Year").reset_index(drop=True)
    return cleaned


def load_nasa_data(file_path):
    """Loads and cleans the NASA global temperature CSV file."""
    data = pd.read_csv(file_path, skiprows=1, na_values="***")
    return clean_nasa_data(data)


def clean_weather_data(data):
    """Cleans recent weather observations and adds date columns."""
    check_columns(data, WEATHER_COLUMNS, "weather data")
    cleaned = data[WEATHER_COLUMNS].copy()
    cleaned["last_updated"] = pd.to_datetime(
        cleaned["last_updated"],
        errors="coerce",
    )

    numeric_columns = [
        "temperature_celsius",
        "humidity",
        "precip_mm",
        "wind_kph",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = cleaned.dropna(
        subset=["last_updated"] + numeric_columns
    )
    cleaned["year"] = cleaned["last_updated"].dt.year
    cleaned["month"] = cleaned["last_updated"].dt.month
    return cleaned.reset_index(drop=True)


def load_weather_data(file_path):
    """Loads selected columns from the global weather CSV file."""
    data = pd.read_csv(file_path, usecols=WEATHER_COLUMNS)
    return clean_weather_data(data)


def reshape_co2_data(data, start_year=1950, end_year=2021):
    """Filters total CO2 emissions and reshapes years into rows."""
    required = ["REF_AREA", "REF_AREA_LABEL", "INDICATOR"]
    check_columns(data, required, "CO2 data")

    year_columns = [
        str(year) for year in range(start_year, end_year + 1)
        if str(year) in data.columns
    ]
    if not year_columns:
        raise ValueError("CO2 data does not contain the requested years.")

    total_emissions = data[
        data["INDICATOR"] == "OWID_CB_CO2"
    ].copy()
    total_emissions = total_emissions[
        required + year_columns
    ]

    long_data = total_emissions.melt(
        id_vars=required,
        value_vars=year_columns,
        var_name="Year",
        value_name="CO2_Emissions",
    )
    long_data["Year"] = pd.to_numeric(
        long_data["Year"],
        errors="coerce",
    )
    long_data["CO2_Emissions"] = pd.to_numeric(
        long_data["CO2_Emissions"],
        errors="coerce",
    )
    long_data = long_data.dropna(
        subset=["Year", "CO2_Emissions"]
    )
    long_data["Year"] = long_data["Year"].astype(int)
    return long_data.sort_values(
        ["REF_AREA_LABEL", "Year"]
    ).reset_index(drop=True)


def load_co2_data(file_path, start_year=1950, end_year=2021):
    """Loads and reshapes total annual CO2 emissions data."""
    data = pd.read_csv(file_path)
    return reshape_co2_data(data, start_year, end_year)


def get_world_co2(co2_data):
    """Returns annual total CO2 emissions for the World row."""
    world = co2_data[
        co2_data["REF_AREA_LABEL"] == "World"
    ].copy()
    return world.sort_values("Year").reset_index(drop=True)


def get_country_co2(co2_data, countries):
    """Returns annual CO2 emissions for selected countries."""
    selected = co2_data[
        co2_data["REF_AREA_LABEL"].isin(countries)
    ].copy()
    return selected.sort_values(
        ["REF_AREA_LABEL", "Year"]
    ).reset_index(drop=True)


def merge_temperature_co2(nasa_data, world_co2):
    """Merges annual NASA anomalies and world CO2 emissions."""
    temperature = nasa_data[["Year", "J-D"]].dropna().copy()
    merged = temperature.merge(
        world_co2[["Year", "CO2_Emissions"]],
        on="Year",
        how="inner",
    )
    return merged.sort_values("Year").reset_index(drop=True)


def merge_country_co2_weather(
    co2_data,
    weather_data,
    countries,
    co2_year=2021,
):
    """Combines country CO2 emissions with recent weather averages."""
    co2_selected = co2_data[
        (co2_data["Year"] == co2_year)
        & (co2_data["REF_AREA_LABEL"].isin(countries))
    ][["REF_AREA_LABEL", "Year", "CO2_Emissions"]].copy()

    weather_selected = weather_data.copy()
    weather_selected["REF_AREA_LABEL"] = weather_selected[
        "country"
    ].replace(WEATHER_COUNTRY_NAME_MAP)
    weather_selected = weather_selected[
        weather_selected["REF_AREA_LABEL"].isin(countries)
    ]

    measurements = [
        "temperature_celsius",
        "humidity",
        "precip_mm",
        "wind_kph",
    ]
    weather_summary = weather_selected.groupby(
        "REF_AREA_LABEL",
        as_index=False,
    )[measurements].mean()

    merged = co2_selected.merge(
        weather_summary,
        on="REF_AREA_LABEL",
        how="inner",
    )
    return merged.sort_values(
        "CO2_Emissions",
        ascending=False,
    ).reset_index(drop=True)


def summarize_weather_by_month(weather_data):
    """Calculates monthly averages for major weather measurements."""
    measurements = [
        "temperature_celsius",
        "humidity",
        "precip_mm",
        "wind_kph",
    ]
    summary = weather_data.groupby(
        ["year", "month"],
        as_index=False,
    )[measurements].mean()
    return summary.sort_values(
        ["year", "month"]
    ).reset_index(drop=True)


def calculate_slope(data, x_column, y_column):
    """Calculates the linear slope of one numeric column over another."""
    subset = data[[x_column, y_column]].dropna()
    if len(subset) < 2:
        raise ValueError("At least two complete rows are needed for a slope.")
    slope, _ = np.polyfit(subset[x_column], subset[y_column], 1)
    return float(slope)


def test_temperature_period_difference(nasa_data):
    """Runs a one-sided Welch t-test on two temperature periods."""
    early = nasa_data[
        nasa_data["Year"].between(1950, 1999)
    ]["J-D"].dropna()
    recent = nasa_data[
        nasa_data["Year"].between(2000, 2024)
    ]["J-D"].dropna()

    if len(early) < 2 or len(recent) < 2:
        raise ValueError("Each temperature period needs at least two values.")

    result = ttest_ind(
        recent,
        early,
        equal_var=False,
        alternative="greater",
    )
    return (
        float(early.mean()),
        float(recent.mean()),
        float(result.statistic),
        float(result.pvalue),
    )
