import pandas as pd
import plotly.express as px


url = (
    "https://raw.githubusercontent.com/leontoddjohnson/"
    "datasets/main/data/titanic.csv"
)


def load_data():
    """Load the Titanic dataset and clean the column names."""
    df = pd.read_csv(url)

    df.columns = (
        df.columns
        .str.replace(r"([a-z0-9])([A-Z])", r"\1_\2", regex=True)
        .str.replace(r"[\s-]+", "_", regex=True)
        .str.lower()
    )

    return df


def survival_demographics():
    """Return survival statistics by class, sex, and age group."""
    df = load_data()

    # Divide passenger ages into the required age groups.
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 19, 59, float("inf")],
        labels=["Child", "Teen", "Adult", "Senior"],
        include_lowest=True
    )

    results = (
        df.groupby(
            ["pclass", "sex", "age_group"],
            observed=False
        )
        .agg(
            n_passengers=("survived", "size"),
            n_survivors=("survived", "sum")
        )
        .reset_index()
    )

    results["survival_rate"] = (
        results["n_survivors"] / results["n_passengers"]
    )

    return (
        results.sort_values(
            ["pclass", "sex", "age_group"]
        )
        .reset_index(drop=True)
    )


def visualize_demographic():
    """Return a Plotly figure comparing Titanic survival rates."""
    df = survival_demographics()

    return px.bar(
        df,
        x="age_group",
        y="survival_rate",
        color="sex",
        facet_col="pclass",
        barmode="group",
        labels={
            "age_group": "Age Group",
            "survival_rate": "Survival Rate",
            "sex": "Sex",
            "pclass": "Passenger Class"
        },
        title=(
            "Titanic Survival Rates by Age Group, Sex, "
            "and Passenger Class"
        )
    )


def family_groups():
    """Return fare statistics by family size and passenger class."""
    df = load_data()

    # Include the passenger when calculating total family size.
    df["family_size"] = df["sib_sp"] + df["parch"] + 1

    return (
        df.groupby(["pclass", "family_size"])
        .agg(
            n_passengers=("passenger_id", "size"),
            avg_fare=("fare", "mean"),
            min_fare=("fare", "min"),
            max_fare=("fare", "max")
        )
        .reset_index()
        .sort_values(["pclass", "family_size"])
        .reset_index(drop=True)
    )


def last_names():
    """Return the number of passengers sharing each last name."""
    df = load_data()

    # Names are formatted with the passenger's last name first.
    last_names = (
        df["name"]
        .str.split(",")
        .str[0]
        .str.strip()
    )

    return last_names.value_counts()


def visualize_families():
    """Return a Plotly figure comparing family size and average fare."""
    df = family_groups()

    # Treat passenger class as a category in the visualization.
    df["pclass"] = df["pclass"].astype(str)

    return px.scatter(
        df,
        x="family_size",
        y="avg_fare",
        color="pclass",
        size="n_passengers",
        labels={
            "family_size": "Family Size",
            "avg_fare": "Average Fare",
            "pclass": "Passenger Class",
            "n_passengers": "Number of Passengers"
        },
        title="Average Fare by Family Size and Passenger Class"
    )
