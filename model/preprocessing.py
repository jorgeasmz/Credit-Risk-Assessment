import pandas as pd

from model.config import COLUMNS, DATA_URL, TARGET_COLUMN, TARGET_MAPPING


def load_data(source: str = DATA_URL) -> pd.DataFrame:
    """
    Loads the German Credit dataset and maps the target to 0 = good, 1 = bad.

    Accepts a URL or a local path.

    Raises:
        ValueError: if the target column holds an undocumented value.
    """
    df = pd.read_csv(source, sep=" ", names=COLUMNS)

    mapped = df[TARGET_COLUMN].map(TARGET_MAPPING)
    if mapped.isna().any():
        unexpected = sorted(set(df.loc[mapped.isna(), TARGET_COLUMN]))
        raise ValueError(
            f"Unexpected value(s) in '{TARGET_COLUMN}': {unexpected}. "
            f"Expected one of {sorted(TARGET_MAPPING)}."
        )
    df[TARGET_COLUMN] = mapped.astype(int)

    return df
