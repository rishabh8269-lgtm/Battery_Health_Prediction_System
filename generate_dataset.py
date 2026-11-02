"""
Generate the synthetic battery dataset used by train_model.py.

SOH (state of health, %) falls as the battery accumulates *effective*
cycles. Each real cycle counts for more or less depending on how hard
the battery is used:

    effective_cycles = cycles x stress

where stress multiplies independent penalty factors for temperature,
depth of discharge, charge rate, average cell voltage and current.
SOH then decays toward a 50 % floor (the lowest value the app reports):

    SOH = 100 - 50 x (1 - exp(-effective_cycles / SCALE)) + noise

These are textbook-style assumptions (heat, deep discharge, fast
charging and high-voltage storage all age lithium-ion cells faster),
NOT measurements. The data is synthetic and for demonstration only.

Run:  python generate_dataset.py   then   python train_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "data" / "battery_dataset.csv"

N_ROWS = 6000
SEED = 42
SCALE = 3500.0        # larger = slower aging overall
FLOOR_DROP = 50.0     # SOH approaches 100 - 50 = 50 %
NOISE_SD = 1.0        # measurement / cell-to-cell noise (SOH points)


def stress_factor(temp, volt, curr, dod, rate):
    """Multiplier on how fast each real cycle ages the battery."""
    hot = np.maximum(temp - 25.0, 0.0)
    cold = np.maximum(15.0 - temp, 0.0)

    f_temp = 1.0 + 0.035 * hot + 0.0008 * hot ** 2 + 0.04 * cold
    f_dod = (dod / 70.0) ** 1.3
    f_rate = 1.0 + 0.5 * np.maximum(rate - 0.5, 0.0) ** 1.5
    f_volt = 1.0 + 3.0 * np.maximum(volt - 4.0, 0.0) \
                 + 0.5 * np.maximum(3.3 - volt, 0.0)
    f_curr = 1.0 + 0.10 * np.maximum(curr - 2.0, 0.0)

    return f_temp * f_dod * f_rate * f_volt * f_curr


def soh_from_inputs(cycles, temp, volt, curr, dod, rate, rng=None):
    stress = stress_factor(temp, volt, curr, dod, rate)

    if rng is not None:
        # Cell-to-cell variation in how sensitive each battery is.
        stress = stress * rng.lognormal(mean=0.0, sigma=0.10, size=len(stress))

    effective = cycles * stress
    soh = 100.0 - FLOOR_DROP * (1.0 - np.exp(-effective / SCALE))

    if rng is not None:
        soh = soh + rng.normal(0.0, NOISE_SD, size=len(soh))

    return np.clip(soh, 50.0, 100.0)


def main():
    rng = np.random.default_rng(SEED)
    n = N_ROWS

    cycles = rng.integers(0, 1501, n)

    # Half realistic mid-range values, half uniform so extremes are covered.
    mix = rng.random(n) < 0.5
    temp = np.where(mix, rng.normal(30, 8, n), rng.uniform(0, 55, n))
    temp = np.clip(temp, 0.0, 55.0)

    mix = rng.random(n) < 0.5
    volt = np.where(mix, rng.normal(3.85, 0.15, n), rng.uniform(3.0, 4.25, n))
    volt = np.clip(volt, 3.0, 4.25)

    curr = rng.uniform(0.1, 4.0, n)
    dod = rng.integers(20, 101, n)
    rate = rng.uniform(0.2, 2.5, n)

    soh = soh_from_inputs(cycles, temp, volt, curr, dod, rate, rng)

    df = pd.DataFrame(
        {
            "Cycle_Count": cycles,
            "Avg_Temperature_C": temp.round(1),
            "Avg_Cell_Voltage_V": volt.round(3),
            "Avg_Current_A": curr.round(2),
            "Depth_of_Discharge_Percent": dod,
            "Charge_Rate_C": rate.round(2),
            "SOH_Percent": soh.round(2),
        }
    )

    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(df):,} rows to {OUT_PATH}")
    print(df["SOH_Percent"].describe().round(2))
    print(f"Share of rows below 90 %: {(df['SOH_Percent'] < 90).mean():.1%}")
    print(f"Share of rows below 80 %: {(df['SOH_Percent'] < 80).mean():.1%}")


if __name__ == "__main__":
    main()
