"""
Build a sidecar survival-rate table for Greece from DATA_GR.xlsx → data/survival_GR.npz.

Source: 'Survival rates' sheet — Eurostat demo_mlifetable, px = probability of
surviving between exact ages, Greece, Total sex. Real ages 0–85 ("85 or over"),
years 1960–2023 (1960 is all ':' = missing; usable 1961–2023).

The OLG model enters at real age 25 (model age 0), T=60 → real ages 25–84.
We slice columns to real ages 25–84 and store px[year, model_age] with
model_age j ↔ real age 25+j.

Output npz:
  years     int   (Ny,)        calendar years, ascending (1961..2023)
  px        float (Ny, 60)     survival prob, px[i, j] = px(real age 25+j, years[i])
  model_ages int  (60,)        0..59
  real_ages  int  (60,)        25..84
"""
import numpy as np
import pandas as pd

SRC = 'data/DATA_GR.xlsx'      # run from repo root (dsa/)
OUT = 'data/survival_GR.npz'
PROJ = 'data/europop2023_GR.npz'
ENTRY_AGE = 25
T = 75                          # model horizon → real ages 25..99
N_HIST = 60                     # real ages 25..84 are in the historical table

df = pd.read_excel(SRC, sheet_name='Survival rates', header=None)
# Layout: row 8 = age labels (cols 2..87 → real ages 0..85),
#         row 9 = 'GEO (Labels)' / 'TIME', rows 10..73 = data (col0=Greece, col1=year).
years_all = df.iloc[10:74, 1].astype(int).to_numpy()
data_all = df.iloc[10:74, 2:88].replace(':', np.nan).astype(float).to_numpy()  # (64, 86) real ages 0..85

# Real ages 25..84 from the sheet (columns 25..84 in the 0..85 block); real
# ages 85..99 from the EUROPOP2023 2022 schedule, the same in every year.
real_ages = np.arange(ENTRY_AGE, ENTRY_AGE + T)        # 25..99
px_hist = data_all[:, real_ages[:N_HIST]]              # (64, 60)
proj = np.load(PROJ)
assert proj['px'].shape[1] == T and int(proj['years'][0]) == 2022
px_old = np.tile(proj['px'][0, N_HIST:], (px_hist.shape[0], 1))   # (64, 15)
px_all = np.concatenate([px_hist, px_old], axis=1)     # (64, 75)

# Drop years with any missing value in the model age range (1960 only).
keep = ~np.isnan(px_all).any(axis=1)
years = years_all[keep]
px = px_all[keep]

assert np.all((px > 0) & (px <= 1)), "px out of (0,1] range"
order = np.argsort(years)
years = years[order]
px = px[order]

np.savez(OUT,
         years=years.astype(int),
         px=px.astype(float),
         model_ages=np.arange(T, dtype=int),
         real_ages=real_ages.astype(int))
print(f"wrote {OUT}: years {years[0]}..{years[-1]} (n={len(years)}), "
      f"px shape {px.shape}, real ages {real_ages[0]}..{real_ages[-1]}")
print(f"  px[2020, age0..4] = {np.round(px[years==2020][0, :5], 5)}")
print(f"  px[2020, age55..59] = {np.round(px[years==2020][0, 55:60], 5)}")
print(f"  px[2020, age70..74] = {np.round(px[years==2020][0, -5:], 5)} (EUROPOP 2022 schedule)")
