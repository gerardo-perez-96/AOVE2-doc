"""
Benchmark de LSTM (PyTorch) sobre 'Quality Prediction in a Mining Process'
con barrido de configuraciones -> tabla de resultados para dimensionar specs.

Controla tres fuentes de error habituales en este tipo de benchmark:
  1. Sesgo termico de orden: las configs se corren en orden ALEATORIO y en
     varias rondas intercaladas; se reporta la MEDIANA, no la media de un
     unico pase.
  2. Estado del optimizer: cada config crea modelo + Adam nuevos. Reusar el
     optimizer entre configs deja momentum de la config anterior y falsea
     los primeros batches.
  3. Throttling sostenido: un benchmark de 50 batches (~6s) no lo detecta.
     Se añade una comprobacion de deriva (primera mitad vs segunda mitad)
     en un run mas largo para la config mas grande del barrido.

Uso:
    python lstm_benchmark_sweep.py --csv MiningProcess_Flotation_Plant_Database.csv \
        --seq-lens 60,180,360 --hiddens 32,64,128 --repeats 3 --target-epochs 20
"""

import argparse
import itertools
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[aviso] psutil no instalado -> no se medira RAM. "
          "pip install psutil --break-system-packages")

# ----------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--csv", default="MiningProcess_Flotation_Plant_Database.csv")
p.add_argument("--seq-lens", default="60,180,360")
p.add_argument("--hiddens", default="32,64,128")
p.add_argument("--layers", type=int, default=1)
p.add_argument("--stride", type=int, default=30)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--bench-batches", type=int, default=50)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--repeats", type=int, default=3,
               help="rondas completas del barrido, intercaladas, para sacar mediana")
p.add_argument("--target-epochs", type=int, default=20,
               help="epochs objetivo para extrapolar el coste total real de tu proyecto")
p.add_argument("--sustained-batches", type=int, default=300,
               help="batches para el chequeo de throttling en la config mas grande")
p.add_argument("--cooldown", type=float, default=2.0,
               help="segundos de pausa entre configs, para no acumular calor artificialmente")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="benchmark_results.csv")
args = p.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
proc = psutil.Process() if HAS_PSUTIL else None

# ----------------------------------------------------------------------------
# 1. Datos (una sola vez, se reutilizan tensores base entre configs)
# ----------------------------------------------------------------------------
df = pd.read_csv(args.csv, decimal=",", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

TARGET = "% Silica Concentrate"
DROP = ["date", "% Iron Concentrate", TARGET]  # leakage: casi complemento del target
feature_cols = [c for c in df.columns if c not in DROP]

X = df[feature_cols].to_numpy(dtype=np.float32)
y = df[TARGET].to_numpy(dtype=np.float32)
n = len(df)

i_tr = int(n * 0.70)
i_va = int(n * 0.85)

mu, sigma = X[:i_tr].mean(0), X[:i_tr].std(0) + 1e-8
X = (X - mu) / sigma
y_mu, y_sigma = y[:i_tr].mean(), y[:i_tr].std() + 1e-8
y = (y - y_mu) / y_sigma

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)

print(f"Filas: {n:,} | Features: {len(feature_cols)} | Device: {DEVICE} | "
      f"HAS_PSUTIL: {HAS_PSUTIL}")


class WindowDataset(Dataset):
    def __init__(self, X, y, lo, hi, seq_len, stride):
        self.X, self.y, self.L = X, y, seq_len
        self.starts = np.arange(lo, hi - seq_len + 1, stride, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return self.X[s:s + self.L], self.y[s + self.L - 1]


class LSTMRegressor(nn.Module):
    def __init__(self, n_feat, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def make_loader(seq_len):
    ds = WindowDataset(X_t, y_t, 0, i_tr, seq_len, args.stride)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True), len(ds)


def run_one_config(seq_len, hidden, n_batches, warmup):
    """Modelo y optimizer NUEVOS -> sin contaminacion de estado entre configs."""
    dl, n_windows = make_loader(seq_len)
    model = LSTMRegressor(len(feature_cols), hidden, args.layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    model.train()

    it = iter(dl)
    per_batch_ms = []
    if HAS_PSUTIL:
        ram_before = proc.memory_info().rss / 1e9

    for i in range(warmup + n_batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)

        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        sync(); t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()

        sync()
        dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            per_batch_ms.append(dt)

    ram_peak = None
    if HAS_PSUTIL:
        ram_peak = proc.memory_info().rss / 1e9

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "per_batch_ms": per_batch_ms,
        "n_windows_train": n_windows,
        "n_params": n_params,
        "ram_gb": ram_peak,
        "batches_per_epoch": n_windows // args.batch_size,
    }


# ----------------------------------------------------------------------------
# 2. Barrido: rondas intercaladas en orden aleatorio (anti-sesgo termico)
# ----------------------------------------------------------------------------
seq_lens = [int(x) for x in args.seq_lens.split(",")]
hiddens = [int(x) for x in args.hiddens.split(",")]
configs = list(itertools.product(seq_lens, hiddens))

raw = {c: [] for c in configs}   # (seq_len, hidden) -> lista de listas de ms/batch
meta = {}

print(f"\n{len(configs)} configs x {args.repeats} rondas intercaladas "
      f"= {len(configs)*args.repeats} runs\n")

for r in range(args.repeats):
    order = configs.copy()
    random.shuffle(order)
    for (seq_len, hidden) in order:
        res = run_one_config(seq_len, hidden, args.bench_batches, args.warmup)
        raw[(seq_len, hidden)].append(res["per_batch_ms"])
        meta[(seq_len, hidden)] = res  # nos quedamos con la ultima para metadatos fijos
        median_ms = np.median(res["per_batch_ms"])
        print(f"[ronda {r+1}/{args.repeats}] seq_len={seq_len:4d} hidden={hidden:4d} "
              f"-> mediana {median_ms:7.1f} ms/batch")
        time.sleep(args.cooldown)

# ----------------------------------------------------------------------------
# 3. Chequeo de throttling sostenido en la config MAS CARA del barrido
# ----------------------------------------------------------------------------
biggest = max(configs, key=lambda c: c[0] * c[1] * c[1])  # ~ seq_len * hidden^2
print(f"\nChequeo de throttling sostenido en la config mas cara: "
      f"seq_len={biggest[0]}, hidden={biggest[1]} ({args.sustained_batches} batches)")

sustained = run_one_config(biggest[0], biggest[1], args.sustained_batches, args.warmup)
half = len(sustained["per_batch_ms"]) // 2
first_half = np.median(sustained["per_batch_ms"][:half])
second_half = np.median(sustained["per_batch_ms"][half:])
drift_pct = (second_half - first_half) / first_half * 100

print(f"  primera mitad : {first_half:7.1f} ms/batch")
print(f"  segunda mitad : {second_half:7.1f} ms/batch")
print(f"  deriva        : {drift_pct:+6.1f} %")
if drift_pct > 15:
    print("  ALERTA: deriva >15% -> throttling probable. Los tiempos extrapolados "
          "de un benchmark corto se quedan CORTOS. Usa la segunda mitad como referencia,")
    print("  no la mediana del barrido corto.")
throttling_flag = drift_pct > 15

# ----------------------------------------------------------------------------
# 4. Tabla final
# ----------------------------------------------------------------------------
rows = []
for (seq_len, hidden), lists in raw.items():
    all_ms = np.concatenate(lists)          # todas las rondas juntas
    m = meta[(seq_len, hidden)]
    median_ms = np.median(all_ms)
    std_ms = np.std(all_ms)
    epoch_min = median_ms / 1000 * m["batches_per_epoch"] / 60
    total_h = epoch_min * args.target_epochs / 60

    # si detectamos throttling, corregimos con el factor de deriva medido
    corrected_total_h = total_h * (1 + max(drift_pct, 0) / 100) if throttling_flag else total_h

    rows.append({
        "seq_len": seq_len,
        "hidden": hidden,
        "n_params": m["n_params"],
        "n_windows_train": m["n_windows_train"],
        "batches_epoch": m["batches_per_epoch"],
        "ms_batch_mediana": round(median_ms, 1),
        "ms_batch_std": round(std_ms, 1),
        "ram_gb": round(m["ram_gb"], 2) if m["ram_gb"] else None,
        "min_por_epoch": round(epoch_min, 2),
        f"horas_{args.target_epochs}_epochs": round(total_h, 2),
        f"horas_{args.target_epochs}_epochs_corregido_throttling": round(corrected_total_h, 2),
    })

table = pd.DataFrame(rows).sort_values(["seq_len", "hidden"]).reset_index(drop=True)
pd.set_option("display.width", 160)
print("\n=== TABLA DE RESULTADOS ===\n")
print(table.to_string(index=False))

table.to_csv(args.out, index=False)
print(f"\nGuardado en {args.out}")

if throttling_flag:
    print(f"\n[!] Se detecto throttling ({drift_pct:+.1f}% de deriva). La columna "
          f"'corregido_throttling' es una estimacion conservadora, no una medicion "
          f"directa. Para decidir specs de proyecto con confianza, corre la config "
          f"que finalmente elijas durante {args.sustained_batches*3}+ batches reales "
          f"y compara contra esta tabla.")