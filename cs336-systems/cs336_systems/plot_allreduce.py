import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("allreduce_results.csv")

for backend in df.backend.unique():
    plt.figure(figsize=(6,4))
    sub = df[df.backend==backend]
    for dev in sub.device.unique():
        df2 = sub[sub.device==dev]
        plt.plot(
            df2.elements/1e6,
            df2.time_s,
            marker='o',
            label=f"{dev.upper()}, procs={df2.processes.values}"
        )
    plt.xscale('log')
    plt.xlabel("Tensor size (M elements)")
    plt.ylabel("All-reduce time (s)")
    plt.title(f"Backend: {backend.upper()}")
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.show()
