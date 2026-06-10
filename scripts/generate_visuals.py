"""Generate all exact-solution visualizations and save to results/figures/."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tgv.visualization import (
    plot_flow_field,
    plot_energy_decay,
    animate_vortex,
    plot_time_strip,
)

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)

print("Generating flow field snapshots...")
for Re in [10, 100]:
    for t in [0.0, 0.5, 1.0]:
        plot_flow_field(t=t, Re=Re, N=128,
                        save_path=f"{FIG}/flow_field_Re{Re:03d}_t{t:.1f}.png")

print("Generating time-strip (vorticity)...")
for Re in [10, 100]:
    plot_time_strip(Re=Re, field="vorticity", times=[0.0, 0.25, 0.5, 1.0, 2.0], N=128,
                    save_path=f"{FIG}/time_strip_vorticity_Re{Re:03d}.png")

print("Generating time-strip (speed)...")
plot_time_strip(Re=100, field="speed", times=[0.0, 0.25, 0.5, 1.0, 2.0], N=128,
                save_path=f"{FIG}/time_strip_speed_Re100.png")

print("Generating energy decay curves...")
plot_energy_decay(Re_list=[10, 50, 100, 200, 400], t_end=2.0,
                  save_path=f"{FIG}/energy_decay.png")

print("Generating animation (Re=100, 40 frames)...")
animate_vortex(Re=100, t_end=2.0, n_frames=40, N=96, fps=10,
               save_path=f"{FIG}/vortex_animation_Re100.gif")

print("Generating animation (Re=10, 30 frames)...")
animate_vortex(Re=10, t_end=1.0, n_frames=30, N=96, fps=8,
               save_path=f"{FIG}/vortex_animation_Re10.gif")

print("\nAll visuals generated in results/figures/")
