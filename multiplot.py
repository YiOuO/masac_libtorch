# multi_plot.py
import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

def infer_num_agents(ncols: int) -> int:
    # schema: epoch + 4N (pos/goal) + N (rewards) + done + status  => 5N+3
    if (ncols - 3) % 5 != 0 or ncols < 8:
        raise ValueError(f"CSV column count {ncols} does not match schema: epoch + 4N + N + done + status (5N+3).")
    return (ncols - 3) // 5

def col_indices(ncols: int, N: int):
    # After epoch(0), we have:
    # pos/goal block: indices 1 .. 1+4N-1  (each agent: 4 columns)
    # rewards block : indices 1+4N .. 1+4N+N-1
    # done          : index  1+5N
    # status        : index  2+5N
    pos_goal_start = 1
    rewards_start  = 1 + 4 * N
    done_idx       = 1 + 5 * N
    status_idx     = 2 + 5 * N

    # Mapping for each agent
    agent_cols = []
    for i in range(N):
        base = pos_goal_start + 4 * i
        agent_cols.append({
            "pos_x": base + 0,
            "pos_y": base + 1,
            "goal_x": base + 2,
            "goal_y": base + 3,
            "reward": rewards_start + i
        })
    return agent_cols, rewards_start, done_idx, status_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_file", type=str, default="data/ma_independent_q.csv")
    ap.add_argument("--epochs", nargs="+", default=["10"])
    ap.add_argument("--online_view", action="store_true")
    ap.add_argument("--output_path", type=str, default="img")
    ap.add_argument("--output_file", type=str, default="ma")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--tail_len", type=int, default=20)
    args = ap.parse_args()

    outdir = pathlib.Path(args.output_path)
    if not args.online_view:
        outdir.mkdir(parents=True, exist_ok=True)

    # 读取：跳过表头（你的 C++ 写了列名）
    data = np.genfromtxt(args.csv_file, delimiter=",", skip_header=1)
    if data.ndim == 1:  # 单行也变二维
        data = data.reshape(1, -1)

    ncols = data.shape[1]
    N = infer_num_agents(ncols)
    agent_cols, rewards_start, done_idx, status_idx = col_indices(ncols, N)
    print(f"[Info] detected {N} agents; columns={ncols}, done_idx={done_idx}, status_idx={status_idx}")

    # 全局坐标范围（所有 agent 的 pos 与 goal）
    xs, ys = [], []
    for c in agent_cols:
        xs += [data[:, c["pos_x"]], data[:, c["goal_x"]]]
        ys += [data[:, c["pos_y"]], data[:, c["goal_y"]]]
    x_min, x_max = np.nanmin(np.stack(xs)), np.nanmax(np.stack(xs))
    y_min, y_max = np.nanmin(np.stack(ys)), np.nanmax(np.stack(ys))
    if x_min == x_max: x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max: y_min, y_max = y_min - 1.0, y_max + 1.0
    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4","C5"])

    for e in args.epochs:
        e = int(e)
        epoch_rows = data[data[:, 0] == e]
        if len(epoch_rows) == 0:
            print(f"[Warn] no rows for epoch {e}, skip.")
            continue
        # epoch_rows = epoch_rows[::step]
        epoch_rows = epoch_rows[:1500]
        print(f"[Info] making animation: epoch={e}, frames={len(epoch_rows)}")

        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_title(f"Multi-Agent Environment (epoch {e})")
        ax.plot(0, 0, "x", c="black", label="Spawn")

        # per-agent artists
        agents, trails, goals, circles = [], [], [], []
        for i in range(N):
            col = colors[i % len(colors)]
            ag, = ax.plot([], [], "o", c=col, label=f"Agent {i}")
            tr, = ax.plot([], [], "-", c=col, alpha=0.8)
            gl, = ax.plot([], [], "o", c=col, alpha=0.7 if N > 1 else 1.0, label=f"Goal {i}")
            circ = plt.Circle((0, 0), 10, linestyle="--", color="gray", fill=False)
            ax.add_patch(circ)
            agents.append(ag); trails.append(tr); goals.append(gl); circles.append(circ)

        ax.legend(loc="upper right", ncol=2 if N > 3 else 1)
        title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                        bbox={"facecolor":"w","alpha":0.6,"pad":4})

        tails = [0]*N
        frame = 0

        def animate(_):
            nonlocal frame, tails
            if frame >= len(epoch_rows):
                frame = 0

            row = epoch_rows[frame]
            done = int(row[done_idx])
            status = int(row[status_idx])

            for i, c in enumerate(agent_cols):
                x, y = row[c["pos_x"]], row[c["pos_y"]]
                gx, gy = row[c["goal_x"]], row[c["goal_y"]]

                agents[i].set_data([x], [y])
                goals[i].set_data([gx], [gy])
                circles[i].center = (gx, gy)

                # 尾巴：遇到 done 或 status!=0 时重置
                if done == 1 or status != 0:
                    tails[i] = 0

                start = max(frame - tails[i], 0)
                trails[i].set_data(epoch_rows[start:frame, c["pos_x"]],
                                   epoch_rows[start:frame, c["pos_y"]])
                if tails[i] < args.tail_len:
                    tails[i] += 1

            title.set_text(f"Epoch {e} | Frame {frame+1}/{len(epoch_rows)} | Done={done} Status={status}")
            frame += 1

            artists = []
            for i in range(N):
                artists += [agents[i], trails[i], goals[i], circles[i]]
            artists += [title]
            return artists

        ani = animation.FuncAnimation(fig, animate, blit=True,
                                      interval=5, frames=len(epoch_rows), repeat=False)

        if args.online_view:
            plt.show()
        else:
            out = outdir / f"{args.output_file}_{e}.gif"
            print(f"[Info] saving to {out}")
            ani.save(str(out), writer=PillowWriter(fps=args.fps))
            plt.close(fig)

if __name__ == "__main__":
    main()
