import argparse
import pathlib

import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="data/data.csv",
                        help="Path to the generated trajectories.")
    parser.add_argument("--epochs", nargs="+", default="0",
                        help="Epochs to be plotted.")
    parser.add_argument("--online_view", action="store_true",
                        help="Whether to show online view or generate gif.")
    parser.add_argument("--output_path", type=str, default="img",
                        help="The path to write generated gifs to.")
    parser.add_argument("--output_file", type=str,
                        default="test", help="The prefix of the gif.")
    args = parser.parse_args()

    # Create output path
    path = pathlib.Path(args.output_path)
    if not args.online_view and not path.exists():
        path.mkdir(parents=True)

    # Load CSV data
    data = np.genfromtxt(args.csv_file, delimiter=",")

    for e in args.epochs:
        e = int(e)
        epoch_data = data[np.where(data[:, 0] == e)]

        if len(epoch_data) == 0:
            print(f"[Warning] No data found for epoch {e}, skipping.")
            continue
        # save export time
        epoch_data = epoch_data[:1500]
        print(f"[Info] Creating animation for epoch {e}, total frames: {len(epoch_data)}")

        fig, ax = plt.subplots()

        # Spawn point
        ax.plot(0, 0, "x", c="black", label="Spawn")

        # Placeholder elements
        agent, = ax.plot([], [], "o", c="b", label="Agent")
        agent_line, = ax.plot([], [], "-", c="b")
        goal, = ax.plot([], [], "o", c="r", label="Goal")
        circle = plt.Circle((0, 0), 10, linestyle="--", color="gray", fill=False, label="Maximum Goal Distance")
        ax.add_patch(circle)

        # Axes settings
        ax.set_xlabel("x / a.u.")
        ax.set_ylabel("y / a.u.")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_title("Agent in Test Environment")
        ax.legend()
        title = ax.text(0.15, 0.85, "", bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
                        transform=ax.transAxes, ha="center")

        tail, frame = 0, 0

        def animate(i):
            nonlocal tail, frame

            if frame >= len(epoch_data):
                frame = 0  # Could also call ani.event_source.stop() to stop looping

            x = epoch_data[frame, 1]
            y = epoch_data[frame, 2]
            agent.set_data([x], [y])

            if epoch_data[frame, 5] in [1, 2, 3]:
                tail = 0
            agent_line.set_data(
                epoch_data[max(frame - tail, 0):frame, 1],
                epoch_data[max(frame - tail, 0):frame, 2]
            )
            if tail < 50:
                tail += 1

            gx, gy = epoch_data[frame, 3], epoch_data[frame, 4]
            goal.set_data([gx], [gy])
            circle.center = (gx, gy)

            title.set_text(f"Epoch {int(epoch_data[frame, 0])}")
            frame += 1

            return agent, agent_line, goal, circle, title

        ani = animation.FuncAnimation(
            fig, animate, blit=True, interval=5, frames=len(epoch_data), repeat=False
        )

        if args.online_view:
            plt.show()
        else:
            output_file = path / f"{args.output_file}_{e}.gif"
            print(f"[Info] Saving gif to {output_file}")
            writer = PillowWriter(fps=30)
            ani.save(str(output_file), writer=writer)
            plt.close(fig)

            # # write to mp4
            # writer = FFMpegWriter(fps=50, bitrate=1800)
            # ani.save(str(output_file.with_suffix(".mp4")), writer=writer)


if __name__ == "__main__":
    main()
