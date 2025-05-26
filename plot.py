import argparse
import pathlib

import matplotlib.animation as animation
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

    for e in args.epochs:
        e = int(e)
        epoch_data = data[np.where(data[:, 0] == e)]

        # Animation state
        global tail, frame
        tail, frame = 0, 0

        def animate(i):
            global tail, frame

            if frame >= len(epoch_data):
                frame = 0  # Loop or stop here if needed

            # Position update
            x = epoch_data[frame, 1]
            y = epoch_data[frame, 2]
            agent.set_data([x], [y])

            # Tail logic (reset when goal/win/lose)
            if epoch_data[frame, 5] in [1, 2, 3]:
                tail = 0
            agent_line.set_data(
                epoch_data[max(frame - tail, 0):frame, 1],
                epoch_data[max(frame - tail, 0):frame, 2]
            )
            if tail < 50:
                tail += 1

            # Goal + visual circle
            gx, gy = epoch_data[frame, 3], epoch_data[frame, 4]
            goal.set_data([gx], [gy])
            circle.center = (gx, gy)

            # Title
            title.set_text(f"Epoch {int(epoch_data[frame, 0])}")

            frame += 1
            return agent, agent_line, goal, circle, title

        # Create animation
        ani = animation.FuncAnimation(
            fig, animate, blit=True, interval=5, frames=len(epoch_data))

        # Show or save
        if args.online_view:
            plt.show()
        else:
            output_file = path / f"{args.output_file}_{e}.gif"
            ani.save(str(output_file), writer="imagemagick", fps=100)


if __name__ == "__main__":
    main()
