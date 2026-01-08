import os
import argparse

# Force an interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from stable_baselines3 import PPO
from two_robot_meetup_env import TwoRobotMeetupEnvCfg


def rollout(model_path: str, n_episodes: int = 5, max_steps: int = 300, curriculum_phase: int = 2):
    """
    Visualize trained policy with detailed metrics (holonomic version).
    
    Args:
        model_path: Path to PPO checkpoint
        n_episodes: Number of episodes to visualize
        max_steps: Max steps per episode
        curriculum_phase: Which curriculum phase to use (0=approach, 1=orient, 2=meetup)
    """
    
    # Load env & model
    env = TwoRobotMeetupEnvCfg(curriculum_phase=curriculum_phase)
    model = PPO.load(model_path)

    room_half = env.room_half_size
    robot_radius = env.robot_radius
    arrow_len = 1.5 * robot_radius

    # Set up matplotlib figure
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: 2D visualization
    ax1.set_xlim(-room_half, room_half)
    ax1.set_ylim(-room_half, room_half)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Two-Robot Meetup - Holonomic (PPO Policy)")
    
    # Draw room boundary
    rect = plt.Rectangle(
        (-room_half, -room_half),
        2 * room_half,
        2 * room_half,
        fill=False,
        linewidth=2,
    )
    ax1.add_patch(rect)

    # Robot 1 (blue)
    circle1 = Circle(
        (0.0, 0.0),
        radius=robot_radius,
        fill=False,
        linewidth=2,
        edgecolor="blue",
    )
    arrow1 = FancyArrowPatch(
        (0.0, 0.0), (robot_radius, 0.0),
        mutation_scale=15,
        linewidth=1.5,
        color="blue",
    )
    ax1.add_patch(circle1)
    ax1.add_patch(arrow1)

    # Robot 2 (red)
    circle2 = Circle(
        (0.0, 0.0),
        radius=robot_radius,
        fill=False,
        linewidth=2,
        edgecolor="red",
    )
    arrow2 = FancyArrowPatch(
        (0.0, 0.0), (robot_radius, 0.0),
        mutation_scale=15,
        linewidth=1.5,
        color="red",
    )
    ax1.add_patch(circle2)
    ax1.add_patch(arrow2)

    # Target zone visualization
    target_circle = Circle(
        (0.0, 0.0),
        radius=env.target_dist,
        fill=False,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
        color="green",
    )
    ax1.add_patch(target_circle)

    # Link between robots
    link_plot, = ax1.plot([], [], "--", linewidth=1, color="gray")

    # Right plot: metrics over time
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Value")
    ax2.set_title("Episode Metrics")
    ax2.grid(True, alpha=0.3)

    fig.canvas.draw()
    fig.show()

    # Statistics tracking
    episode_successes = 0
    episode_distances = []
    episode_orient_errors = []
    episode_final_distances = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_success = False
        
        # Metrics for this episode
        distances = []
        orient_errs = []
        times = []

        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            # Action is now 6D: [vx1, vy1, w1, vx2, vy2, w2]
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # Extract state for visualization
            (x1, y1, theta1, vx1, vy1, omega1,
             x2, y2, theta2, vx2, vy2, omega2) = env.state

            dist = info.get("dist", 0.0)
            orient_err = max(abs(info.get("orient_err1", 0.0)), 
                           abs(info.get("orient_err2", 0.0)))
            
            distances.append(dist)
            orient_errs.append(orient_err)
            times.append(t)

            # Update visualization
            circle1.center = (x1, y1)
            x1_head = x1 + arrow_len * np.cos(theta1)
            y1_head = y1 + arrow_len * np.sin(theta1)
            arrow1.set_positions((x1, y1), (x1_head, y1_head))

            circle2.center = (x2, y2)
            x2_head = x2 + arrow_len * np.cos(theta2)
            y2_head = y2 + arrow_len * np.sin(theta2)
            arrow2.set_positions((x2, y2), (x2_head, y2_head))

            link_plot.set_data([x1, x2], [y1, y2])

            # Update target circle position
            target_circle.center = ((x1 + x2) / 2, (y1 + y2) / 2)

            # Update metrics plot
            ax2.clear()
            ax2.plot(times, distances, label="Distance", linewidth=2, color="blue")
            ax2.axhline(y=env.target_dist, color="green", linestyle="--", 
                       label=f"Target dist ({env.target_dist:.2f}m)", linewidth=1.5)
            ax2.axhline(y=env.target_dist - env.target_tol, color="green", linestyle=":", alpha=0.5)
            ax2.axhline(y=env.target_dist + env.target_tol, color="green", linestyle=":", alpha=0.5)
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Distance (m)")
            ax2.set_title(f"Episode {ep + 1} | Step {t} | Distance: {dist:.3f}m")
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 10])

            # Display action magnitudes for holonomic robots
            lin_speed_1 = np.hypot(vx1, vy1)
            lin_speed_2 = np.hypot(vx2, vy2)
            
            ax1.set_title(
                f"Episode {ep + 1}/{n_episodes} | Step {t}/{max_steps} | "
                f"dist={dist:.3f}m | v1={lin_speed_1:.2f} v2={lin_speed_2:.2f} | "
                f"success={info.get('success_count', 0)}/5"
            )

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

            if terminated or truncated:
                ep_success = info.get("success", False)
                reason = "success" if ep_success else "timeout/wall"
                print(
                    f"[EP {ep + 1}/{n_episodes}] "
                    f"Finished at step {t} | "
                    f"Return: {ep_reward:>8.2f} | "
                    f"Final distance: {dist:.3f}m | "
                    f"Success: {ep_success} ({reason})"
                )
                break

        if ep_success:
            episode_successes += 1
        
        if distances:
            episode_distances.append(np.min(distances))
            episode_final_distances.append(distances[-1])
            episode_orient_errors.append(np.mean(orient_errs))

    # Print summary
    print("\n" + "=" * 80)
    print("ROLLOUT SUMMARY (HOLONOMIC ROBOTS)")
    print("=" * 80)
    print(f"Episodes completed: {n_episodes}")
    print(f"Successes: {episode_successes}/{n_episodes} ({episode_successes/n_episodes*100:.1f}%)")
    if episode_distances:
        print(f"Avg closest distance: {np.mean(episode_distances):.3f}m "
              f"(±{np.std(episode_distances):.3f}m)")
    if episode_final_distances:
        print(f"Avg final distance: {np.mean(episode_final_distances):.3f}m "
              f"(±{np.std(episode_final_distances):.3f}m)")
    if episode_orient_errors:
        print(f"Avg orientation error: {np.mean(episode_orient_errors):.3f}rad "
              f"(±{np.std(episode_orient_errors):.3f}rad)")
    print("=" * 80 + "\n")

    plt.ioff()
    plt.show()
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to PPO checkpoint (e.g., logs/.../ppo_two_robot_phase2_final.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to visualize.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Curriculum phase to evaluate (0=approach, 1=orient, 2=meetup).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    rollout(
        args.model_path,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        curriculum_phase=args.phase,
    )


if __name__ == "__main__":
    main()
