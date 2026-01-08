All of these files go into:

<IsaacLab Directory>/scripts/reinforcement_learning/custom/.

Commands:

Low-Level Training:
./isaaclab.sh -p <IsaacLab Directory>/scripts/reinforcement_learning/custom/train_ppo_g1.py   --task Isaac-Velocity-Flat-OneG1-v0   --device cuda:0   --phase walk  --num_envs 256 --load_checkpoint <Low-Level Checkpoint Directory>/ppo_walk_final.zip   --vecnorm_path <Low-Level Vecnorm Directory>/vecnormalize_final.pkl  --total_timesteps 1000000

Low-Level Demo:
./isaaclab.sh -p <IsaacLab Directory>/scripts/reinforcement_learning/custom/play_ppo_g1.py   --task Isaac-Velocity-Flat-OneG1-v0   --device cuda:0   --phase walk  --num_envs 4 --checkpoint <Low-Level Checkpoint Directory>/ppo_walk_final.zip --max_steps 5000

High-level Training:
python3 <IsaacLab Directory>/scripts/reinforcement_learning/custom/train_hl_robot_meetup.py --phase 2 --phase2_steps 400000

High-level Demo:
python3 <IsaacLab Directory>/scripts/reinforcement_learning/custom/play_hl_robot_meetup.py   --model_path <High-Level Checkpoint Directory>/ppo_two_robot_phase2_final.zip   --episodes 10   --phase 2   --max_steps 300

Full demo:
./isaaclab.sh -p <IsaacLab Directory>/scripts/reinforcement_learning/custom/play_two_g1_meetup.py   --task Isaac-Velocity-Flat-OneG1-v0   --device cuda:0   --g1_checkpoint <Low-Level Checkpoint Directory>/ppo_walk_final.zip   --g1_vecnorm <Low-Level Vecnorm Directory>/vecnormalize_final.pkl   --meetup_checkpoint <High-Level Checkpoint Directory>/ppo_two_robot_phase2_final.zip   --max_steps 5000



