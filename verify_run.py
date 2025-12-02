from Env2DAirfoil import run_visualization_task
import os

print("Starting verification run...")
# Run for 1000 steps (0.1s), saving every 500 steps (0.05s). 
# This should produce 2 frames for each field.
try:
    run_visualization_task(total_steps=50000, save_interval=500,video_fps = 20)
    print("Verification run completed successfully.")
except Exception as e:
    print(f"Verification run failed: {e}")
