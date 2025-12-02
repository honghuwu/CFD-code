from Env2DAirfoil import run_visualization_task as Airfoil
import os
from Env2DCylinder import run_visualization_task as Cylinder
import multiprocessing

if __name__ == '__main__':
    # Create processes for parallel execution
    p1 = multiprocessing.Process(target=Airfoil, kwargs={'total_steps': 50000, 'save_interval': 500, 'video_fps': 20})
    p2 = multiprocessing.Process(target=Cylinder, kwargs={'total_steps': 50000, 'save_interval': 500, 'video_fps': 20})

    print("Starting parallel simulations...")
    
    # Start processes
    p1.start()
    p2.start()

    # Wait for completion
    p1.join()
    p2.join()
    
    print("All simulations completed.")

