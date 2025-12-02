import numpy as np
import matplotlib.pyplot as plt

def naca0012(x):
    """
    Calculate the half-thickness of NACA 0012 airfoil at position x.
    """
    t = 0.12
    # NACA 4-digit equation
    # y_t = 5 * t * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x^2 + 0.2843 * x^3 - 0.1015 * x^4)
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

def generate_naca0012_points(num_points=100):
    """
    Generate x, y coordinates for NACA 0012 airfoil.
    """
    # Use cosine spacing for better resolution at LE and TE
    beta = np.linspace(0, np.pi, num_points)
    x = (1 - np.cos(beta)) / 2
    yt = naca0012(x)
    
    # Upper surface
    xu = x
    yu = yt
    
    # Lower surface
    xl = x
    yl = -yt
    
    # Combine into a single array from TE (upper) to LE to TE (lower)
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    
    return x_coords, y_coords

def rotate_points(x, y, angle_deg):
    """
    Rotate points by angle_deg degrees.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    # Standard rotation matrix (Counter-Clockwise)
    x_rot = x * c - y * s
    y_rot = x * s + y * c
    
    return x_rot, y_rot

def main():
    # Generate points
    x, y = generate_naca0012_points(200)
    
    # Rotate by -25 degrees (Clockwise) to simulate 25 deg Angle of Attack (Nose Up)
    # Assuming flow is horizontal left-to-right, and LE is pivot at (0,0).
    # Pitching up means TE moves down.
    angle = -25
    x_rot, y_rot = rotate_points(x, y, angle)

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot rotated airfoil
    plt.plot(x_rot, y_rot, 'r-', linewidth=2, label=f'NACA 0012 (AoA = {-angle}°)')
    plt.fill(x_rot, y_rot, 'r', alpha=0.1)
    
    # Plot original airfoil for reference
    plt.plot(x, y, 'k--', linewidth=1, alpha=0.5, label='NACA 0012 (AoA = 0°)')
    
    # Draw flow direction arrow (Horizontal)
    plt.arrow(-0.2, 0, 0.1, 0, head_width=0.02, head_length=0.03, fc='b', ec='b', label='Flow Direction')
    
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.title(f'NACA 0012 Airfoil at {-angle}° Angle of Attack')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Save and show
    plt.savefig('naca0012_25deg.png', dpi=300)
    print("Image saved to naca0012_25deg.png")
    plt.show()

if __name__ == "__main__":
    main()
