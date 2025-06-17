import math

# Constants
g = 9.8  # Acceleration due to gravity (m/s^2)

# Get inputs from the user
V = float(input("Enter the initial velocity (V) in m/s: "))
theta_degrees = float(input("Enter the launch angle (theta) in degrees: "))
m = float(input("Enter the mass (m) of the projectile in kg: "))

# Convert angle to radians
theta_radians = math.radians(theta_degrees)

# Calculate initial velocities
Vy = V * math.sin(theta_radians)
Vx = V * math.cos(theta_radians)

# Calculate time to reach maximum height
t = Vy / g

# Calculate maximum height
h = Vy * t - 0.5 * g * t**2

# Calculate total horizontal range
total_time = 2 * t
R = Vx * total_time

# Calculate kinetic energy at maximum height
KE_max = 0.5 * m * Vx**2

# Calculate potential energy at maximum height
PE_max = m * g * h

# Print the results
print("\n--- Results ---")
print(f"Initial vertical velocity (Vy): {Vy:.2f} m/s")
print(f"Initial horizontal velocity (Vx): {Vx:.2f} m/s")
print(f"Time to reach maximum height (t): {t:.2f} s")
print(f"Maximum height (h): {h:.2f} m")
print(f"Total horizontal range (R): {R:.2f} m")
print(f"Kinetic energy at maximum height (KE_max): {KE_max:.2f} J")
print(f"Potential energy at maximum height (PE_max): {PE_max:.2f} J")