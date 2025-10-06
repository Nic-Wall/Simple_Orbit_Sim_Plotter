# https://likegeeks.com/3d-sphere-python/
# https://fitelson.org/NASA_memo.pdf
# https://math.stackexchange.com/questions/3372017/angle-of-visible-part-of-circle

import matplotlib.pyplot as plt
import numpy as np
import random

def main():
    # Creating the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Creating Theta and Phi for the circle/ sphere calculations
    theta = np.linspace(0, 2 * np.pi, 100)  # Equator angle
    phi = np.linspace(0, np.pi, 50)         # Polar angle
    theta, phi = np.meshgrid(theta, phi)    # Creating 2D grid of the sphere's surface

    # Plotting the Earth
    earth_r = 6.372   # Radius of the Earth if the core was exactly 0,0,0
    earth_x = earth_r * np.sin(phi) * np.cos(theta)
    earth_y = earth_r * np.sin(phi) * np.sin(theta)
    earth_z = earth_r * np.cos(phi)
    earth_center = np.array([0,0,0])
    ax.plot_surface(earth_x, earth_y, earth_z, alpha=0.6)

    sat_radia = [7.172, 61.941, 35.786] # 800km above the earth, HEO, weather satellite
    satellites = []
    for sat_r in sat_radia:
        # Plotting the polar satellite path
        # https://stackoverflow.com/questions/56870675/how-to-do-a-3d-circle-in-matplotlib
        # https://stackoverflow.com/questions/9879258/how-can-i-generate-random-points-on-a-circles-circumference-in-javascript
        # Orbit setup
        sat_orb_x = sat_r * np.cos(theta[0])    # Theta is still a meshgrid, but only one array is necessary for the circle itself, the other 49 would be copies of the first
        sat_orb_z = sat_r * np.sin(theta[0])
        sat_orb_y = np.zeros_like(theta[0]) #sat_r * np.sin(theta[0])     # Filling the y vector with 0's
        rotated_orbit = np.vstack((sat_orb_x, sat_orb_y, sat_orb_z))
        # Satellite setup
        random_theta = (random.randint(0,100)/100) * np.pi * 2  # Generating a random point on the orbit
        sat_x = sat_r * np.cos(random_theta)
        sat_z = sat_r * np.sin(random_theta)
        sat_y = 0 #sat_r * np.sin(random_theta)
        sat_center = np.array([sat_x, sat_y, sat_z])
        # Matrices for rotations in a 3D space (x, y, z)
        for rotation_matrix in range(0,3): # For each rotation matrix... I miss pointers :(
            random_angle = np.radians(np.random.uniform(0,360))      # Generate a random angle to rotate on
            if rotation_matrix == 0: # Rotate on X
                rot_matrix = np.array([[1,0,0],[0, np.cos(random_angle), -1*np.sin(random_angle)],[0, np.sin(random_angle), np.cos(random_angle)]])
            elif rotation_matrix == 1: # Rotate on Y
                rot_matrix = np.array([[np.cos(random_angle), 0, np.sin(random_angle)], [0,1,0],[-1 * np.sin(random_angle), 0, np.cos(random_angle)]])
            elif rotation_matrix == 2: # Rotate on Z
                rot_matrix = np.array([[np.cos(random_angle), -1*np.sin(random_angle), 0],[np.sin(random_angle), np.cos(random_angle), 0],[0,0,1]])
            rotated_orbit = rot_matrix @ rotated_orbit      # Rotate the orbit accordingly
            sat_center = rot_matrix.dot(sat_center) # Rotate the satellite's position on the orbit accordingly
        # Plotting the orbit
        ax.plot(rotated_orbit[0], rotated_orbit[1], rotated_orbit[2], c="red")
        # Plotting the satellite's position on the orbit
        plt.plot(sat_center[0], sat_center[1], sat_center[2], marker="o", color="orange")
        satellites.append(sat_center)      # Ensuring the satellite's current position exists in the satellites array, so it's vision of the planet can be determined

    # Determined
    # Visibility
        # https://en.wikipedia.org/wiki/Spherical_cap
        # https://stackoverflow.com/questions/45344402/plotting-spherical-caps
    """
    1. Calculate the direct distance from the satellite to the earth center (done with the following equation (linalg normalization))
        sqrt(((6.66835-0)**2)+((0-0)**2)+((2.64019-0)**2) = 7.1799
    2. Calculate the right triangle arms from the furthest visible point on the horizon, the planet's center, and the satellite
        a^2+b^2=c^2
        6.372^2+b^2=7.1799^2     means b=3.30886
    3. Calculate one of the missing angles (one will always be 90 degree). Kept in radians because of the cos/sin transformation in step 9
        sin^-1(3.30886/7.1799) = 0.478953 radians for the angle on the arm from the earth's center to the satellite and the earth's center to it's crust
    4. Normalize the distance vector
        satellite_position / distance_calculated_in_step_1
    5. Acquire the cross product of the normalized vector (acquired in step 4) and a non-parallel vector (say [0,1,0])
            i            j            k
            0            1            0            (non-parallel vector)
            0.9287       0            0.3673       (normalized vector)
    6. Acquire the normalized cross product
        cross_product / np.linalg.norm(cross_product)
    7. Acquire the cross product of the normalized distance vector and the normalized cross product
        np.cross(normalized_distance_vector, normalized_cross_product)
    8. Make a note of radian positions when rotating around a circle...
        Direction        deg        rad        tangent component
        East             0          0          norm_cross_x
        North            90         pi/2       norm_cross_y
        West             180        pi         -norm_cross_x
        South            270        3pi/2      -norm_cross_y
    9. Calculate the furthest visible point on the horizon with the following equation...
        planet_radius * (np.cos(rad_from_step3) * normalized_distance_vector + np.sin(rad_from_step3 * (np.cos(rad_from_step3) * normalized_cross_product + np.sin(rad_from_step3 * cross_prod_of_normDistVec_and_normCrossProd))))
    """
    for sat_center in satellites:
        direct_distance = np.linalg.norm(sat_center - earth_center) # 1. Calculate the direct distance
        b_arm = np.sqrt(direct_distance**2 - earth_r**2)            # 2. a arm is radius of the earth, c arm is the direct distance, so b is...
        ang_1_rad = (np.arcsin(b_arm/direct_distance))              # 3. Calculate the angles (one will always be 90 deg), this is the angle from the center to the horizon
        norm_z = sat_center / direct_distance                       # 4. Normalize the distance vector
        cross_x = np.cross(norm_z, np.array([0,1,0]))               # 5. Acquire the cross product of the normalized vector using non-parallel vector
        norm_x = cross_x / np.linalg.norm(cross_x)                  # 6. Acquire the normalized cross product
        norm_y = np.cross(norm_z, norm_x)                           # 7. Acquire the cross product of the normalized distance vector and the normalized cross product
        # 8. Plot the calculated rotations using the formula in the list above
        cardinal_dir_rads = np.array([0, np.pi/2, np.pi, (np.pi*3)/2])
        #cardinal_dir_rads = np.array([2*np.pi,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,5*np.pi/6,np.pi,7*np.pi/6,4*np.pi/3,3*np.pi/2,5*np.pi/6,11*np.pi/6,0])
        for rad in cardinal_dir_rads:
            # 9. Calculate the furthest visible point on the horizon
            horizon_point = earth_r * (np.cos(ang_1_rad) * norm_z + np.sin(ang_1_rad) * (np.cos(rad) * norm_x + np.sin(rad) * norm_y))
            plt.plot(horizon_point[0], horizon_point[1], horizon_point[2], marker="o", color="green")
            plt.plot([sat_center[0], horizon_point[0]], [sat_center[1], horizon_point[1]], [sat_center[2], horizon_point[2]], color="purple")

    # Plotting the ground stations

    # Plotting the Sun's shadow
        # https://www.pveducation.org/pvcdrom/properties-of-sunlight/declination-angle
    # The Earth stays "straight perpindicular" according to the plotting, it's the Sun's position that changes (simulating the Earth's angle and rotation)

    # Printing important coordinates
    print("EarthPos:", earth_center[0], earth_center[1], earth_center[2])
    for satellite in range(0,3):
        print("SatR", sat_radia[satellite], ":", satellites[satellite][0], " ", satellites[satellite][1], " ", satellites[satellite][2])
    # Setting the pixel per data unit to be equal (otherwise spheres appear oblong) and showing the plot in a QT window
    ax.axis("equal")    # https://stackoverflow.com/questions/9230389/why-is-matplotlib-plotting-my-circles-as-ovals
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

main()