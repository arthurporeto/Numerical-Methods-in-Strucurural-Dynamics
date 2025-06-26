import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
ex3_dir = os.path.join(parent_dir, 'EX3')
sys.path.append(ex3_dir)
from Exercise_3_tools import *
import numpy as np
import scipy.linalg

#Getting the structure from previous exercise
area_cross_section = {1: 10, 2: 10, 3: 10,4:10,5:10,6:10,7:10,8:10,9:10,10:10,11:10,12:10}
length1 = 400
length2 = 300
moment_of_area = {1: 100, 2: 100, 3: 100,4:100,5:100,6:100,7:100,8:100,9:100,10:100,11:100,12:100}
Youngs_modulus = {1: 100000, 2: 100000, 3: 100000,4:100000,5:100000,6:100000,7:100000,8:100000,9:100000,10:100000,11:100000,12:100000}
spring_stiffness = {'spring_1':10000}
linear_density = {1: 1e-3, 2: 1e-3, 3: 1e-3,4:1e-3,5:1e-3,6:1e-3,7:1e-3,8:1e-3,9:1e-3,10:1e-3,11:1e-3,12:1e-3}
Nodes = {
    1: (0, 0),
    2: (length1, 0),
    3: (length1, length2),
    4: (2 * length1, 0),
    5: (2 * length1, 2 * length2), 
    6: (3 * length1, 0),
    7: (3 * length1, length2),
    8: (4 * length1, 0)
}
LINES = {
    1: (1, 3),
    2: (2, 3),
    3: (3, 4),
    4: (2, 4),
    5: (3, 5),
    6: (4, 5),
    7: (4, 6),
    8: (4, 7),
    9: (5, 7),
    10: (6, 7),
    11: (6, 8),
    12: (7, 8),
    'spring_1': (1, 2)
}
degrees_of_freedom = {
    1: (0, 1, 2, 6, 7, 8),
    2: (3, 4, 5, 6, 7, 24),
    3: (6, 7, 25, 9, 10, 11),
    4: (3, 4, 5, 9, 10, 11),
    5: (6, 7, 26, 12, 13, 14),
    6: (9, 10, 11, 12, 13, 14),
    7: (9, 10, 11, 15, 16, 17),
    8: (9, 10, 11, 18, 19, 20),
    9: (12, 13, 14, 18, 19, 20), # This appears to be a duplicate of element 8's mapping.
                            # Double-check if this is intentional or a copy-paste error.
    10: (15, 16, 17, 18, 19, 20),
    11: (15, 16, 17, 21, 22, 23),
    12: (18, 19, 20, 21, 22, 27),
    'spring_1': (0, 3)}

structure = Struss_Structure(nodes = Nodes,lines = LINES,DOF=degrees_of_freedom, constrained_dof=np.array([0,1,22]),
    area_cross_section=area_cross_section,moment_of_area=moment_of_area,Youngs_modulus=Youngs_modulus,springs_stiffness=spring_stiffness,linear_density=linear_density,etol=1e-5,dtol=1e-5)

K_total, M_total = structure.assembly()

#we want to solve Ku=f
def solve_lu_scipy(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves a linear system Ax = b using LU decomposition.

    Args:
        A (np.ndarray): The square coefficient matrix (NxN).
        b (np.ndarray): The right-hand side vector (Nx1 or 1D array).

    Returns:
        np.ndarray: The solution vector x.

    Raises:
        ValueError: If A is not square or if dimensions of A and b do not match.
        np.linalg.LinAlgError: If the matrix A is singular.
    """
    # 1. Input Validation
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimensions of A and b must match.")
    
    n = A.shape[0]

    # 2. Perform LU decomposition with pivoting: PA = LU
    # P is the permutation matrix
    # L is the lower triangular matrix (with 1s on the diagonal)
    # U is the upper triangular matrix
    P, L, U = scipy.linalg.lu(A)

    # 3. Apply permutation to b: b_prime = P @ b
    # This effectively reorders b according to the row exchanges made by P
    b_prime = P @ b

    # 4. Forward substitution: Solve Ly = b_prime for y
    # solve_triangular is efficient for triangular systems
    y = scipy.linalg.solve_triangular(L, b_prime, lower=True)

    # 5. Backward substitution: Solve Ux = y for x
    x = scipy.linalg.solve_triangular(U, y, lower=False)

    return x



def solve_static(pressure_load, #constant force per unit length acting in the transverse direction of member 12.
                 force_load2, #concentrated force acting downward on node 5
                 force_load3): #return two arrays: displacement and reaction forces
    n = np.shape(K_total[0])
    f = np.zeros((28,1))
    f[13] =force_load2
    f[7] = force_load3
    # Get coordinates for Node 7 and Node 8
    # Node 7 coordinates from your nodes dict: (3*L1_value, L2_value)
    # Node 8 coordinates from your nodes dict: (4*L1_value, 0)
    # Using the 'nodes' dictionary passed to the function:
    node7_coords = np.array(Nodes[7])
    node8_coords = np.array(Nodes[8])

    # Calculate element length (L_e) and angle (alpha)
    delta_x = node8_coords[0] - node7_coords[0]
    delta_y = node8_coords[1] - node7_coords[1]
    L_element12 = np.linalg.norm(node8_coords - node7_coords) # Inclined length of element 12
    
    if L_element12 == 0:
        # Handle zero length elements to avoid division by zero
        # In a real scenario, this would indicate a structural error.
        print("Warning: Element 12 has zero length, distributed load will not be applied.")
        return f

    # Angle of element 12 with the positive global x-axis
    angle_element12 = np.arctan2(delta_y, delta_x)

    # Assume q_magnitude is the uniform load perpendicular to the element, per unit length
    w_local_perpendicular = pressure_load

    # Calculate Local Fixed-End Forces (FEMs) for a uniformly loaded beam
    # Local y-direction forces (shear):
    F_y1_local = w_local_perpendicular * L_element12 / 2
    F_y2_local = w_local_perpendicular * L_element12 / 2

    # Local z-direction moments (bending moments):
    M_z1_local = w_local_perpendicular * (L_element12**2) / 12  # At first node (Node 7), typically counter-clockwise (+)
    M_z2_local = -w_local_perpendicular * (L_element12**2) / 12 # At second node (Node 8), typically clockwise (-)

    # Create the Local Fixed-End Force Vector for the 6 DOFs of a beam element
    # Order: (Fx1', Fy1', Mz1', Fx2', Fy2', Mz2') in local coordinates
    local_fixed_end_forces_vector = np.array([
        0,          # Fx1_local (axial force is zero for purely perpendicular load)
        F_y1_local, # Fy1_local
        M_z1_local, # Mz1_local
        0,          # Fx2_local
        F_y2_local, # Fy2_local
        M_z2_local  # Mz2_local
    ])

    # Create the 6x6 Transformation Matrix (T_element) for a 2D Frame Element
    # This matrix relates global displacements/forces to local ones.
    # d_local = T_element @ d_global
    # F_global = T_element.T @ F_local
    cos_a = np.cos(angle_element12)
    sin_a = np.sin(angle_element12)

    T_element = np.array([
        [cos_a, sin_a, 0, 0, 0, 0],
        [-sin_a, cos_a, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, cos_a, sin_a, 0],
        [0, 0, 0, -sin_a, cos_a, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    # Transform local fixed-end forces to global coordinates
    global_fixed_end_forces = T_element.T @ local_fixed_end_forces_vector

    # --- 3. Add these Global Fixed-End Forces to the main 'f' vector ---
    # Global DOFs for Element 12 are: q18 (Node 7 ux), q19 (Node 7 uy), q20 (Node 7 rz)
    #                                  q21 (Node 8 ux), q22 (Node 8 uy), q27 (Node 8 rz)
    # The 'global_fixed_end_forces' vector is already ordered (Fx1, Fy1, Mz1, Fx2, Fy2, Mz2)
    # in global coordinates for these DOFs.

    f[18] += global_fixed_end_forces[0] # Node 7, ux
    f[19] += global_fixed_end_forces[1] # Node 7, uy
    f[20] += global_fixed_end_forces[2] # Node 7, rz

    f[21] += global_fixed_end_forces[3] # Node 8, ux
    f[22] += global_fixed_end_forces[4] # Node 8, uy
    f[27] += global_fixed_end_forces[5] # Node 8, rz

    f = np.delete(f,[0,1,22])
    #np.delete(f,1)
    #np.delete(f,22)

    u = solve_lu_scipy(K_total,f)

    return  u #displacements = solve_lu_scipy(K,f) #displacements
    #reactions = np.zeros(3,1)



u = solve_static(pressure_load=0.01,force_load2=-60,force_load3=-40)
print(u)   