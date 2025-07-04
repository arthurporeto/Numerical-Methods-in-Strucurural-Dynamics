import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg 

##EXERCISE 1 - BC AND MATRICES CALCULATIONS

def is_dirichlet_compatible(fun, x_0, f_0, etol):
    """
    Verifies if a single-variable function satisfies a given Dirichlet boundary condition at a point.

    Parameters:
        fun (callable): A function taking a single float argument.
        x_0 (float): Boundary point to evaluate.
        f_0 (float): Expected function value at the boundary point.
        etol (float): Tolerance for equality check.

    Returns:
        bool: True if the function satisfies the Dirichlet condition at x within tolerance, False otherwise.
    """
    return np.abs(fun(x_0) - f_0) < etol

def is_neumann_compatible(fun, x_0, dfdx_0, dtol, etol):
    """
    Verifies if a single-variable function satisfies a given Neumann boundary condition at a point.

    Parameters:
        fun (callable): A function taking a single float argument.
        x_0 (float): Boundary point to evaluate.
        dfdx_0 (float): Expected derivative value at the boundary point.
        dtol (float): Tolerance for derivative calculation (dx). Assume dx != 0.
        etol (float): Tolerance for equality check.

    Returns:
        bool: True if the function satisfies the Neumann condition at x within tolerance, False otherwise.
    """
    # Approximate derivative using central difference
    # (f(x + dx) - f(x - dx)) / (2 * dx)
    # In this case, dtol is our dx
    derivative_approx = (fun(x_0 + dtol) - fun(x_0 - dtol)) / (2 * dtol)
    return np.abs(derivative_approx - dfdx_0) < etol    

def generate_mass_matrix(f, length, linear_density, etol):
    # Calculate the width of each trapezoid
    n = int(round(1/etol)) #number of trapezoid
    h = (length - 0) / n

    # Generate the x-values for each point
    # x_values = [a + i * h for i in range(n + 1)] # List comprehension way
    x_values = np.linspace(0, length, n + 1) # More efficient NumPy way

    # Evaluate the function at each x-value
    y_values = [np.outer(f(x,length),f(x,length)) for x in x_values]

    # Apply the trapezoidal rule formula:
    # Integral approx = (h/2) * [f(x0) + 2f(x1) + 2f(x2) + ... + 2f(xn-1) + f(xn)]
    integral_sum = y_values[0] + y_values[-1]  # Add the first and last terms

    # Add the middle terms (multiplied by 2)
    for i in range(1, n):
        integral_sum += 2 * y_values[i]

    integral_approximation = (h / 2) * integral_sum

    return integral_approximation*linear_density

'''
def generate_stiffness_matrix(f,length,axial_stiffness,dtol,etol):
    # Calculate the width of each trapezoid
    n = int(round(1/etol)) #number of trapezoid
    h = (length - 0) / n

    # Generate the x-values for each point
    # x_values = [a + i * h for i in range(n + 1)] # List comprehension way
    x_values = np.linspace(0, length, n + 1) # More efficient NumPy way

    # Evaluate the function at each x-value
    y_values = [np.outer((f(x+dtol,length)-f(x-dtol,length))/(2*dtol),(f(x+dtol,length)-f(x-dtol,length))/(2*dtol)) for x in x_values]

    # Apply the trapezoidal rule formula:
    # Integral approx = (h/2) * [f(x0) + 2f(x1) + 2f(x2) + ... + 2f(xn-1) + f(xn)]
    integral_sum = y_values[0] + y_values[-1]  # Add the first and last terms

    # Add the middle terms (multiplied by 2)
    for i in range(1, n):
        integral_sum += 2 * y_values[i]

    integral_approximation = (h / 2) * integral_sum

    return integral_approximation*axial_stiffness*length
'''

def generate_stiffness_matrix(f,length,axial_stiffness,transverse_stiffness,dtol,etol):
    # Calculate the width of each trapezoid
    n = int(round(1/etol)) #number of trapezoid
    h = (length - 0) / n
    f_length = lambda coord: f(coord, length=length)

    # Generate the x-values for each point
    # x_values = [a + i * h for i in range(n + 1)] # List comprehension way
    x_values = np.linspace(0, length, n + 1) # More efficient NumPy way

    # Evaluate the function at each x-value
    if axial_stiffness != 0:
        y_values = [np.outer(derivative(f_length,x,dtol),derivative(f_length,x,dtol)) for x in x_values]
    #y_values_Bv = [np.outer(derivative_beam_shape_functions_num(f,x,dtol),derivative_beam_shape_functions_num(f,x,dtol)) for x in x_values]

    # Apply the trapezoidal rule formula:
    # Integral approx = (h/2) * [f(x0) + 2f(x1) + 2f(x2) + ... + 2f(xn-1) + f(xn)]
        integral_sum_Bu = y_values[0] + y_values[-1]  # Add the first and last terms
    #integral_sum_Bv = y_values_Bv[0] + y_values_Bv[-1]  # Add the first and last terms

    # Add the middle terms (multiplied by 2)
        for i in range(1, n):
            integral_sum_Bu += 2 * y_values[i]
        #integral_sum_Bv += 2 * y_values_Bv[i]


        integral_approximation = (h / 2) * integral_sum_Bu *axial_stiffness *length
   # integral_approximation += (h/2) * integral_sum_Bv * transverse_stiffness * length**2

        return integral_approximation
    else:
        y_values = [np.outer(derivative_beam_shape_functions_num(f_length,x,dtol),derivative_beam_shape_functions_num(f_length,x,dtol)) for x in x_values]
        integral_sum_Bu = y_values[0] + y_values[-1]  # Add the first and last terms
        for i in range(1, n):
            integral_sum_Bu += 2 * y_values[i]
        #integral_sum_Bv += 2 * y_values_Bv[i]


        integral_approximation = (h / 2) * integral_sum_Bu *transverse_stiffness *length
        return integral_approximation

##EXERCISE 2 -- ASSEMBLY
def delete_degrees_of_freedom_from_matrix(matrix,deleted_degrees_of_freedom):
    if np.max(deleted_degrees_of_freedom) <= np.shape(matrix)[0]:
        matrix = np.delete(matrix,deleted_degrees_of_freedom,axis=0) #excluding rows
        matrix = np.delete(matrix,deleted_degrees_of_freedom,axis=1) #excluding columns
        return(matrix)
    else:
        print('Invalid indices provided')

def derivative(f,x,dtol):
    return (f(x+dtol)-f(x-dtol))/(2*dtol)


def rotation_matrix_bar_beam_2d(angle_radians: float) -> np.ndarray:
    """
    Generates the 6x6 rotation (transformation) matrix for a 2D bar-beam element.
    This matrix transforms global displacements/forces to local displacements/forces
    (or vice-versa when transposed).

    Args:
        angle_radians (float): The angle (in radians) of the element's local x'-axis
                                (from node 1 to node 2) with respect to the global X-axis.

    Returns:
        np.ndarray: A 6x6 NumPy array representing the rotation matrix.
                    Order of DOFs: (ux1, uy1, rz1, ux2, uy2, rz2)
    """
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)

    # Initialize a 6x6 zero matrix
    R = np.zeros((6, 6))

    # Fill the blocks for translational DOFs
    R[0, 0] = cos_a
    R[0, 1] = sin_a
    R[1, 0] = -sin_a
    R[1, 1] = cos_a

    R[3, 3] = cos_a
    R[3, 4] = sin_a
    R[4, 3] = -sin_a
    R[4, 4] = cos_a

    # Fill the blocks for rotational DOFs (rotations are the same in local/global)
    R[2, 2] = 1
    R[5, 5] = 1

    return R

def transformed_matrix(original_matrix,transformation_matrix):
    return transformation_matrix.T @ original_matrix @ transformation_matrix

def reorder_matrix(original_matrix,reordering_of_the_degrees_of_freedom):
    if original_matrix.shape[0] != original_matrix.shape[1]:
        raise ValueError("The input matrix must be square.")
    
    if sorted(reordering_of_the_degrees_of_freedom) != list(range(original_matrix.shape[0])):
        raise ValueError("index_vector must be a permutation of row/column indices.")
    
    return original_matrix[np.ix_(reordering_of_the_degrees_of_freedom, reordering_of_the_degrees_of_freedom)]

from typing import Union

def insert_element(global_matrix: Union[np.ndarray, list],
                   element_matrix: Union[np.ndarray, list],
                   localization_array: Union[np.ndarray, list]) -> np.ndarray:
    """
    Inserts a smaller square matrix into specific positions of a larger square matrix
    using element-wise addition, without modifying the original global matrix.

    Parameters:
    - global_matrix: 2D NumPy array (N x N), the base matrix.
    - element_matrix: 2D NumPy array (n x n), values to insert via addition.
    - localization_array: 1D NumPy array (length n), indices in the global matrix where
                          rows and columns of element_matrix are to be added.

    Returns:
    - A new NumPy array of shape (N x N) with values from element_matrix added at the
      positions specified by localization_array.

    Raises:
    - ValueError: If input shapes are inconsistent or if matrices are not square.
    """
    # Convert inputs to NumPy arrays
    global_matrix = np.asarray(global_matrix)
    element_matrix = np.asarray(element_matrix)
    localization_array = np.asarray(localization_array)

    # Check matrix shapes
    N = global_matrix.shape[0]
    if global_matrix.shape[0] != global_matrix.shape[1]:
        raise ValueError("global_matrix must be square.")
    if element_matrix.shape[0] != element_matrix.shape[1]:
        raise ValueError("element_matrix must be square.")
    if element_matrix.shape[0] != localization_array.shape[0]:
        raise ValueError("localization_array length must match element_matrix dimensions.")
    if np.any(localization_array < 0) or np.any(localization_array >= N):
        raise ValueError("localization_array contains invalid indices.")

    # Copy global matrix to avoid modifying the original
    result = global_matrix.copy()

    # Use advanced indexing with np.add.at for in-place accumulation
    row_idx = localization_array[:, None]
    col_idx = localization_array[None, :]
    np.add.at(result, (row_idx, col_idx), element_matrix)

    return result

"""code here"""
from typing import List, Tuple

def assemble_global_matrix(
    total_number_of_degrees_of_freedom: int,
    list_of_elements: List[Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    N = total_number_of_degrees_of_freedom
    global_matrix = np.zeros((N, N))

    for element_matrix, localization_array in list_of_elements:
        element_matrix = np.asarray(element_matrix)
        localization_array = np.asarray(localization_array)

        # Validate shapes
        n = element_matrix.shape[0]
        if element_matrix.shape[0] != element_matrix.shape[1]:
            raise ValueError("Element matrix must be square.")
        if localization_array.shape[0] != n:
            raise ValueError("Localization array length must match element matrix size.")
        if np.any(localization_array < 0) or np.any(localization_array >= N):
            raise ValueError("Localization array contains out-of-bounds indices.")

        # Generate index grids and accumulate
        row_idx = localization_array[:, None]
        col_idx = localization_array[None, :]
        np.add.at(global_matrix, (row_idx, col_idx), element_matrix)

    return global_matrix

def bar_shape_functions(longitudinal_coordinate,length):
        #length = 2
        xi = longitudinal_coordinate/length
        f1 = 1 - xi
        f2 = xi
        return np.array([f1, f2])

def beam_shape_functions(longitudinal_coordinate,length):
    #length = 2
    xi = longitudinal_coordinate/length
    f1 = 1 - 3*xi**2 + 2*xi**3
    f2 = length * xi * (1-xi)**2
    f3 = xi**2 * (3-2*xi)
    f4 = length * xi**2 * (xi-1)
    return np.array([f1, f2, f3, f4])

def derivative_beam_shape_functions_num(f,x,dtol):
    # or you can write the first derivatives yourself by hand
    return (f(x + dtol) - 2 * f(x) + f(x - dtol)) / (dtol**2)

def spring_matrices(spring_stiffness):
    M_spring = np.array([[0,0],[0,0]])
    K_spring = np.array([[spring_stiffness,-spring_stiffness],[-spring_stiffness,spring_stiffness]])
    return K_spring,M_spring

##EXERCISE 4 - LU DECOMPOSITION

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