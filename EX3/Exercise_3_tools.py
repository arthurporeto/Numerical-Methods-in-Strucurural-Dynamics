import numpy as np

def generate_mass_matrix(f, length, linear_density, etol):
    # Calculate the width of each trapezoid
    n = int(round(1/etol)) #number of trapezoid
    h = (length - 0) / n

    # Generate the x-values for each point
    # x_values = [a + i * h for i in range(n + 1)] # List comprehension way
    x_values = np.linspace(0, length, n + 1) # More efficient NumPy way

    # Evaluate the function at each x-value
    y_values = [np.outer(f(x),f(x)) for x in x_values]

    # Apply the trapezoidal rule formula:
    # Integral approx = (h/2) * [f(x0) + 2f(x1) + 2f(x2) + ... + 2f(xn-1) + f(xn)]
    integral_sum = y_values[0] + y_values[-1]  # Add the first and last terms

    # Add the middle terms (multiplied by 2)
    for i in range(1, n):
        integral_sum += 2 * y_values[i]

    integral_approximation = (h / 2) * integral_sum

    return integral_approximation*linear_density


def generate_stiffness_matrix(f,length,axial_stiffness,dtol,etol):
    # Calculate the width of each trapezoid
    n = int(round(1/etol)) #number of trapezoid
    h = (length - 0) / n

    # Generate the x-values for each point
    # x_values = [a + i * h for i in range(n + 1)] # List comprehension way
    x_values = np.linspace(0, length, n + 1) # More efficient NumPy way

    # Evaluate the function at each x-value
    y_values = [np.outer((f(x+dtol)-f(x-dtol))/(2*dtol),(f(x+dtol)-f(x-dtol))/(2*dtol)) for x in x_values]

    # Apply the trapezoidal rule formula:
    # Integral approx = (h/2) * [f(x0) + 2f(x1) + 2f(x2) + ... + 2f(xn-1) + f(xn)]
    integral_sum = y_values[0] + y_values[-1]  # Add the first and last terms

    # Add the middle terms (multiplied by 2)
    for i in range(1, n):
        integral_sum += 2 * y_values[i]

    integral_approximation = (h / 2) * integral_sum

    return integral_approximation*axial_stiffness*length

"""code here"""
def delete_degrees_of_freedom_from_matrix(matrix,deleted_degrees_of_freedom):
    if np.max(deleted_degrees_of_freedom) <= np.shape(matrix)[0]:
        matrix = np.delete(matrix,deleted_degrees_of_freedom,axis=0) #excluding rows
        matrix = np.delete(matrix,deleted_degrees_of_freedom,axis=1) #excluding columns
        return(matrix)
    else:
        print('Invalid indices provided')
def rotation_matrix_2d(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

def transformed_matrix(original_matrix,transformation_matrix):
    return transformation_matrix.T @ original_matrix @ transformation_matrix

def reorder_matrix(original_matrix,reordering_of_the_degrees_of_freedom):
    if original_matrix.shape[0] != original_matrix.shape[1]:
        raise ValueError("The input matrix must be square.")
    
    if sorted(reordering_of_the_degrees_of_freedom) != list(range(original_matrix.shape[0])):
        raise ValueError("index_vector must be a permutation of row/column indices.")
    
    return original_matrix[np.ix_(reordering_of_the_degrees_of_freedom, reordering_of_the_degrees_of_freedom)]

"""code here"""

def derivative(f,x,tol):
    return (f(x+tol)-f(x-tol))/(2*tol)

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




class BarBeamElement(object):
    def __init__(self,
                coordinates_first_end: tuple[float, float], # (x1, y1)
                coordinates_second_end: tuple[float, float], # (x2, y2)
                linear_density: float, # rho*A
                axial_stiffness: float, # E*A/L
                transverse_stiffness: float): #E*I/L
        
        """
        The degrees of freedom are the displacements of the nodes in the global frame of reference
        The order of the degrees of freedom is
        x_I, y_I, theta_I, x_II, y_II, theta_II
        where I refers to the first end and II to the second end
        the x and y are the horizontal and vertical displacements in the global frame of reference
        """
        pass

    def generate_structural_mass_matrix(self):
        """
        Outputs the 6x6 mass matrix of the Bar+Beam element.
        """
        pass
        
    def generate_structural_stiffness_matrix(self):
        """
        Outputs the 6x6 stiffness matrix of the Bar+Beam element.
        """
        pass
