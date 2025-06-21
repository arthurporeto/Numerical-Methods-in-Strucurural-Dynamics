import numpy as np

def generate_mass_matrix(f, length, linear_density, etol):
    # Calculate the width of each trapezoid
    n = int(round(1/etol)) #number of trapezoid
    h = (length - 0) / n
    f_length = lambda coord: f(coord, length=length)
    # Generate the x-values for each point
    # x_values = [a + i * h for i in range(n + 1)] # List comprehension way
    x_values = np.linspace(0, length, n + 1) # More efficient NumPy way

    # Evaluate the function at each x-value
    y_values = [np.outer(f_length(x),f_length(x)) for x in x_values]

    # Apply the trapezoidal rule formula:
    # Integral approx = (h/2) * [f(x0) + 2f(x1) + 2f(x2) + ... + 2f(xn-1) + f(xn)]
    integral_sum = y_values[0] + y_values[-1]  # Add the first and last terms

    # Add the middle terms (multiplied by 2)
    for i in range(1, n):
        integral_sum += 2 * y_values[i]

    integral_approximation = (h / 2) * integral_sum

    return integral_approximation*linear_density


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

def derivative(f,x,dtol):
    return (f(x+dtol)-f(x-dtol))/(2*dtol)

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
        #print(f'n:{n}')
        #print(f'N:{N}')
        #print(f'localization_array:{localization_array}')
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




class BarBeamElement(object):
    def __init__(self,
                coordinates_first_end: tuple[float, float], # (x1, y1)
                coordinates_second_end: tuple[float, float], # (x2, y2)
                linear_density: float, # rho*A
                axial_stiffness: float, # E*A/L
                transverse_stiffness: float,
                etol: float,
                dtol: float): #E*I/L
        
        """
        The degrees of freedom are the displacements of the nodes in the global frame of reference
        The order of the degrees of freedom is
        x_I, y_I, theta_I, x_II, y_II, theta_II
        where I refers to the first end and II to the second end
        the x and y are the horizontal and vertical displacements in the global frame of reference
        """
        self.coordinates_first_end = np.array(coordinates_first_end)
        self.coordinates_second_end = np.array(coordinates_second_end)
        self.linear_density = linear_density
        self.axial_stiffness = axial_stiffness
        self.transverse_stiffness = transverse_stiffness
        self.length = np.linalg.norm(self.coordinates_second_end-self.coordinates_first_end)
        self.etol = etol
        self.dtol = dtol

    def generate_structural_mass_matrix(self):
        """
        Outputs the 6x6 mass matrix of the Bar+Beam element.
        """
        M_bar = generate_mass_matrix(bar_shape_functions,self.length,self.linear_density,self.etol)
        M_beam = generate_mass_matrix(beam_shape_functions, self.length, self.linear_density, self.etol)
        self.structural_mass_matrix = np.zeros((6,6))
        loc_array_bar = np.array([0,3])
        loc_array_beam = np.array([1,2,4,5])
        self.structural_mass_matrix = assemble_global_matrix(6,[(M_bar,loc_array_bar),(M_beam,loc_array_beam)])
        return self.structural_mass_matrix
    
    def generate_structural_stiffness_matrix(self):
        """
        Outputs the 6x6 stiffness matrix of the Bar+Beam element.
        """
    
        K_bar = generate_stiffness_matrix(bar_shape_functions,self.length,self.axial_stiffness,0,self.dtol,self.etol)
        K_beam = generate_stiffness_matrix(beam_shape_functions, self.length, 0, self.transverse_stiffness,self.dtol, self.etol)
        self.structural_stiffness_matrix = np.zeros((6,6))
        loc_array_bar = np.array([0,3])
        loc_array_beam = np.array([1,2,4,5])
        self.structural_stiffness_matrix = assemble_global_matrix(6,[(K_bar,loc_array_bar),(K_beam,loc_array_beam)])
        return self.structural_stiffness_matrix
    

    
    #K_global = np.zeros((total_DOF,total_DOF))


class Struss_Structure(object):
    def __init__(self,
                 nodes,
                 lines,
                 DOF,
                 constrained_dof,
                 area_cross_section,
                 moment_of_area,
                 Youngs_modulus,
                 linear_density, #rho*A
                 springs_stiffness, #{spring_1: spring_stiffness}
                 etol: float,
                 dtol: float):
        self.nodes = nodes #nodes = {1: (0,0), 2: (length,0)}
        self.lines = lines #lines = {1: (1,3),'spring': (1,2)}
        self.DOF = DOF
        self.constrained_dof = constrained_dof
        self.linear_density = linear_density # density = {1: float, spring: 0}
        self.area_cross_section = area_cross_section # area_cross_section = {1: float, spring: 0}
        self.moment_of_area = moment_of_area
        self.Youngs_modulus = Youngs_modulus 
        self.springs_stiffness = springs_stiffness
        self.etol = etol
        self.dtol = dtol
        self.K_matrices = []
        self.M_matrices = []
        #self.spring_stiffness = 10000


    def assembly(self):
        for index, (key,value) in enumerate(self.lines.items()): 
            first_node, second_node =  value
            first_node_coordinates = np.array(self.nodes[first_node])
            second_node_coordinates = np.array(self.nodes[second_node]) 
            length = np.linalg.norm(second_node_coordinates-first_node_coordinates)
            if key == 'spring_1':
                break
            else:
                linear_density = self.linear_density[key]
                area_cross_section = self.area_cross_section[key]
                moment_of_area = self.moment_of_area[key]
                Youngs_modulus = self.Youngs_modulus[key]
            
                element = BarBeamElement(first_node_coordinates, # (x1, y1)
                    second_node_coordinates, # (x2, y2)
                    linear_density, # rho*A
                    Youngs_modulus*area_cross_section/length, # E*A/L
                    Youngs_modulus*moment_of_area/length,
                    self.etol,
                    self.dtol)
                K = element.generate_structural_stiffness_matrix()
                M = element.generate_structural_mass_matrix()
                self.K_matrices.append(K)
                self.M_matrices.append(M)
        for index, (key,value) in enumerate(self.springs_stiffness.items()):
            #print(f'key:{key}')
            #print(f'self.springs_stiffess[key]:{self.springs_stiffness[key]}')
            K_spring, M_spring = spring_matrices(self.springs_stiffness[key])
            self.K_matrices.append(K_spring)
            self.M_matrices.append(M_spring)
        print(f'self.M_matrices:{self.M_matrices}')
        #print(f'shape(self.K_matrices:{np.shape(self.K_matrices)}')
        max_DOFs = []
        for key,value in self.DOF.items():
            max_DOFs.append(max(value))
        max_DOF = max(max_DOFs)+1
        #print(f'max_DOF:{max_DOF}')
        input_assemble_K = []
        input_assemble_M = []
        for index, (key,value) in enumerate(self.DOF.items()):
            input_assemble_K.append((self.K_matrices[index],self.DOF[key]))
            input_assemble_M.append((self.M_matrices[index],self.DOF[key]))
        #print(f'input_assemble_K[-1]:{input_assemble_K[-1]}')
        #print(f'input_assemble_M[-1]:{input_assemble_M[-1]}')
        K_total = assemble_global_matrix(max_DOF,input_assemble_K)
        M_total = assemble_global_matrix(max_DOF,input_assemble_M)

        print(f'K_total with all DOF:{K_total}')
        print(f'M_total with all DOF:{M_total}')

        K_total = delete_degrees_of_freedom_from_matrix(K_total,self.constrained_dof)
        M_total = delete_degrees_of_freedom_from_matrix(M_total,self.constrained_dof)

        return K_total, M_total

            
        


    

    