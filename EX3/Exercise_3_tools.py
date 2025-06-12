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


def generate_stiffness_matrix(f,length,axial_stiffness,transverse_stiffness,dtol,etol):
    # Calculate the width of each trapezoid
    n = int(round(1/etol)) #number of trapezoid
    h = (length - 0) / n

    # Generate the x-values for each point
    # x_values = [a + i * h for i in range(n + 1)] # List comprehension way
    x_values = np.linspace(0, length, n + 1) # More efficient NumPy way

    # Evaluate the function at each x-value
    if axial_stiffness != 0:
        y_values = [np.outer(derivative(f,x,dtol),derivative(f,x,dtol)) for x in x_values]
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
        y_values = [np.outer(derivative_beam_shape_functions_num(f,x,dtol),derivative_beam_shape_functions_num(f,x,dtol)) for x in x_values]
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

def bar_shape_functions(longitudinal_coordinate):
        length = 2
        xi = longitudinal_coordinate/length
        f1 = 1 - xi
        f2 = xi
        return np.array([f1, f2])

def beam_shape_functions(longitudinal_coordinate):
    length = 2
    xi = longitudinal_coordinate/length
    f1 = 1 - 3*xi**2 + 2*xi**3
    f2 = length * xi * (1-xi)**2
    f3 = xi**2 * (3-2*xi)
    f4 = length * xi**2 * (xi-1)
    return np.array([f1, f2, f3, f4])

def derivative_beam_shape_functions_num(f,x,dtol):
    # or you can write the first derivatives yourself by hand
    return (f(x + dtol) - 2 * f(x) + f(x - dtol)) / (dtol**2)
    




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
    


def assembly_struss_structure(area_cross_section,
length1,
length2,
moment_of_area,
Youngs_modulus,
spring_stiffness,
linear_density):
    #Define each element
    element1 = BarBeamElement(coordinates_first_end=(0,0),
                              coordinates_second_end=(length1,0),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element2 = BarBeamElement(coordinates_first_end=(length1,0),
                              coordinates_second_end=(length1,length2),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element3 = BarBeamElement(coordinates_first_end=(length1,length2),
                              coordinates_second_end=(length1*2,0),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element4 = BarBeamElement(coordinates_first_end=(length1,0),
                              coordinates_second_end=(length1*2,0),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element5 = BarBeamElement(coordinates_first_end=(length1,length2),
                              coordinates_second_end=(length1*2,length2*2),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element6 = BarBeamElement(coordinates_first_end=(length1*2,length2*2),
                              coordinates_second_end=(length1*2,0),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element7 = BarBeamElement(coordinates_first_end=(length1*2,0),
                              coordinates_second_end=(length1*3,0),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element8 = BarBeamElement(coordinates_first_end=(length1*2,0),
                              coordinates_second_end=(length1*3,length2),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element9 = BarBeamElement(coordinates_first_end=(length1*2,length2*2),
                              coordinates_second_end=(length1*3,length2),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element10 = BarBeamElement(coordinates_first_end=(length1*3,length2),
                              coordinates_second_end=(length1*3,0),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element11 = BarBeamElement(coordinates_first_end=(length1*3,0),
                              coordinates_second_end=(length1*4,0),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    element12 = BarBeamElement(coordinates_first_end=(length1*3,length2),
                              coordinates_second_end=(length1*4,0),
                              linear_density=linear_density,
                              axial_stiffness=Youngs_modulus*area_cross_section/length1,
                              transverse_stiffness=Youngs_modulus*moment_of_area/length1,
                              etol=1e-5,
                              dtol=1e-5)
    spring = BarBeamElement(coordinates_first_end=(0,0),
                            coordinates_second_end=(length1,0),
                            linear_density=0,
                            axial_stiffness=spring_stiffness,
                            transverse_stiffness=0,
                            etol=1e-5,
                            dtol=1e-5)
    #Create mass and stiffness matrices
    M1 = element1.generate_structural_mass_matrix()
    K1 = element1.generate_structural_stiffness_matrix()
    M2 = element2.generate_structural_mass_matrix()
    K2 = element2.generate_structural_stiffness_matrix()
    M3 = element3.generate_structural_mass_matrix()
    K3 = element3.generate_structural_stiffness_matrix()
    M4 = element4.generate_structural_mass_matrix()
    K4 = element4.generate_structural_stiffness_matrix()
    M5 = element5.generate_structural_mass_matrix()
    K5 = element5.generate_structural_stiffness_matrix()
    M6 = element6.generate_structural_mass_matrix()
    K6 = element6.generate_structural_stiffness_matrix()
    M7 = element7.generate_structural_mass_matrix()
    K7 = element7.generate_structural_stiffness_matrix()
    M8 = element8.generate_structural_mass_matrix()
    K8 = element8.generate_structural_stiffness_matrix()
    M9 = element9.generate_structural_mass_matrix()
    K9 = element9.generate_structural_stiffness_matrix()
    M10 = element10.generate_structural_mass_matrix()
    K10 = element10.generate_structural_stiffness_matrix()
    M11 = element11.generate_structural_mass_matrix()
    K11 = element11.generate_structural_stiffness_matrix()
    M12 = element12.generate_structural_mass_matrix()
    K12 = element12.generate_structural_stiffness_matrix()
    K_spring = spring.generate_structural_stiffness_matrix()

    total_DOF = 25
    loc_array_1 = np.array([0,1,2,6,7,8])
    loc_array_2 = np.array([3,4,5,6,7,24])
    loc_array_3 = np.array([6,7,25,9,10,11])
    loc_array_4 = np.array([3,4,5,9,10,11])
    loc_array_5 = np.array([6,7,26,12,13,14])
    loc_array_6 = np.array([9,10,11,12,13,14])
    loc_array_7 = np.array([9,10,11,15,16,17])
    loc_array_8 = np.array([9,10,11,18,19,20])
    loc_array_9 = np.array([12,13,14,18,19,20])
    loc_array_10 = np.array([15,16,17,18,19,20])
    loc_array_11 = np.array([15,16,17,21,22,23])
    loc_array_12 = np.array([18,19,20,21,22,27])
    loc_array_spring = np.array([0,3])

    K = assemble_global_matrix(total_DOF,[(K1,loc_array_1),
                                          (K2,loc_array_2),
                                          (K3,loc_array_3),
                                          (K4,loc_array_4),
                                          (K5,loc_array_5),
                                          (K6,loc_array_6),
                                          (K7,loc_array_7),
                                          (K8,loc_array_8),
                                          (K9,loc_array_9),
                                          (K10,loc_array_10),
                                          (K11,loc_array_11),
                                          (K12,loc_array_12),
                                          (K_spring,loc_array_spring)])

    return delete_degrees_of_freedom_from_matrix(K,np.array([0,1,2,5,11,14,17,20,22]))

    
    #K_global = np.zeros((total_DOF,total_DOF))

