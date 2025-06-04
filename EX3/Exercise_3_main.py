from Exercise_3_solution import *

def main():
    length = 2
    linear_density = 10 # rho*A
    axial_stiffness = 1000 # E*A/L
    transverse_stiffness = 1 # E*I/L
    def bar_shape_functions(longitudinal_coordinate):
        xi = longitudinal_coordinate/length
        f1 = 1 - xi
        f2 = xi
        return np.array([f1, f2])

    def beam_shape_functions(longitudinal_coordinate):
        xi = longitudinal_coordinate/length
        f1 = 1 - 3*xi**2 + 2*xi**3
        f2 = length * xi * (1-xi)**2
        f3 = xi**2 * (3-2*xi)
        f4 = length * xi**2 * (xi-1)
        return np.array([f1, f2, f3, f4])

    def derivative_beam_shape_functions_num(longitudinal_coordinate):
        # or you can write the first derivatives yourself by hand
        return derivative(beam_shape_functions, longitudinal_coordinate, 1e-5)
    
    # bar matrices
    print(generate_mass_matrix(bar_shape_functions, length, linear_density, 1e-5), '\n')
    print(generate_stiffness_matrix(bar_shape_functions, length, axial_stiffness, 1e-5, 1e-5), '\n')

    # beam matrices
    print(generate_mass_matrix(beam_shape_functions, length, linear_density, 1e-5), '\n')
    print(generate_stiffness_matrix(derivative_beam_shape_functions_num, length, transverse_stiffness, 1e-5, 1e-5))

    
if __name__ == "__main__":
    main()