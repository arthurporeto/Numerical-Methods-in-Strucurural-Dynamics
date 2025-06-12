from Exercise_3_tools import *

def main():
    length = 2
    linear_density = 10 # rho*A
    axial_stiffness = 1000 # E*A/L
    transverse_stiffness = 1 # E*I/L

    # bar matrices
    print(generate_mass_matrix(bar_shape_functions, length, linear_density, 1e-5), '\n')
    #print(derivative_beam_shape_functions_num(beam_shape_functions,1,1e-5))
    print(generate_stiffness_matrix(bar_shape_functions, length, axial_stiffness,0, 1e-5, 1e-5), '\n')

    # beam matrices
    print(generate_mass_matrix(beam_shape_functions, length, linear_density, 1e-5), '\n')
    print(generate_stiffness_matrix(beam_shape_functions, length,0, transverse_stiffness, 1e-5, 1e-5),'\n')

    Element = BarBeamElement(
    coordinates_first_end=(0,0), 
    coordinates_second_end=(2,0), 
    linear_density=10, # rho*A
    axial_stiffness=1000, # E*A/L
    transverse_stiffness=1, # E*I/L
    etol=1e-5,
    dtol=1e-5 #equality tolerance
    )


    print(Element.generate_structural_mass_matrix(), '\n')
    print(Element.generate_structural_stiffness_matrix())

    print(assembly_struss_structure(area_cross_section=10,
length1=400,
length2=300,
moment_of_area=100,
Youngs_modulus=100000,
spring_stiffness=10000,
linear_density=1e-3))

    
if __name__ == "__main__":
    main()