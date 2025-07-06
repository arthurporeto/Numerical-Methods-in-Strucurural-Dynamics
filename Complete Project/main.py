from strussStructure import StrussStructure
from tools import *
from barBeamElement import BarBeamElement
import numpy as np

def main():
    pressure_load=0.01
    force_load2=-60
    force_load3=-40
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
    
    node_map = {
        1: {'ux': 0, 'uy': 1},
        2: {'ux': 3, 'uy': 4},
        3: {'ux': 6, 'uy': 7},
        4: {'ux': 9, 'uy': 10},
        5: {'ux': 12, 'uy': 13},
        6: {'ux': 15, 'uy': 16},
        7: {'ux': 18, 'uy': 19},
        8: {'ux': 21, 'uy': 22}
    }

    # bar matrices
    print(generate_mass_matrix(bar_shape_functions, length=2, linear_density=10, etol=1e-5), '\n')
    print(generate_stiffness_matrix(bar_shape_functions, length=2, axial_stiffness=1000,transverse_stiffness=0, dtol=1e-5, etol=1e-5), '\n')

    # beam matrices
    print(generate_mass_matrix(beam_shape_functions, length=2, linear_density=10, etol=1e-5), '\n')
    print(generate_stiffness_matrix(beam_shape_functions, length=2, axial_stiffness=0,transverse_stiffness=1, dtol=1e-5, etol=1e-5),'\n')

    element: BarBeamElement = BarBeamElement(
        coordinates_first_end=(0,0), 
        coordinates_second_end=(2,0), 
        linear_density=10, # rho*A
        axial_stiffness=1000, # E*A/L
        transverse_stiffness=1, # E*I/L
        dtol=1e-5,
        etol=1e-5
    )

    print(element.generate_structural_mass_matrix(), '\n')
    print(element.generate_structural_stiffness_matrix(),'\n')

    structure = StrussStructure(nodes = Nodes,lines = LINES,DOF=degrees_of_freedom, constrained_dof=np.array([0,1,22]),node_map=node_map,
        area_cross_section=area_cross_section,moment_of_area=moment_of_area,Youngs_modulus=Youngs_modulus,springs_stiffness=spring_stiffness,linear_density=linear_density,etol=1e-5,dtol=1e-5)

    K_total, M_total,K_total_modified,M_total_modified = structure.assembly()

    print(f'K:{K_total_modified}','\n')
    print(f'M:{M_total_modified}','\n')

    u,reactions = structure.solve_static(force_load2=-60,force_load3=-40,pressure_load=0.01)

    print(f'displacements:{u}')
    print(f'reactions:{reactions}')
    

    structure.plot_deformations(scale_factor=100)


    np.set_printoptions(precision=5,    # Show up to 5 digits after decimal point for float numbers
    suppress=True,  # Suppress small numbers to 0
    linewidth=80    # Adjust this value to control line breaks.
                    # 80 is a common default, try larger/smaller to fit your console.
)
   

    return K_total

if __name__ == "__main__":
    main()