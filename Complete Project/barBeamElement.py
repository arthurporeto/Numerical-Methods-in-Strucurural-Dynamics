from tools import *

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
        K_beam = generate_stiffness_matrix(beam_shape_functions, self.length, 0,self.transverse_stiffness,self.dtol, self.etol)
        self.structural_stiffness_matrix = np.zeros((6,6))
        loc_array_bar = np.array([0,3])
        loc_array_beam = np.array([1,2,4,5])
        self.structural_stiffness_matrix = assemble_global_matrix(6,[(K_bar,loc_array_bar),(K_beam,loc_array_beam)])
        return self.structural_stiffness_matrix
    

    
    #K_global = np.zeros((total_DOF,total_DOF))