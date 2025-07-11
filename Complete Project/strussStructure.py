import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from barBeamElement import BarBeamElement
from tools import *
from matplotlib.lines import Line2D

class StrussStructure(object):
    def __init__(self,
                 max_DOF,
                 nodes,
                 lines,
                 DOF,
                 constrained_dof,
                 node_map,
                 area_cross_section,
                 moment_of_area,
                 Youngs_modulus,
                 linear_density, #rho*A
                 springs_stiffness, #{spring_1: spring_stiffness}
                 etol: float,
                 dtol: float):
        self.max_DOF=max_DOF
        self.nodes = nodes #nodes = {1: (0,0), 2: (length,0)}
        self.lines = lines #lines = {1: (1,3),'spring': (1,2)}
        self.DOF = DOF
        self.constrained_dof = constrained_dof
        self.node_map = node_map
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
                delta_x = second_node_coordinates[0] - first_node_coordinates[0]
                delta_y = second_node_coordinates[1] - first_node_coordinates[1]

                # Use np.arctan2 for robust angle calculation
                # This will give the angle in radians, from -pi to +pi
                angle = np.arctan2(delta_y, delta_x)

                #angle = np.arctan((second_node_coordinates[1]-first_node_coordinates[1])/(second_node_coordinates[0]-first_node_coordinates[0]))
                K_rotation = rotation_matrix_bar_beam_2d(angle).T@K@rotation_matrix_bar_beam_2d(angle)
                M_rotation = rotation_matrix_bar_beam_2d(angle).T@M@rotation_matrix_bar_beam_2d(angle)
                self.K_matrices.append(K_rotation)
                self.M_matrices.append(M_rotation)
        for index, (key,value) in enumerate(self.springs_stiffness.items()):
            #print(f'key:{key}')
            #print(f'self.springs_stiffess[key]:{self.springs_stiffness[key]}')
            K_spring, M_spring = spring_matrices(self.springs_stiffness[key])
            self.K_matrices.append(K_spring)
            self.M_matrices.append(M_spring)
        #print(f'self.M_matrices:{self.M_matrices}')
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
        self.K_total = assemble_global_matrix(max_DOF,input_assemble_K)
        self.M_total = assemble_global_matrix(max_DOF,input_assemble_M)

        

        #print(f'K_total with all DOF:{K_total}')
        #print(f'M_total with all DOF:{M_total}')

        self.K_total_modified = delete_degrees_of_freedom_from_matrix(self.K_total,self.constrained_dof)
        self.M_total_modified = delete_degrees_of_freedom_from_matrix(self.M_total,self.constrained_dof)

        return self.K_total, self.M_total, self.K_total_modified,self.M_total_modified
    
    def solve_static(self,force_load2,force_load3,pressure_load): #return two arrays: displacement and reaction forces on constrained nodes
        
        n = np.shape(self.K_total)[0]
        f = np.zeros((n,1))
        f[13] =force_load2
        f[7] = force_load3
        # Get coordinates for Node 7 and Node 8
        # Node 7 coordinates from your nodes dict: (3*L1_value, L2_value)
        # Node 8 coordinates from your nodes dict: (4*L1_value, 0)
        # Using the 'nodes' dictionary passed to the function:
        node7_coords = np.array(self.nodes[7])
        node8_coords = np.array(self.nodes[8])

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

        #f = np.reshape(1,-1)

        f_reduced = np.delete(f,self.constrained_dof)

        #np.delete(f,1)
        #np.delete(f,22)

        u = solve_lu_scipy(self.K_total_modified,f_reduced)

        u_full = np.zeros(n)
        all_dof_indices = np.arange(n)
        # Find the indices that are NOT constrained
        unconstrained_dof_indices = np.setdiff1d(all_dof_indices, self.constrained_dof)
        u_full[unconstrained_dof_indices] = u

        #u_full = np.insert(u,[0,0,22],[0,0,0])

        all_forces_in_equilibrium = self.K_total@u_full

        #print(f'all forces in equilibrium:{all_forces_in_equilibrium}')
        #print(f'f:{f}')
        #print(f'{f[constrained_DOF]}')

        #f = f.reshape(1,-1)

        #f.reshape(1,-1)
        self.reaction_forces = all_forces_in_equilibrium[self.constrained_dof] - f[self.constrained_dof] #already for the constrained DOFs
        self.reaction_forces = -np.array([self.reaction_forces[0,0],self.reaction_forces[1,1],self.reaction_forces[2,2]])
        self.u_full = -u_full


        #reactions_constrained_DOF = np.array([reaction_forces[0],reaction_forces[1],reaction_forces[22]])
        #reactions_constrained_DOF = reaction_forces[constrained_DOF]

        return  self.u_full,self.reaction_forces #displacements = solve_lu_scipy(K,f) #displacements
        #reactions = np.zeros(3,1)

    def plot_deformations(self, scale_factor, num_points_on_beam=50):
        """
        Plots the undeformed and statically deformed states of the structure,
        accurately showing beam deformation curves and applying a scale factor.

        Args:
            scale_factor (float): Factor to magnify displacements for visualization. Default is 1.0.
            num_points_on_beam (int): Number of intermediate points to plot along each beam's length.
        """
        if self.u_full is None:
            raise ValueError("Displacements (self.u_full) must be calculated first. Call solve_static() first.")

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Global X-coordinate')
        ax.set_ylabel('Global Y-coordinate')
        ax.set_title(f'Deformed vs. Undeformed Structure (Static Load, Scale Factor: {scale_factor:.0f}x)')
        ax.grid(True)

        # Calculate deformed coordinates for node markers
        deformed_nodes_coords = {}
        for node_id, coords in self.nodes.items():
            dx_global = self.u_full[self.node_map[node_id]['ux']] * scale_factor
            dy_global = self.u_full[self.node_map[node_id]['uy']] * scale_factor
            deformed_nodes_coords[node_id] = (coords[0] + dx_global, coords[1] + dy_global)
            
            # Plot original nodes (black circles)
            ax.plot(coords[0], coords[1], 'ko', markersize=6, alpha=0.7)
            # Plot deformed nodes (red circles)
            ax.plot(deformed_nodes_coords[node_id][0], deformed_nodes_coords[node_id][1], 'ro', markersize=6, alpha=0.7)
            
            # Add node labels
            ax.text(coords[0] - 50, coords[1] + 50, str(node_id), color='black', fontsize=9)
            ax.text(deformed_nodes_coords[node_id][0] + 50, deformed_nodes_coords[node_id][1] + 50, str(node_id) + "'", color='red', fontsize=9)


        # Plot original and deformed elements
        plotted_original_element_label = False
        plotted_deformed_element_label = False
        plotted_original_spring_label = False
        plotted_deformed_spring_label = False

        for element_id, node_pair in self.lines.items():
            node1_id, node2_id = node_pair
            
            orig_x1, orig_y1 = self.nodes[node1_id]
            orig_x2, orig_y2 = self.nodes[node2_id]

            # Get element properties (length and angle) for shape function evaluation
            delta_x_orig = orig_x2 - orig_x1
            delta_y_orig = orig_y2 - orig_y1
            element_length = np.linalg.norm(np.array([delta_x_orig, delta_y_orig]))
            
            if element_length == 0: # Skip if zero length
                continue

            angle_radians = np.arctan2(delta_y_orig, delta_x_orig)
            
            # Plot original element (straight line)
            if element_id == 'spring_1':
                ax.plot([orig_x1, orig_x2], [orig_y1, orig_y2], 'b:', linewidth=1.5,
                        label='Original Spring' if not plotted_original_spring_label else "_nolegend_")
                plotted_original_spring_label = True
            else:
                ax.plot([orig_x1, orig_x2], [orig_y1, orig_y2], 'k:', linewidth=1,
                        label='Original Element' if not plotted_original_element_label else "_nolegend_")
                plotted_original_element_label = True
            
            # Plot deformed element (curve for Bar+Beam, line for Spring)
            if element_id == 'spring_1':
                # Springs deform axially, so still a straight line between deformed nodes
                def_x1, def_y1 = deformed_nodes_coords[node1_id]
                def_x2, def_y2 = deformed_nodes_coords[node2_id]
                ax.plot([def_x1, def_x2], [def_y1, def_y2], 'b-', linewidth=2,
                        label='Deformed Spring' if not plotted_deformed_spring_label else "_nolegend_")
                plotted_deformed_spring_label = True
            else: # This is a Bar+Beam element, plot its curved deformation based on shape functions
                # 1. Get global nodal displacements for this element (6 DOFs)
                element_global_dof_indices = list(self.DOF[element_id]) # Get global DOF indices for this element
                global_element_displacements = self.u_full[element_global_dof_indices]

                # 2. Transform global nodal displacements to local nodal displacements
                R_6x6 = rotation_matrix_bar_beam_2d(angle_radians)
                local_element_displacements = R_6x6 @ global_element_displacements

                # Extract local nodal ux, uy, rz for both nodes
                u1_local = local_element_displacements[0]
                v1_local = local_element_displacements[1]
                theta1_local = local_element_displacements[2]
                u2_local = local_element_displacements[3]
                v2_local = local_element_displacements[4]
                theta2_local = local_element_displacements[5]

                # 3. Generate intermediate points along the element's local x'-axis
                x_prime_points = np.linspace(0, element_length, num_points_on_beam)

                # 4. Calculate local axial (u') and transverse (v') displacements at these points
                u_prime_path = np.zeros(num_points_on_beam)
                v_prime_path = np.zeros(num_points_on_beam)

                for i, x_prime in enumerate(x_prime_points):
                    # Axial displacement from bar shape functions
                    N_u_at_x_prime = bar_shape_functions(x_prime, element_length) # N1_u, N2_u
                    u_prime_path[i] = N_u_at_x_prime[0] * u1_local + N_u_at_x_prime[1] * u2_local

                    # Transverse displacement from beam shape functions
                    N_v_at_x_prime_scaled = beam_shape_functions(x_prime, element_length) # N1_v, N2_v*L, N3_v, N4_v*L
                    v_prime_path[i] = (N_v_at_x_prime_scaled[0] * v1_local +
                                       N_v_at_x_prime_scaled[1] * theta1_local +
                                       N_v_at_x_prime_scaled[2] * v2_local +
                                       N_v_at_x_prime_scaled[3] * theta2_local)

                # 5. Transform local deformed points (x' + u', v') back to global coordinates
                #    using the element's original position and angle, and applying scale_factor.
                orig_node1_x, orig_node1_y = self.nodes[node1_id]

                # Current local position (x', y'=0) plus local displacement
                deformed_local_x_prime = x_prime_points + u_prime_path * scale_factor
                deformed_local_y_prime = v_prime_path * scale_factor

                # Rotate these points back to global and translate by original node1 position
                global_deformed_x = orig_node1_x + deformed_local_x_prime * np.cos(angle_radians) - deformed_local_y_prime * np.sin(angle_radians)
                global_deformed_y = orig_node1_y + deformed_local_x_prime * np.sin(angle_radians) + deformed_local_y_prime * np.cos(angle_radians)

                # Plot the deformed path for the current element
                ax.plot(global_deformed_x, global_deformed_y, 'r-', linewidth=2,
                        label='Deformed Element' if not plotted_deformed_element_label else "_nolegend_")
                plotted_deformed_element_label = True

        ax.legend(loc='best')
        #plt.show()

    def eigensolver(self, mode_number, scale_factor, num_points_on_beam=50):
        """
        Calculates and plots a specified eigenmode, clearly showing the undeformed
        structure versus the deformed mode shape.

        Args:
            mode_number (int): The index of the eigenmode to plot (0 for the first mode).
            scale_factor (float): A scaling factor to amplify the mode shape for visualization.
            num_points_on_beam (int): The number of points for rendering beam curves.
        """
        # --- 1. Eigenvalue Problem Solution ---
        eigenvalues, eigenvectors_reduced = scipy.linalg.eigh(self.K_total_modified, self.M_total_modified)
        print(eigenvalues)
        natural_frequencies_rad_s = np.sqrt(np.maximum(eigenvalues, 0))

        sort_indices = np.argsort(natural_frequencies_rad_s)
        sorted_natural_frequencies = natural_frequencies_rad_s[sort_indices]
        sorted_mode_shapes_reduced = eigenvectors_reduced[:, sort_indices]

        if mode_number >= len(sorted_natural_frequencies):
            raise ValueError(f"Mode number {mode_number} is out of bounds. There are {len(sorted_natural_frequencies)} modes.")
        
        selected_frequency_rad_s = sorted_natural_frequencies[mode_number]
        selected_mode_shape_reduced = sorted_mode_shapes_reduced[:, mode_number]

        mode_shape_full = np.zeros(self.max_DOF)
        unconstrained_dof_indices = np.setdiff1d(np.arange(self.max_DOF), self.constrained_dof)
        mode_shape_full[unconstrained_dof_indices] = selected_mode_shape_reduced

        # --- 2. Plotting Setup ---
        fig, ax = plt.subplots(figsize=(13, 10))
        ax.set_aspect('equal', adjustable='box')
        frequency_hz = selected_frequency_rad_s / (2 * np.pi)
        ax.set_title(f'Eigenmode {mode_number + 1} | Natural Frequency: {selected_frequency_rad_s:.3f} rad/s (Scale: {scale_factor:.0f}x)')
        ax.grid(True)

        # --- 3. Plot Undeformed and Deformed Structures ---
        deformed_nodes_coords_mode = {}
        for node_id, coords in self.nodes.items():
            # Calculate deformed coordinates
            dx = mode_shape_full[self.node_map[node_id]['ux']] * scale_factor
            dy = mode_shape_full[self.node_map[node_id]['uy']] * scale_factor
            deformed_nodes_coords_mode[node_id] = (coords[0] + dx, coords[1] + dy)
            # Plot original node
            ax.plot(coords[0], coords[1], 'ko', markersize=6, alpha=0.6)

        for element_id, node_pair in self.lines.items():
            node1_id, node2_id = node_pair
            orig_x1, orig_y1 = self.nodes[node1_id]
            orig_x2, orig_y2 = self.nodes[node2_id]
            
            # Plot original element outline
            ax.plot([orig_x1, orig_x2], [orig_y1, orig_y2], 'k:', linewidth=1.5, alpha=0.8)

            # Plot deformed element shape
            if 'spring' in str(element_id):
                def_x1, def_y1 = deformed_nodes_coords_mode[node1_id]
                def_x2, def_y2 = deformed_nodes_coords_mode[node2_id]
                ax.plot([def_x1, def_x2], [def_y1, def_y2], 'b-', linewidth=2.5)
            else:
                element_length = np.linalg.norm(np.array([orig_x2-orig_x1, orig_y2-orig_y1]))
                if element_length == 0: continue
                angle_radians = np.arctan2(orig_y2-orig_y1, orig_x2-orig_x1)
                
                element_global_dof_indices = list(self.DOF[element_id])
                global_element_displacements = mode_shape_full[element_global_dof_indices]

                R_6x6 = rotation_matrix_bar_beam_2d(angle_radians)
                local_displacements = R_6x6 @ global_element_displacements
                u1, v1, th1, u2, v2, th2 = local_displacements

                x_prime = np.linspace(0, element_length, num_points_on_beam)
                u_prime = np.zeros_like(x_prime)
                v_prime = np.zeros_like(x_prime)

                for i, xp in enumerate(x_prime):
                    N_u = bar_shape_functions(xp, element_length)
                    u_prime[i] = N_u[0] * u1 + N_u[1] * u2
                    N_v = beam_shape_functions(xp, element_length)
                    v_prime[i] = N_v[0] * v1 + N_v[1] * th1 + N_v[2] * v2 + N_v[3] * th2
                
                deformed_local_x = x_prime + u_prime * scale_factor
                deformed_local_y = v_prime * scale_factor
                
                cos_a, sin_a = np.cos(angle_radians), np.sin(angle_radians)
                global_def_x = orig_x1 + deformed_local_x * cos_a - deformed_local_y * sin_a
                global_def_y = orig_y1 + deformed_local_x * sin_a + deformed_local_y * cos_a
                
                ax.plot(global_def_x, global_def_y, 'r-', linewidth=2.5)

        # --- 4. Create a Clear Legend ---
        legend_handles = [
            Line2D([0], [0], color='k', linestyle=':', linewidth=1.5, label='Undeformed Structure'),
            Line2D([0], [0], color='k', marker='o', markersize=6, linestyle='None', label='Undeformed Node'),
            Line2D([0], [0], color='r', linestyle='-', linewidth=2.5, label='Deformed Beam (Mode Shape)'),
            Line2D([0], [0], color='b', linestyle='-', linewidth=2.5, label='Deformed Spring (Mode Shape)')
        ]
        ax.legend(handles=legend_handles, loc='best')
        ax.set_xlabel('Global X-coordinate')
        ax.set_ylabel('Global Y-coordinate')
        #plt.show()