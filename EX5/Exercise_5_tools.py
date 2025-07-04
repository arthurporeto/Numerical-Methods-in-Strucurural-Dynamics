import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
ex3_dir = os.path.join(parent_dir, 'EX3')
sys.path.append(ex3_dir)
from Exercise_3_tools import *
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def calculate_eigenmodes(K_reduced: np.ndarray, M_reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the natural frequencies and mass-normalized mode shapes of a structure.

    Args:
        K_reduced (np.ndarray): The reduced global stiffness matrix (free DOFs only).
        M_reduced (np.ndarray): The reduced global mass matrix (free DOFs only).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - natural_frequencies (np.ndarray): A 1D array of natural frequencies (omega, in rad/s), sorted in ascending order.
            - mass_normalized_mode_shapes (np.ndarray): A 2D array where each column is a mass-normalized mode shape (eigenvector).
                                                        The mode shapes correspond to the sorted natural frequencies.
                                                        These satisfy Phi^T * M * Phi = I.
    Raises:
        ValueError: If K_reduced or M_reduced are not square or their dimensions don't match.
    """
    if K_reduced.shape[0] != K_reduced.shape[1] or M_reduced.shape[0] != M_reduced.shape[1]:
        raise ValueError("K_reduced and M_reduced must be square matrices.")
    if K_reduced.shape != M_reduced.shape:
        raise ValueError("K_reduced and M_reduced must have the same dimensions.")

    # Solve the generalized eigenvalue problem: K * Phi = lambda * M * Phi
    # scipy.linalg.eigh returns eigenvalues (lambda = omega^2) and eigenvectors (Phi).
    # The eigenvectors are automatically mass-normalized (Phi^T * M * Phi = I).
    eigenvalues, eigenvectors_reduced = scipy.linalg.eigh(K_reduced, M_reduced)

    # Filter out any negative eigenvalues (can occur due to numerical precision)
    # and take the square root to get natural frequencies (omega).
    natural_frequencies = np.sqrt(np.maximum(eigenvalues, 0)) # Ensure non-negative before sqrt

    # Sort eigenmodes by natural frequency (eigh usually returns them sorted, but good to ensure)
    sort_indices = np.argsort(natural_frequencies)
    natural_frequencies = natural_frequencies[sort_indices]
    mass_normalized_mode_shapes = eigenvectors_reduced[:, sort_indices]

    return natural_frequencies, mass_normalized_mode_shapes

def plot_specified_eigenmode(
    mode_shape_reduced: np.ndarray,
    mode_number: int,
    natural_frequency_rad_s: float, # Frequency in radians/second
    nodes_dict: dict,
    lines_dict: dict,
    dof_mapping_dict: dict, # Maps element IDs to global DOF tuples
    constrained_dof_indices: np.ndarray,
    total_global_dofs: int,
    scale_factor_modes: float = 1.0,
    plot_undeformed: bool = True,
    num_points_on_beam: int = 50
):
    """
    Visualizes a single eigenmode (mode shape) of the structure.

    Args:
        mode_shape_reduced (np.ndarray): The reduced (free DOFs only) mode shape vector for this specific mode.
        mode_number (int): The sequential number of the mode (e.g., 1st mode, 2nd mode).
        natural_frequency_rad_s (float): The natural frequency of this mode in radians per second.
        nodes_dict (dict): Original node coordinates.
        lines_dict (dict): Element connectivity.
        dof_mapping_dict (dict): Maps element IDs (or 'spring_1') to their global DOF tuple (q-values).
        constrained_dof_indices (np.ndarray): Indices of constrained DOFs in the full system.
        total_global_dofs (int): Total number of DOFs in the unreduced system.
        scale_factor_modes (float): Factor to scale mode shape displacements for visualization.
        plot_undeformed (bool): If True, plots the undeformed shape alongside the deformed mode shape.
        num_points_on_beam (int): Number of intermediate points to plot along each beam element's length.
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Global X-coordinate')
    ax.set_ylabel('Global Y-coordinate')
    
    # Convert frequency to Hz for display
    frequency_hz = natural_frequency_rad_s / (2 * np.pi)
    ax.set_title(f'Mode {mode_number}: Natural Frequency = {frequency_hz:.4f} Hz (Scaled by {scale_factor_modes:.0f})')
    ax.grid(True)

    # Map all DOF indices for reconstruction
    all_dof_indices = np.arange(total_global_dofs)
    unconstrained_dof_indices = np.setdiff1d(all_dof_indices, constrained_dof_indices)

    # Reconstruct the full mode shape vector (insert zeros at constrained DOFs)
    mode_shape_full = np.zeros(total_global_dofs)
    mode_shape_full[unconstrained_dof_indices] = mode_shape_reduced

    # Normalize the mode shape for plotting (e.g., max absolute displacement is 1)
    # This is for visualization scaling, separate from mass-normalization.
    max_abs_displacement_component = np.max(np.abs(mode_shape_full))
    if max_abs_displacement_component > 1e-10: # Avoid division by zero for trivial modes
        mode_shape_full_normalized_for_plot = mode_shape_full / max_abs_displacement_component
    else:
        mode_shape_full_normalized_for_plot = mode_shape_full # Keep as is if it's a zero mode

    # Define node_to_ux_uy_dof for plotting (maps node IDs to their ux/uy DOF indices)
    node_to_ux_uy_dof = {
        1: {'ux': 0, 'uy': 1}, 2: {'ux': 3, 'uy': 4}, 3: {'ux': 6, 'uy': 7},
        4: {'ux': 9, 'uy': 10}, 5: {'ux': 12, 'uy': 13}, 6: {'ux': 15, 'uy': 16},
        7: {'ux': 18, 'uy': 19}, 8: {'ux': 21, 'uy': 22}
    }

    # Plot undeformed structure if requested
    if plot_undeformed:
        for element_id, node_pair in lines_dict.items():
            node1_id, node2_id = node_pair
            orig_x1, orig_y1 = nodes_dict[node1_id]
            orig_x2, orig_y2 = nodes_dict[node2_id]
            if element_id == 'spring_1':
                ax.plot([orig_x1, orig_x2], [orig_y1, orig_y2], 'b:', linewidth=1, alpha=0.6, label='Original Spring')
            else:
                ax.plot([orig_x1, orig_x2], [orig_y1, orig_y2], 'k:', linewidth=1, alpha=0.6, label='Original Element')
        # Ensure labels don't duplicate on subsequent modes
        ax.plot([], [], 'k:', linewidth=1, alpha=0.6, label='_nolegend_') # Dummy for legend spacing
        ax.plot([], [], 'b:', linewidth=1.5, alpha=0.6, label='_nolegend_') # Dummy for legend spacing

    # Calculate and plot deformed shape for the current mode
    deformed_nodes_coords_mode = {}
    for node_id, coords in nodes_dict.items():
        dx_global = mode_shape_full_normalized_for_plot[node_to_ux_uy_dof[node_id]['ux']] * scale_factor_modes
        dy_global = mode_shape_full_normalized_for_plot[node_to_ux_uy_dof[node_id]['uy']] * scale_factor_modes
        deformed_nodes_coords_mode[node_id] = (coords[0] + dx_global, coords[1] + dy_global)

        # Plot deformed nodes
        ax.plot(deformed_nodes_coords_mode[node_id][0], deformed_nodes_coords_mode[node_id][1], 'ro', markersize=6, alpha=0.7)
        # Add node labels (optional, can clutter for many nodes)
        # ax.text(deformed_nodes_coords_mode[node_id][0] + 20, deformed_nodes_coords_mode[node_id][1] + 20, str(node_id), color='red', fontsize=9)

    for element_id, node_pair in lines_dict.items():
        node1_id, node2_id = node_pair
        orig_x1, orig_y1 = nodes_dict[node1_id]
        orig_x2, orig_y2 = nodes_dict[node2_id]

        delta_x_orig = orig_x2 - orig_x1
        delta_y_orig = orig_y2 - orig_y1
        element_length = np.linalg.norm(np.array([delta_x_orig, delta_y_orig]))
        if element_length == 0: continue
        angle_radians = np.arctan2(delta_y_orig, delta_x_orig)

        if element_id == 'spring_1':
            def_x1, def_y1 = deformed_nodes_coords_mode[node1_id]
            def_x2, def_y2 = deformed_nodes_coords_mode[node2_id]
            ax.plot([def_x1, def_y1], [def_x2, def_y2], 'b-', linewidth=2, label='Deformed Spring') # No, this should be def_x1,def_y1 to def_x2,def_y2
            ax.plot([def_x1, def_x2], [def_y1, def_y2], 'b-', linewidth=2, label='Deformed Spring' if mode_number == 1 else "_nolegend_")
        else: # Bar+Beam element
            element_global_dof_indices = list(dof_mapping_dict[element_id])
            global_element_displacements = mode_shape_full_normalized_for_plot[element_global_dof_indices]

            R_6x6 = rotation_matrix_bar_beam_2d(angle_radians)
            local_element_displacements = R_6x6 @ global_element_displacements

            u1_local = local_element_displacements[0]
            v1_local = local_element_displacements[1]
            theta1_local = local_element_displacements[2]
            u2_local = local_element_displacements[3]
            v2_local = local_element_displacements[4]
            theta2_local = local_element_displacements[5]

            x_prime_points = np.linspace(0, element_length, num_points_on_beam)
            u_prime_path = np.zeros(num_points_on_beam)
            v_prime_path = np.zeros(num_points_on_beam)

            for k, x_prime in enumerate(x_prime_points):
                N_u_at_x_prime = bar_shape_functions(x_prime, element_length)
                u_prime_path[k] = N_u_at_x_prime[0] * u1_local + N_u_at_x_prime[1] * u2_local

                N_v_at_x_prime_scaled = beam_shape_functions(x_prime, element_length)
                v_prime_path[k] = (N_v_at_x_prime_scaled[0] * v1_local +
                                   N_v_at_x_prime_scaled[1] * theta1_local +
                                   N_v_at_x_prime_scaled[2] * v2_local +
                                   N_v_at_x_prime_scaled[3] * theta2_local)

            orig_node1_x, orig_node1_y = nodes_dict[node1_id]

            # Global deformed path points for the current mode, scaled for visualization
            global_deformed_x = orig_node1_x + (x_prime_points + u_prime_path * scale_factor_modes) * np.cos(angle_radians) - (v_prime_path * scale_factor_modes) * np.sin(angle_radians)
            global_deformed_y = orig_node1_y + (x_prime_points + u_prime_path * scale_factor_modes) * np.sin(angle_radians) + (v_prime_path * scale_factor_modes) * np.cos(angle_radians)

            ax.plot(global_deformed_x, global_deformed_y, 'r-', linewidth=2, label='Deformed Element' if mode_number == 1 else "_nolegend_")
        
    ax.legend(loc='best')
    plt.show()