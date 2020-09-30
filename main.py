# Pablo Ruiz Royo 28-09-2020

import numpy as np
import matplotlib.pyplot as mp
import matplotlib as cm

R = 1


def init_mesh_rect(dim_x: "in m", dim_y: "in m", dim_z: "in m", node_size: "in m") -> "mesh coord":
    """

    :param dim_x: x dimension, meters
    :param dim_y: y dimension, meters
    :param dim_z: z dimension, meters
    :param node_size: side size of nodes, meters
    :rtype: Mesh array of dimensions (nx, ny, nz)
    """

    n_dims = (int(dim_x / node_size), int(dim_y / node_size), int(dim_z / node_size))

    ns_x = dim_x / n_dims[0]
    ns_y = dim_y / n_dims[1]
    ns_z = dim_z / n_dims[2]

    X = np.arange(n_dims[0], dtype=float)
    Y = np.arange(n_dims[1], dtype=float)
    Z = np.arange(n_dims[2], dtype=float)

    X = (X + 0.5) * ns_x
    Y = (Y + 0.5) * ns_y
    Z = (Z + 0.5) * ns_z

    return X, Y, Z, n_dims


def init_parameters(n_dims) -> "pressures, velocities, densities, temperatures (in SI)":
    P = np.ones(n_dims, dtype=float)

    V = np.ones(n_dims + (3,), dtype=float)

    rho = np.ones(n_dims, dtype=float)

    T = np.ones(n_dims, dtype=float)

    return P, V, rho, T


def initial_conditions(P, V, rho, T):
    P[:, :, :] = 1

    V[:, :, :, 1] = 0

    rho[:, :, :] = 1

    T[:, :, :] = np.divide(P, rho) * R

    return P, V, rho, T


def boundary_conditions(P, V):
    P[1, :, :] = 2

    return P, V


def temporal_scheme(V, V_, time_step):
    """
    Euler temporal scheme
    :param V: next step
    :param V_: previous step
    :param time_step: time step, seconds
    """
    dt = (V - V_) / time_step

    return dt


def space_scheme(V, coord, axis):
    d = np.zeros(V.shape, dtype=float)
    coord1 = np.expand_dims(np.expand_dims(np.expand_dims(coord, axis=1), axis=2), axis=3)
    coord2 = np.expand_dims(np.expand_dims(np.expand_dims(coord, axis=1), axis=2), axis=0)
    coord3 = np.expand_dims(np.expand_dims(np.expand_dims(coord, axis=1), axis=0), axis=0)

    if axis == 0:
        d[0, :, :, :] = (V[1, :, :, :] - V[0, :, :, :]) / (coord1[1] - coord1[0])
        d[1:-2, :, :, :] = np.divide(V[2:-1, :, :, :] - V[0:-3, :, :, :], coord1[2:-1] - coord1[0:-3])
        d[-1, :, :, :] = (V[-1, :, :, :] - V[-2, :, :, :]) / (coord1[-1] - coord1[-2])
    elif axis == 1:
        d[:, 0, :, :] = (V[:, 1, :, :] - V[:, 0, :, :]) / (coord2[:, 1] - coord2[:, 0])
        d[:, 1:-2, :, :] = (V[:, 2:-1, :, :] - V[:, 0:-3, :, :]) / (coord2[:, 2:-1] - coord2[:, 0:-3])
        d[:, -1, :, :] = (V[:, -1, :, :] - V[:, -2, :, :]) / (coord2[:, -1] - coord2[:, -2])
    elif axis == 2:
        d[:, :, 0, :] = (V[:, :, 1, :] - V[:, :, 0, :]) / (coord3[:, :, 1] - coord3[:, :, 0])
        d[:, :, 1:-2, :] = (V[:, :, 2:-1, :] - V[:, :, 0:-3, :]) / (coord3[:, :, 2:-1] - coord3[:, :, 0:-3])
        d[:, :, -1, :] = (V[:, :, -1, :] - V[:, :, -2, :]) / (coord3[:, :, -1] - coord3[:, :, -2])

    return d


def gradient(coords, P):
    """

    :param P: Current pressure field (Pa)
    :param coords: rectangle mesh coordinates [X, Y, Z]
    """
    dP = np.zeros(P.shape + (3,))
    # Reshape coords so it can broadcast to more dimensions
    coords1 = np.expand_dims(np.expand_dims(coords, axis=2), axis=3)
    coords2 = np.expand_dims(np.expand_dims(coords, axis=2), axis=0)
    coords3 = np.expand_dims(np.expand_dims(coords, axis=0), axis=0)
    # X gradient component
    dP[0, :, :, 0] = (P[1, :, :] - P[0, :, :]) / (coords[0, 1] - coords[0, 0])
    dP[1:-2, :, :, 0] = (P[2:-1, :, :] - P[0:-3, :, :]) / (coords1[0, 2:-1] - coords1[0, 0:-3])
    dP[-1, :, :, 0] = (P[-1, :, :] - P[-2, :, :]) / (coords[0, -1] - coords[0, -2])

    # Y gradient component
    dP[:, 0, :, 1] = (P[:, 1, :] - P[:, 0, :]) / (coords[1, 1] - coords[1, 0])
    dP[:, 1:-2, :, 1] = (P[:, 2:-1, :] - P[:, 0:-3, :]) / (coords2[:, 1, 2:-1] - coords2[:, 1, 0:-3])
    dP[:, -1, :, 1] = (P[:, -1, :] - P[:, -2, :]) / (coords[1, -1] - coords[1, -2])

    # Z gradient component
    dP[:, :, 0, 2] = (P[:, :, 1] - P[:, :, 0]) / (coords[2, 1] - coords[2, 0])
    dP[:, :, 1:-2, 2] = (P[:, :, 2:-1] - P[:, :, 0:-3]) / (coords3[:, :, 2, 2:-1] - coords3[:, :, 2, 0:-3])
    dP[:, :, -1, 2] = (P[:, :, -1] - P[:, :, -2]) / (coords[2, -1] - coords[2, -2])

    return dP


def compute_derivatives(X, Y, Z, P, V, rho):
    """
    Apply Euler and mass conservation eqs

    :param X: X coordinates, shape (n_x)
    :param Y: Y coordinates, shape (n_y)
    :param Z: Z coordinates, shape (n_z)
    :param P: current step pressure (Pa), shape (n_x, n_y, n_z)
    :param V: current step velocity (m/s), shape (n_x, n_y, n_z, components)
    :param rho: current step density (kg/m^3), shape (n_x, n_y, n_z)
    :return: derivatives of V and rho
    """
    P, V = boundary_conditions(P, V)

    # Linear momentum conservation

    dVdx = space_scheme(V, X, axis=0)
    dVdy = space_scheme(V, Y, axis=1)
    dVdz = space_scheme(V, Z, axis=2)

    VdV = np.zeros(V.shape)

    VdV[:, :, :, 0] = np.multiply(dVdx[:, :, :, 0], V[:, :, :, 0]) + \
                      np.multiply(dVdy[:, :, :, 0], V[:, :, :, 1]) + \
                      np.multiply(dVdz[:, :, :, 0], V[:, :, :, 2])

    VdV[:, :, :, 1] = np.multiply(dVdx[:, :, :, 1], V[:, :, :, 0]) + \
                      np.multiply(dVdy[:, :, :, 1], V[:, :, :, 1]) + \
                      np.multiply(dVdz[:, :, :, 1], V[:, :, :, 2])

    VdV[:, :, :, 2] = np.multiply(dVdx[:, :, :, 2], V[:, :, :, 0]) + \
                      np.multiply(dVdy[:, :, :, 2], V[:, :, :, 1]) + \
                      np.multiply(dVdz[:, :, :, 2], V[:, :, :, 2])

    dP = gradient(np.array([X, Y, Z]), P)

    dP_rho = np.zeros(dP.shape, dtype=float)

    dP_rho[:, :, :, 0] = np.divide(dP[:, :, :, 0], rho)
    dP_rho[:, :, :, 1] = np.divide(dP[:, :, :, 1], rho)
    dP_rho[:, :, :, 2] = np.divide(dP[:, :, :, 2], rho)

    g = np.array([0., 0., 1.], dtype=float)

    dVdt = g - dP_rho - VdV

    #Mass conservation

    drhodt = - np.multiply(dVdx[:, :, :, 0], rho) - np.multiply(dVdx[:, :, :, 1], rho) - np.multiply(dVdx[:, :, :, 2], rho)

    return dVdt,drhodt


def compute_step(X, Y, Z, P_, V_, rho_, T, time_step):
    """

    :param X: X coordinates, shape (n_x)
    :param Y: Y coordinates, shape (n_y)
    :param Z: Z coordinates, shape (n_z)
    :param P_: previous step pressure (Pa), shape (n_x, n_y, n_z)
    :param V_: previous step velocity (m/s), shape (n_x, n_y, n_z, components)
    :param T: current step temperature (K), shape (n_x, n_y, n_z)
    :param rho_: previous step density (kg/m^3), shape (n_x, n_y, n_z)
    :param time_step: time step (s)
    """
    dV, drho = compute_derivatives(X, Y, Z, P_, V_, rho_)

    V = V_ + dV * time_step
    rho = rho_ + drho * time_step

    P = R * T * rho

    return P, V, rho


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Pablo')

    time_step = 0.1
    steps = 2

    X, Y, Z, n_dims = init_mesh_rect(1, 0.1, 0.1, 0.001)

    P, V, rho, T = init_parameters(n_dims)

    P, V, rho, T = initial_conditions(P, V, rho, T)

    P_graph = np.zeros((steps,) + P.shape, dtype=float)
    V_graph = np.zeros((steps,) + V.shape, dtype=float)
    rho_graph = np.zeros((steps,) + rho.shape, dtype=float)

    mp.figure(figsize=(12, 5))

    mp.subplot(131)
    mp.streamplot(Y, X, V[:, :, 0, 0], V[:, :, 0, 1])
    mp.subplot(132)
    mp.plot(X, rho[:, 0, 0])
    for i in range(steps):
        P_graph[i], V_graph[i], rho_graph[i] = P, V, rho
        P, V, rho = compute_step(X, Y, Z, P, V, rho, T, time_step)

    mp.subplot(133)
    mp.streamplot(X, Y, V[:, :, 0, 0], V[:, :, 0, 1])
    mp.subplot(132)
    mp.plot(X, rho[:, 0, 0])
    mp.show()