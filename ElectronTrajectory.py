import numpy as np

# ----------------------------------------------------------

def runge_kutta_4th_2fun(fun1, fun2, x0, xn, n, y10, y20):
    h = (xn - x0) / n
    # y1, y2 = y10, y20

    y1_arr = np.zeros(n+1, dtype=np.float32)
    y2_arr = np.zeros(n+1, dtype=np.float32)

    for i in range(n+1):
        K11 = fun1(x0, y10, y20)
        K12 = fun2(x0, y10, y20)

        x = x0 + h/2
        y1 = y10 + (h/2) * K11
        y2 = y20 + (h/2) * K12

        K21 = fun1(x, y1, y2)
        K22 = fun2(x, y1, y2)

        y1 = y10 + (h/2) * K21
        y2 = y20 + (h/2) * K22

        K31 = fun1(x, y1, y2)
        K32 = fun2(x, y1, y2)

        x = x0 + h
        y1 = y10 + h * K31
        y2 = y20 + h * K32

        K41 = fun1(x, y1, y2)
        K42 = fun2(x, y1, y2)

        y1 = y10 + (h/6) * (K11 + 2*K21 + 2*K31 + K41)
        y2 = y20 + (h/6) * (K12 + 2*K22 + 2*K32 + K42)

        y1_arr[i] = y1
        y2_arr[i] = y2

        x0 = x
        y10 = y1
        y20 = y2

    return y1_arr, y2_arr

# ----------------------------------------------------------

def rad2deg(rad):
    return rad * 180.0 / np.pi

def const_fun(z):
    return 1

def B_bell_fun(z, B0, a):
    return B0 / (1 + (z/a)**2)

# ----------------------------------------------------------
# Radial equation
# ----------------------------------------------------------

def radial_eq(z0, zn, n, y10, y20, B_fun=const_fun, eta=1.0, psi=1.0):
    A = -(eta ** 2) / (4 * psi)

    def radial_eq_fun1(z, y1, y2):
        return y2

    def radial_eq_fun2(z, y1, y2, B_func=B_fun, A_coeff=A):
        return A_coeff * (B_func(z) ** 2) * y1

    dists, slopes = runge_kutta_4th_2fun(radial_eq_fun1, radial_eq_fun2, z0, zn, n, y10, y20)

    z_arr = np.linspace(z0, zn, n+1, dtype=np.float32)
    z_vs_dists = np.vstack((z_arr, dists)).T
    z_vs_slopes = np.vstack((z_arr, slopes)).T
    np.savetxt('dists.txt', z_vs_dists, fmt='%.4e')
    np.savetxt('slopes.txt', z_vs_slopes, fmt='%.4e')

    return dists[n-1], slopes[n-1]

# ----------------------------------------------------------
# Azimuthal equation
# ----------------------------------------------------------

def azimuthal_eq(z0, zn, n, phi0, B_fun=const_fun, eta=1.0, psi=1.0):
    dz = (zn-z0) / n
    const_val = (eta * dz) / (6 * np.sqrt(psi))
    z_arr = np.linspace(z0, zn, n + 1, dtype=np.float32)

    phi_tot = phi0
    phi_tot_file = open('phi_total.txt', 'w')
    # phi_tot_file.write('{0}\n'.format(phi_tot))

    for i in range(0, n+1):
        B_comp = B_fun(z_arr[i])
        if 0 < i < n:
            B_comp *= 4 if i % 2 else 2

        phi_tot += const_val * B_comp
        phi_tot_file.write('{0:.4e}\t{1:.4e}\n'.format(z_arr[i], phi_tot))

    phi_tot_file.close()
    return phi_tot

# ----------------------------------------------------------
# Calculate electron trajectory
# ----------------------------------------------------------

# Constants
el_charge = 1.602e-19       # C
el_rest_mass = 9.109e-31    # kg
light_speed = 2.998e8       # m/s
Uacc_300kV = 3.0e5          # V
# lambda_300kV = 1.969e-12  # m

def calc_el_trajectory(z0, zn, n, phi0, y10, y20, B_fun=B_bell_fun):
    eta = np.sqrt(el_charge / (2 * el_rest_mass))
    eps = (eta ** 2) / (light_speed ** 2)
    psi = Uacc_300kV + eps * (Uacc_300kV ** 2)

    dist, slope = radial_eq(z0, zn, n, y10, y20, B_fun, eta, psi)
    total_angle = azimuthal_eq(z0, zn, n, phi0, B_fun, eta, psi)

    print('For z = {0:.3f} um:\n'
          'R distance = {1:.3f} um\n'
          'R slope = {2:.3f} deg\n'
          'Total angle = {3:.3f} deg'.format(zn * 1e6, dist * 1e6, rad2deg(slope), rad2deg(total_angle)))

if __name__ == '__main__':
    z_0 = -5.0e-3           # initial distance from the middle of magnetic lens
    z_n = 5.0e-3            # final distance for which the trajectory will be calculated
    n_steps = 500           # define precision of results (more steps == higher precision)
    phi_0 = np.pi / 2.0     # initial azimuthal angle
    r_dist_0 = 1.0e-3       # initial distance from optical axis
    r_slope_0 = 0.0         # initial slope of electron trajectory (if 0, then trajectory is parallel to the optical axis)
    B_0 = 2.0               # maximum B value of the magnetic lens [T]
    a_par = 1.0e-3          # parameter of Glaser's bell-shaped field

    def B_bell_fun2(z):
        return B_bell_fun(z, B0=B_0, a=a_par)

    calc_el_trajectory(z_0, z_n, n_steps, phi_0, r_dist_0, r_slope_0, B_fun=B_bell_fun2)