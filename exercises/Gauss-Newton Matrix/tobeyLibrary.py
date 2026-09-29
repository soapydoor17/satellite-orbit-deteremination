import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from astropy.time import Time
from astropy.coordinates import EarthLocation, GCRS
import astropy.units as u

MU_EARTH = 3.986004418*(10**14)
SPEED_OF_LIGHT = 299792458

#---------------------------------------------------
# Anomaly Propagation:
#   M0 -> M(t) -> E -> nu
#---------------------------------------------------

# Calculate mean motion (n)
def nCalc(a, mu=MU_EARTH):
    # Returns the mean motion in rads/s
    return np.sqrt(mu / a**3)

# Solve Kepler's equation M = E - e*sin(E) for E, via Newton-Raphson
def ECalc(M_deg, e, tolerance=1e-8):
    # Inputs:
    #   M - Mean anomaly (degrees), scalar or array-like
    #   e - Eccentricity
    #   tolerance - default to 1e-8
    # Outputs:
    #   E - Eccentric anomaly
    
    M_deg_arr = np.atleast_1d(np.asarray(M_deg, dtype=float))
    M_rad = np.deg2rad(M_deg_arr)

    # Initial guess
    E = np.where(((M_rad > -np.pi) & (M_rad < 0)) | (M_rad > np.pi),
                 M_rad - e, M_rad + e)

    for i in range(len(E)):
        E_i = E[i]
        while True:
            sin_E = np.sin(E_i)
            cos_E = np.cos(E_i)
            next_E = E_i + (M_rad[i] - E_i + e * sin_E) / (1 - e * cos_E)
            abs_diff = abs(next_E - E_i)
            E_i = next_E
            if abs_diff < tolerance:
                break
        E[i] = E_i

    return E[0] if E.size == 1 else E

# Propagate mean anomaly from a reference epoch to one or more observation
#   times (given as ISO 8601 strings), then solve for eccentric anomaly
def propagateM02E(elapsed_sec, M0_deg, n, e, tolerance=1e-8):
    # Inputs:
    #   elapsed_sec - Elapsed time since reference epoch t0 (s), i.e. (t - t0).sec
    #   M0_deg - initial mean anomaly (deg)
    #   n - mean motion (rads/s) 
    #   e - eccentricity
    # Outputs:
    #   E - eccentric anomaly (rad)

    n_deg_s = np.rad2deg(n)
    M_deg = M0_deg + n_deg_s * elapsed_sec

    return ECalc(M_deg, e, tolerance=tolerance)

# Calculate true anomaly (nu)
def nuCalc(E, e):
    # Inputs:
    #   E - Eccentric anomaly (radians)
    #   e - Eccentricity
    # Outputs:
    #   nu - True Anomaly (radians)
    nu = np.arctan2(np.sin(E) * np.sqrt(1-e**2), np.cos(E) - e)
    return nu


#---------------------------------------------------
# COE2RV:
#   Convert Classical Orbital Elements (COE) to 
#   Cartesian Position and Velocity vectors (RV)
#---------------------------------------------------
def COE2RV(coe, mu=MU_EARTH):
    # INPUT: 
    #   coe is an array of the Keplerian Orbital Elements
    #       a - semi-major axis (m)
    #       e - eccentricity
    #       i - inclination (deg)
    #       node - right ascension of the ascending node (deg)
    #       arg - argument of perigee (deg)
    #       nu - true anomaly (radian)
    #   mu - gravitational parameters (=GM). Default set to the value for Earth

    # OUTPUT:
    #   r - position vector of satellite
    #   v - velocity vector of satellite


    a, e, i, node, arg, nu = coe

    sin_nu = np.sin(nu)
    cos_nu = np.cos(nu)

    # Calculate semiparameter (p)
    p = a * (1-e**2)

    # Perifocal Coordinate System
    r_PQW = np.zeros(3)
    v_PQW = np.zeros(3)

    r_PQW[0] = (p * cos_nu) / (1 + e*cos_nu)
    r_PQW[1] = (p * sin_nu) / (1 + e*cos_nu)
    
    v_PQW[0] = -1 * np.sqrt(mu/p) * sin_nu
    v_PQW[1] = np.sqrt(mu/p) * (e + cos_nu)

    # Rotation Matrix
    R_node_z = R.from_euler('z', np.deg2rad(node)).as_matrix()
    R_i_x    = R.from_euler('x', np.deg2rad(i)).as_matrix()
    R_arg_z  = R.from_euler('z', np.deg2rad(arg)).as_matrix()
    R_total  = R_node_z @ R_i_x @ R_arg_z

    # Rotate Perifocal to IJK
    r_IJK = R_total @ r_PQW
    v_IJK = R_total @ v_PQW

    return r_IJK, v_IJK, R_total

#---------------------------------------------------
# Time conversion
#---------------------------------------------------
def ISOString(year, month, day, hour, minute, second):
    # Inputs:
    #   year, month, day, hour, minute - integer
    #   second - float
    # Output:
    #   dt_str - ISO 8601 datetime string
    dt_str = (
        f"{int(year):04d}-{int(month):02d}-{int(day):02d}T"
        f"{int(hour):02d}:{int(minute):02d}:{second:09.6f}"
    )
    return dt_str

# Get astropy.time.Time object of a specific row from a DataFrame
# Apply to whole dataframe:
#       df["Time"] = df.apply(getTime, axis=1)
def getTime(row, t_scale='utc'):
    # Input:  row of DataFrame that has the columns: "Year", "Month", "Day", "Hour", "Minute", "Second"
    #         t_scale - time scale, default 'utc'
    # Output: astropy.time.Time object for the inputted row
    iso_str = ISOString(
        row["Year"], row["Month"], row["Day"],
        row["Hour"], row["Minute"], row["Second"]
    )
    return Time(iso_str, scale=t_scale)


#---------------------------------------------------
# Groundstation Vectors:
#   Geodetic coords -> ECI
#---------------------------------------------------
def groundstationECI(lat_deg, lon_deg, alt_m, t):
    # Inputs:
    #   lat_deg - latitude of groundstation in degrees
    #   lon_deg - longitude of groundstation in degrees
    #   alt_m   - altitude of groundstation in metres
    #   t       - astropy.time.Time object (single time, or array of times)
    # Outputs:
    #   r_gs - cartesian position vector(s) of groundstation (m)
    #   v_gs - cartesian velocity vector(s) of groundstation (m/s)
    station = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg, height=alt_m*u.m)
    
    pos, vel = station.get_gcrs_posvel(obstime=t)

    r_gs = pos.xyz.to(u.m).value
    v_gs = vel.xyz.to(u.m/u.s).value
    return r_gs, v_gs


#---------------------------------------------------
# Doppler Shift Calculations from r and v
#---------------------------------------------------
# rho - position vector of the satellite w.r.t groundstation
def rhoCalc(r, r_gs):
    return r-r_gs

# rho_hat - unit vector of rho
def rho_hatCalc(rho):
    return rho / np.sqrt(rho.dot(rho))

# v_rel - velocity vector of the satellite w.r.t groundstation
def v_relCalc(v, v_gs):
    return v-v_gs

# frequency-to-range-rate scale factor
def kCalc(f_c, c=SPEED_OF_LIGHT):
    return f_c/c


# Doppler shift calculation
# NOTE: NOT USED?
def fDCalc(k, rho_hat, v_rel):
    return k * rho_hat * (-v_rel)

# Residual Calculation
# NOTE: NOT USED?
def residual(y_pred, y_true):
    return y_pred-y_true

#---------------------------------------------------
# Gradient Calculations
#---------------------------------------------------
# Relationship between Doppler shift and satellite's cartesian state
def dfD_dXCalc(k, v_rel, rho, rho_hat):
    mag_rho = np.sqrt(rho.dot(rho))

    I3 = np.eye(3)

    dfD_dr = -k * v_rel.T @ ( I3/mag_rho - (np.outer(rho, rho))/(mag_rho**3) )
    dfD_dv = -k * rho_hat.T
    return np.concatenate([dfD_dr, dfD_dv])


# Derivative of the satellite's cartesian state
def dX_dtCalc(v, r, mu=MU_EARTH):
    mag_r = np.sqrt(r.dot(r))
    dv_dt = -mu * r / (mag_r**3)

    return np.concatenate([v, dv_dt])

# Initial Mean Anomaly Gradient
def dfD_dM0Calc(dfD_dX, dX_dt, n):
    # Inputs:
    #   dfD_dX  - Change in Doppler measurement w.r.t. satellite's cartesian state (1x6 array)
    #   dX_dt   - Change in satellite's cartesian state w.r.t. time (6x1 array)
    #   n      - Mean motion (rad/s)
    # Outputs:
    #   dfD_dM0 - Change in Doppler shift w.r.t. Initial Mean Anomaly (Hz/deg)

    dM_dt_reciprocal = 1 / np.rad2deg(n)

    return dfD_dX @ dX_dt * dM_dt_reciprocal

# Semi-Major Axis Gradient
def dfD_daCalc(dfD_dX, dX_dt, elapsed_sec, rot_mat, n, a, e, E, nu, mu=MU_EARTH):
    # Inputs:
    #   dfD_dX      - Change in Doppler measurement w.r.t. satellite's cartesian state (1x6 array)
    #   dX_dt       - Change in satellite's cartesian state w.r.t. time (6x1 array)
    #   elapsed_sec - Elapsed time since reference epoch t0 (s), i.e. (t - t0).sec
    #   rot_mat     - Perifocal-to-ECI rotation matrix (from COE2RV)
    #   n           - Mean motion (rad/s)
    #   a           - Semi-major axis (m)
    #   e           - Eccentricity
    #   E           - Eccentric anomaly (rad)
    #   nu          - True anomaly (rad)
    #   mu          - Gravitational parameter (m^3/s^2) (Defaults to Earth's mu value)
    # Outputs:
    #   dfD_da  - Change in Doppler shift w.r.t semi-major axis
    # NOTE: For calculations in this function, M is in radians

    nu_arr = np.array([np.cos(nu), np.sin(nu), 0])
    E_arr = np.array([-np.sin(E), np.sqrt(1-e**2) * np.cos(E), 0])

    dr_da_M = (1 - e*np.cos(E)) * rot_mat @ nu_arr
    dv_da_M = -n / (2- 2*e*np.cos(E)) * rot_mat @ E_arr

    dX_da_M = np.concatenate([dr_da_M, dv_da_M])
    dX_dM = dX_dt / n
    dM_da = -3/2 * np.sqrt(mu / (a**5)) * elapsed_sec

    dX_da = dX_da_M + dX_dM * dM_da

    return dfD_dX @ dX_da    

#---------------------------------------------------
# Jacobian Calculations
#---------------------------------------------------
def Jacobian(dfD_dM0, dfD_da):
    # Create Jacobian Matrix
    # Inputs:
    #   dfD_dM0 - Initial mean anomaly gradient (scalar or Nx1 array)
    #   dfD_da   - Semi-major axis gradient (scalar or Nx1 array)
    # Outputs:
    #   J - Outputs Nx2 array, where N is the number of observations 

    # Convert scalar values to 1D arrays
    dfD_dM0 = np.atleast_1d(dfD_dM0)
    dfD_da = np.atleast_1d(dfD_da)

    N = len(dfD_dM0)       # Number of entries

    # Create matrix
    J = np.zeros((N, 2))
    for i in range(N):
        J[i] = np.array([dfD_dM0[i], dfD_da[i]])

    return J

#---------------------------------------------------
# Gauss-Newton Matrix
#---------------------------------------------------
def GaussNewton(J):
    # Create Gauss-Newton Matrix
    # Inputs:
    #   J - Nx2 Jacobian Matrix, where N is the number of observations
    # Outputs:
    #   G - 2x2 Gauss-Newton Matrix
     
    N = J.shape[0]
    return 1/N * J.T @ J

def standardisedGaussNewton_M0_a(J):
    # Create Standardised Gauss-Newton Matrix 
    # Inputs:
    #   J - Nx2 Jacobian Matrix, where N is the number of observations
    # Outputs:
    #   G_tilde - 2x2 Gauss-Newton Matrix

    G = GaussNewton(J)
    sigma_M0 = 1 / np.sqrt(G[0, 0])
    sigma_a = 1 / np.sqrt(G[1, 1])
    D = np.array([[sigma_M0, 0],[0, sigma_a]])

    return D.T @ G @ D