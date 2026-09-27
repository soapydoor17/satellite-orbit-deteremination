import numpy as np
from scipy.spatial.transform import Rotation as R

from astropy.time import Time
from astropy.coordinates import EarthLocation, GCRS
import astropy.units as u

MU_EARTH = 3.986004418*(10**14)
SPEED_OF_LIGHT = 299792458

#---------------------------------------------------
# Anomaly Propagation:
#   M0 -> M -> E -> nu
#---------------------------------------------------

# Calculate mean motion (n)
def nCalc(a, mu=MU_EARTH):
    # Returns the mean motion in rads/s
    return np.sqrt(mu / a**3)

# Newton Raphson to get eccentric anomaly (E) from mean anomaly (M)
def NewtRaph(M, e, tolerance=10**-8):
    # Inputs:
    #   M - Mean anomaly (degrees)
    #   e - Eccentricity
    #   tolerance - default to 10^8
    # Outputs:
    #   E - Eccentric anomaly
    
    M_rad = np.deg2rad(M)

    if (M_rad>-np.pi and M_rad<0) or (M_rad>np.pi):
        E = M_rad-e
    else:
        E = M_rad+e

    while True:
        sin_E = np.sin(E)
        cos_E = np.cos(E)
        nextE = E + (M_rad-E+e*sin_E)/(1-e*cos_E)

        abs_diff = abs(nextE-E)
        E = nextE

        if abs_diff < tolerance:
            break

    return E

# Calculate eccentric anomaly (E) across several timestamps
def MultiNewtRaph(t, M_0, n, e, tolerance=10**-8):
    # Inputs:
    #   t - array of time entries (s)
    #   M_0 - initial mean anomaly (deg)
    #   n - mean motion (rads/s) 
    #   e - eccentricity
    # Outputs:
    #   nu_t - true anomaly (in radians) at each point in time

    t_0 = t[0]

    # Mean anomalies
    M_t = np.zeros_like(t)
    M_t = M_0 + n*(t-t_0)

    # Newton-Raphson to find eccentric anomaly (E)
    E_t = np.zeros_like(M_t)
    for i in range(len(M_t)):
        E_t[i] = NewtRaph(M[i], e, tolerance=tolerance)

    return E_t

# Calculate true anomaly (nu)
def nuCalc(E, e):
    # Inputs:
    #   E - Eccentric anomaly (radians)
    #   e - Eccentricity
    # Outputs:
    #   nu - True Anomaly (radians)
     
    cosE = np.cos(E)
    nu = np.arccos( (cosE - e)/(1 - e*cosE) ) # in radians
    return nu


#---------------------------------------------------
# COE2RV:
#   Convert Classical Orbital Elements (COE) to 
#   Cartesian Position and Velocity vectors (RV)
#---------------------------------------------------
def COE2RV(coe, mu=MU_EARTH):
    # INPUT: 
    #   coe is an array of the Keplerian Orbital Elements
    #       a - semi-major axis
    #       e - eccentricity
    #       i - inclination
    #       node - right ascension of the ascending node
    #       arg - argument of perigee
    #       nu - true anomaly
    #   mu - gravitational parameters (=GM). Default set to the value for Earth

    # OUTPUT:
    #   r - position vector of satellite
    #   v - velocity vector of satellite


    a, e, i, node, arg, nu = coe

    sin_nu = np.sin(np.deg2rad(nu))
    cos_nu = np.cos(np.deg2rad(nu))

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
# Groundstation Vectors:
#   Geodetic coords -> ECI
#---------------------------------------------------
def groundstationECI(lat_deg, lon_deg, alt_m, time, t_scale='utc'):
    # Inputs:
    #   lat_deg - latitude of groundstation in degrees
    #   lon_deg - longitude of groundstation in degrees
    #   alt_m   - altitude of groundstation in meteers
    #   time    - ISO 8601 datetime string
    #   t_scale - Assumed timezone is UTC
    # Outputs:
    #   r_gs - cartesian position vector of groundstation
    #   v_gs - cartesian velocity vector of groundstation 
    station = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg, height=alt_m*u.m)
    t = Time(time, scale=t_scale)
    
    pos, vel = station.get_gcrs_posvel(obstime=t)
	
    r_gs = pos.xyz.to(u.m).value 
    v_gs = vel.xyz.to(u.m/u.s).value
	
    return r_gs, v_gs

# Convert separate Y/M/D/H/M/S values into an ISO 8601 datetime string
def row_to_iso_string(year, month, day, hour, minute, second):
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

# Get ISO 8601 datetime string of a specific row from a DataFrame
# Apply to whole dataframe: 
#       df["ISO Time"] = df.apply(convert_row, axis=1)
def convert_row(row):
    # Input:  row of DataFrame that has the columns: "Year", "Month", "Day", "Hour", "Minute", "Second"
    # Output: ISO 8601 datetime string of the inputted row
    return row_to_iso_string(
        row["Year"], row["Month"], row["Day"],
        row["Hour"], row["Minute"], row["Second"]
    )

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
def fDCalc(k, rho_hat, v_rel):
    return k * rho_hat * (-v_rel)

# Residual Calculation
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
def dfD_dM_0Calc(dfD_dX, dX_dt, n):
    # Inputs:
    #   dfD_dX  - Change in Doppler measurement w.r.t. satellite's cartesian state (1x6 array)
    #   dX_dt   - Change in satellite's cartesian state w.r.t. time (6x1 array)
    #   n      - Mean motion (rad/s)
    # Outputs:
    #   dfD_dM_0 - Change in Doppler shift w.r.t. Initial Mean Anomaly (Hz/deg)

    dM_dt_reciprocal = 1 / np.rad2deg(n)

    return dfD_dX @ dX_dt * dM_dt_reciprocal

# Semi-Major Axis Gradient
def dfD_daCalc(dfD_dX, dX_dt, t, rot_mat, n, a, e, E, nu, mu=MU_EARTH):
    # Inputs:
    #   dfD_dX  - Change in Doppler measurement w.r.t. satellite's cartesian state (1x6 array)
    #   dX_dt   - Change in satellite's cartesian state w.r.t. time (6x1 array)
    #   t       - Time of observation (s)
    #   rot_mat - Perifocal-to-ECI rotation matrix (from COE2RV)
    #   n       - Mean motion (rad/s)
    #   a       - Semi-major axis (m)
    #   e       - Eccentricity
    #   E       - Eccentric anomaly (rad)
    #   nu      - True anomaly (rad)
    #   mu      - Gravitational parameter (m^3/s^2) (Defaults to Earth's mu value)
    # Outputs:
    #   dfD_da  - Change in Doppler shift w.r.t semi-major axis
    # NOTE: For calculations in this function, M is in radians

    nu_arr = np.array([np.cos(nu), np.sin(nu), 0])
    E_arr = np.array([-np.sin(E), np.sqrt(1-e**2) * np.cos(E), 0])

    dr_da_M = (1 - e*np.cos(E)) * rot_mat @ nu_arr
    dv_da_M = -n / (2- 2*e*np.cos(E)) * rot_mat @ E_arr

    dX_da_M = np.concatenate([dr_da_M, dv_da_M])
    dX_dM = dX_dt / n
    dM_da = -3/2 * np.sqrt(mu / (a**5)) * t

    dX_da = dX_da_M + dX_dM * dM_da

    return dfD_dX @ dX_da    

#---------------------------------------------------
# Jacobian Calculations
#---------------------------------------------------
def Jacobian(dfD_dM_0, dfD_da):
    # Create Jacobian Matrix
    # Inputs:
    #   dfD_dM_0 - Initial mean anomaly gradient (scalar or Nx1 array)
    #   dfD_da   - Semi-major axis gradient (scalar or Nx1 array)
    # Outputs:
    #   J - Outputs Nx2 array, where N is the number of observations 

    # Convert scalar values to 1D arrays
    dfD_dM_0 = np.atleast_1d(dfD_dM_0)
    dfD_da = np.atleast_1d(dfD_da)

    N = len(dfD_dM_0)       # Number of entries

    # Create matrix
    J = np.zeros((N, 2))
    for i in range(N):
        J[i] = np.array([dfD_dM_0[i], dfD_da[i]])

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