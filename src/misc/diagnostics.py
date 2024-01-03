import numpy as np

M_d = 28.8
M_v = 18.0
epsilon = M_v / M_d
g0 = 9.8

def expand(arr):
    
    new_arr = np.zeros((len(arr)+2,))
    new_arr[1:-1] = arr
    new_arr[0] = new_arr[1]
    new_arr[-1] = new_arr[-2]
    
    return new_arr


def getVirtualTemperature(T, q):
    return T * ( 1 + q / epsilon ) / (1 + q)
    

def getBoundaryLayerHeight(u, v, theta, q, z_W, Ri_c=0.25, method='grad', skip=1, debug=False):
   
    if method == 'grad':
        dz_T = z_W[1:] - z_W[:-1]
        dz_W = ( dz_T[1:] + dz_T[:-1] ) / 2
        r = getRichardsonNumber(u, v, theta, q, dz_W)

    elif method == 'bulk':
        r = getBulkRichardsonNumber(u, v, theta, q, z_W, ref=skip,)


    Ri = r['Ri']
    h = findBoundaryLayerWithRi(z_W, Ri, Ri_c = Ri_c, skip=skip)

    if debug:
        return h, r
    else:
        return h

def getBulkRichardsonNumber(u, v, theta, q, z_W, ref=0):

    if len(z_W) != len(u) + 1:
        raise Exception("Length of z_W should be the same as len(u)+1")
    if len(z_W) != len(v) + 1:
        raise Exception("Length of z_W should be the same as len(v)+1")
    if len(z_W) != len(theta) + 1:
        raise Exception("Length of z_W should be the same as len(theta)+1")
    if len(z_W) != len(q) + 1:
        raise Exception("Length of z_W should be the same as len(q)+1")
    
    theta_v = getVirtualTemperature(theta, q)
    dtheta_v = theta_v - theta_v[ref]
    du = u - u[ref]
    dv = v - v[ref]

    theta_v  = (theta_v[1:] + theta_v[:-1] ) /2
    dtheta_v = (dtheta_v[1:] + dtheta_v[:-1] ) /2
    du = (du[1:] + du[:-1]) / 2
    dv = (dv[1:] + dv[:-1]) / 2

    theta_v = expand(theta_v)
    dtheta_v = expand(dtheta_v)
    du = expand(du)
    dv = expand(dv) 

    dU2 = (du**2 + dv**2)
    mask = dU2 == 0
    dU2[mask] = 1.0

    Ri = (g0 / theta_v) * dtheta_v * z_W / dU2
    Ri[mask] = 1.0
    
    return dict(
        Ri = Ri,
        du = du,
        dv = dv,
        theta_v = theta_v,
        dtheta_v = dtheta_v,
    )
        


def getRichardsonNumber(u, v, theta, q, dz_W):

    if len(dz_W) != len(u) - 1:
        raise Exception("Length of dz_W should be len(u) - 1")

    if len(dz_W) != len(v) - 1:
        raise Exception("Length of dz_W should be len(v) - 1")

    if len(dz_W) != len(theta) - 1:
        raise Exception("Length of dz_W should be len(theta) - 1")

    if len(dz_W) != len(q) - 1:
        raise Exception("Length of dz_W should be len(q) - 1")


    theta_v = getVirtualTemperature(theta, q)
    
    theta_v_mid = ( theta_v[1:] + theta_v[:-1] ) / 2

    dtheta_vdz = (theta_v[1:] - theta_v[:-1]) / dz_W
    dudz = (u[1:] - u[:-1]) / dz_W
    dvdz = (v[1:] - v[:-1]) / dz_W
    dvdz = (v[1:] - v[:-1]) / dz_W

    dudz = expand(dudz)
    dvdz = expand(dvdz)
    dtheta_vdz = expand(dtheta_vdz)
    theta_v_mid = expand(theta_v_mid)
    Ri = (g0 / theta_v_mid) * dtheta_vdz / (dudz**2 + dvdz**2)

    return dict(
        Ri = Ri,
        dudz = dudz,
        dvdz = dvdz,
        dtheta_vdz = dtheta_vdz,
        theta_v = theta_v_mid,
    )

def findBoundaryLayerWithRi(z_W, Ri, Ri_c = 0.25, skip=1):
    
    if len(z_W) != len(Ri):
        raise Exception("Length of z_w should be the same as Ri.")

    h = [] #np.zeros((N,))

    if Ri[skip] >= Ri_c:
        h.append(z_W[skip+1])

    else:

        for k in range(skip, len(z_W)-1):
            if Ri[k] < Ri_c and Ri[k+1] >= Ri_c:
                h.append(z_W[k] + (z_W[k+1] - z_W[k]) / (Ri[k+1] - Ri[k])  * ( Ri_c - Ri[k]))

    return h



