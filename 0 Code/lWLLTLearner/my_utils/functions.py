from math import factorial
import numpy as np

def nCr(n,r):
    f = factorial
    return f(n) // f(r) // f(n-r)

def is_bijection(arr1, arr2) -> bool:
    """
    Returns if the two arrays are a bijection.
    """
    l = len(arr1)
    if l != len(arr2):
        print("ERROR: The two sequences do not have the same length and can never be bijective to each other!")
        return False, 0

    z = zip(arr1, arr2) 

    bijection_found = False

    for z1 in z:
        for z2 in z:
            if (z1[0] == z2[0]) != (z1[1] == z2[1]):
                bijection_found = True
                break
            
    return bijection_found

def get_full_symm_dist_matrix(n, utri_mat) -> np.array:
    """
    Returns a square symmetric matrix with ones on the diagonal.
    The data is taken from an array which is interpreted as upper triangle matrix values.
    """
    row_ids, col_ids = np.triu_indices(n)
    symm_matrix = np.zeros((n, n), dtype= utri_mat.dtype)
    symm_matrix[row_ids, col_ids] = utri_mat
    symm_matrix[col_ids, row_ids] = utri_mat

    return symm_matrix

def _test_is_bijection():    
    a = np.array([1, 1, 2, 4, 4, 1, 1, 4, 8, 1, 10])
    b = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    c = list(b)
    d = list(b)
    e = list(b)
    f = list(b)

    b[2] = -1 # 0
    c[0] = -1 # 1
    d[1] = -1; d[2] = -1 # 2
    e[1] = -1; e[2] = -1; e[10] = -1 # 3
    f[1] = -1; f[2] = -1; f[10] = -1; f[6] = -1 # 4

    lst = [a, b, c, d, e, f]

    print(a)
    for l in lst:
        print(l)
        is_bij, lvl = is_bijection(a, l)
        print(f"{is_bij}\n")        

if __name__ == "__main__":    
    _test_is_bijection()