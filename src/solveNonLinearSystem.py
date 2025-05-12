import numpy as np
from scipy.optimize import root
import time


def fun(x):
    S = sum(x)
    R = 4.056734
    return [
        x[0] + x[3] - 3,
        2*x[0] + x[1] + x[3] + x[6] + x[7] + x[8] + 2*x[9] - R - 10,
        2*x[1] + 2*x[4] + x[5] + x[6] - 8,
        2*x[2] + x[4] - 4*R,
        x[0]*x[4] - 0.193*x[1]*x[3],
        x[5]*np.sqrt(x[1]) - 0.002597*np.sqrt(x[1]*x[3]*S),
        x[6]*np.sqrt(x[3]) - 0.003448*np.sqrt(x[0]*x[3]*S),
        x[3]*x[7] - 0.00001799*x[1]*S,
        x[3]*x[8] - 0.0002155*x[0]*np.sqrt(x[2]*S),
        (x[3]**2)*(x[9] - 0.00003846*S)
    ]


def jac(x):
    S = sum(x)
    J = np.zeros((10, 10))

    # Equation 1
    J[0, 0] = 1
    J[0, 3] = 1

    # Equation 2
    J[1, 0] = 2
    J[1, 1] = 1
    J[1, 3] = 1
    J[1, 6] = 1
    J[1, 7] = 1
    J[1, 8] = 1
    J[1, 9] = 2

    # Equation 3
    J[2, 1] = 2
    J[2, 4] = 1
    J[2, 5] = 1
    J[2, 6] = 1

    # Equation 4
    J[3, 2] = 2
    J[3, 4] = 1

    # Equation 5
    J[4, 0] = x[4]
    J[4, 1] = -0.193*x[3]
    J[4, 3] = -0.193*x[1]
    J[4, 4] = x[0]

    # Equation 6
    for i in range(10):
        J[5, i] = -0.002597*np.sqrt(x[1]*x[3])*(1/(2*np.sqrt(S)))
    k = 1
    factor = 1/(2*np.sqrt(x[k]))
    J[5, 1] = x[5]*factor - 0.002597*np.sqrt(x[3])*(factor*np.sqrt(S) + np.sqrt(x[k])*1/(2*np.sqrt(S)))

    k = 3
    factor = 1/(2*np.sqrt(x[k]))
    J[5, 3] = -0.002597*np.sqrt(x[1])*(factor*np.sqrt(S) + np.sqrt(x[k])*1/(2*np.sqrt(S)))
    J[5, 5] = np.sqrt(x[1]) - 0.002597*np.sqrt(x[1]*x[3])*(1/(2*np.sqrt(S)))

    # Equation 7
    for i in range(10):
        J[6, i] = -0.003448*np.sqrt(x[0]*x[3])*(1/(2*np.sqrt(S)))
    k = 0
    factor = 1/(2*np.sqrt(x[k]))
    J[6, 0] = -0.003448*np.sqrt(x[3])*(factor*np.sqrt(S) + np.sqrt(x[k])*1/(2*np.sqrt(S)))

    k = 3
    factor = 1/(2*np.sqrt(x[k]))
    J[6, 3] = x[6]*factor - 0.003448*np.sqrt(x[0]*x[3])*(factor*np.sqrt(S) + np.sqrt(x[k])*1/(2*np.sqrt(S)))
    J[6, 6] = np.sqrt(x[3]) - 0.003448*np.sqrt(x[0]*x[3])*(1/(2*np.sqrt(S)))

    # Equation 8
    for i in range(10):
        J[7, i] = -0.00001799*x[1]
    J[7, 1] = -0.00001799*(x[1] + S)
    J[7, 3] = x[7] - 0.00001799*x[1]
    J[7, 7] = x[3] - 0.00001799*x[1]

    # Equation 9
    for i in range(10):
        J[8, i] = -0.0002155*x[0]*np.sqrt(x[2])*(1/(2*np.sqrt(S)))
    J[8, 0] = -0.0002155*(np.sqrt(x[2]*S)+x[0]*np.sqrt(x[2])/(2*np.sqrt(S)))

    k = 2
    factor = 1/(2*np.sqrt(x[k]))
    J[8, 2] = -0.0002155*x[0]*(factor*np.sqrt(S) + np.sqrt(x[k])*1/(2*np.sqrt(S)))
    J[8, 3] = x[8] - 0.0002155*x[0]*np.sqrt(x[2])*1/(2*np.sqrt(S))
    J[8, 8] = x[3] - 0.0002155*x[0]*np.sqrt(x[2])*1/(2*np.sqrt(S))

    # Equation 10
    for i in range(10):
        J[9, i] = -0.00003846*(x[3]**2)
    J[9, 3] = 2*x[3]*(x[9] - 0.00003846*S) - 0.00003846*(x[3]**2)
    J[9, 9] = (x[3]**2)*(1 - 0.00003846)

    return J


def benchmark(x, fun, method, jac):
    try:
        time1 = time.perf_counter()
        print("Solution:", root(fun, x, method=method))
        time2 = time.perf_counter()
        print("Time:", time2-time1)
        print("\n")
    except:
        print("Error in method:", method)

    try:
        time1 = time.perf_counter()
        print("Solution with Jacobian:", root(fun, x, jac=jac, method=method))
        time2 = time.perf_counter()
        print("Time:", time2-time1)
        print("\n\n")
    except:
        print("Error in method with Jacobian:", method)


x = np.ones(10)
benchmark(x, fun, "hybr", jac)
benchmark(x, fun, "lm", jac)
benchmark(x, fun, "broyden1", jac)
benchmark(x, fun, "broyden2", jac)
benchmark(x, fun, "anderson", jac)
benchmark(x, fun, "linearmixing", jac)
benchmark(x, fun, "diagbroyden", jac)
benchmark(x, fun, "excitingmixing", jac)
benchmark(x, fun, "krylov", jac)
benchmark(x, fun, "df-sane", jac)

print("Finished")
