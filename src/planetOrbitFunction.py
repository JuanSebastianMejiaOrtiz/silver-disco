import numpy as np
import matplotlib.pyplot as plt

from getPlanetInfo import getCoordinatesFromFile


def householderMethod(points: list[list[float]]) -> np.ndarray | None:
    pointsArray = np.array(points)

    x = pointsArray[:, 0]
    y = pointsArray[:, 1]
    z = pointsArray[:, 2]

    A = np.column_stack([
        x**2, y**2, z**2,
        x*y, x*z, y*z,
        x, y, z,
        np.ones(len(x))
    ])

    # To avoid the trivial solution b gets a small perturbation
    # this way it is not zero but close enough
    b = np.zeros(len(x))
    epsilon = 1e-12
    b[0] = epsilon

    m, n = A.shape
    Q = np.eye(m)  # Orthogonal matrix
    R = A.copy()    # Upper triangular matrix

    for k in range(n):
        x = R[k:, k]

        # Calculation of the Householder vector
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
        v = v / np.linalg.norm(v)

        # Apply the Householder transformation to R
        R[k:, k:] = R[k:, k:] - 2 * np.outer(v, v.T @ R[k:, k:])

        # Apply the Householder transformation to Q
        Q[k:, :] = Q[k:, :] - 2 * np.outer(v, v.T @ Q[k:, :])

    # Solving the system Rx = Q^T b
    # Only using the upper triangular matrix
    R_upper = R[:n, :n]
    QTb = Q.T @ b
    QTb_upper = QTb[:n]

    try:
        x = np.linalg.solve(R_upper, QTb_upper)

        x = x / np.linalg.norm(x)
        return x
    except np.linalg.LinAlgError:
        print("Matrix is singular or ill-conditioned")
        return None


def choleskyMethod(points: list[list[float]]) -> np.ndarray | None:
    pointsArray = np.array(points)

    x = pointsArray[:, 0]
    y = pointsArray[:, 1]
    z = pointsArray[:, 2]

    A = np.column_stack([
        x**2, y**2, z**2,
        x*y, x*z, y*z,
        x, y, z,
        np.ones(len(x))
    ])

    ATA = A.T @ A

    b = np.zeros(len(x))

    # To avoid the trivial solution b gets a small perturbation
    # this way it is not zero but close enough
    epsilon = 1e-12
    b[0] = epsilon
    ATB = A.T @ b

    try:
        L = np.linalg.cholesky(ATA)
        y = np.linalg.solve(L, ATB)
        x = np.linalg.solve(L.T, y)

        # Normalized solution
        x = x / np.linalg.norm(x)
        return x
    except np.linalg.LinAlgError:
        print("Matrix is not positive definite")
        return None


def elipsoide(x, y, z, coeff):
    return (coeff[0]*x**2 + coeff[1]*y**2 + coeff[2]*z**2 +
            coeff[3]*x*y + coeff[4]*x*z + coeff[5]*y*z +
            coeff[6]*x + coeff[7]*y + coeff[8]*z + coeff[9])


def plotElipsoide(coeff, plotRange: list[float] = [-1e9, 1e9], planetName: str = ""):
    x = np.linspace(plotRange[0], plotRange[1], 100)
    y = np.linspace(plotRange[0], plotRange[1], 100)
    X, Y = np.meshgrid(x, y)

    a = coeff[2]
    b = coeff[4]*X + coeff[5]*Y + coeff[8]
    c = (coeff[0]*X**2 + coeff[1]*Y**2 + coeff[3]*X*Y +
         coeff[6]*X + coeff[7]*Y + coeff[9])

    discriminant = b**2 - 4*a*c
    Z1 = (-b + np.sqrt(discriminant)) / (2*a)
    Z2 = (-b - np.sqrt(discriminant)) / (2*a)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z1, alpha=0.5, color='blue')
    ax.plot_surface(X, Y, Z2, alpha=0.5, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('{} orbit'.format(planetName) if planetName else 'Orbit')


def solveForPlanet(planetName, choosenMethod, plotRange: list[float] = [-1e9, 1e9]):
    coordinates: list[list[float]] = []
    try:
        coordinates = getCoordinatesFromFile(
            fileName="./public/planetOrbitData/horizons_results_monthly_{}.txt"
            .format(planetName)
        )
    except EOFError:
        print("Error reading file")
        return

    coefficients: np.ndarray | None = choosenMethod(coordinates)
    if coefficients is None:
        exit(1)

    print("{} function: {}x^2 + {}y^2 + {}z^2 + {}xy + {}xz + {}yz + {}x + {}y + {}z + {}"
          .format(planetName, *coefficients))
    plotElipsoide(coefficients, plotRange, planetName)


print("Using Cholesky method")
solveForPlanet("earth", choleskyMethod)
solveForPlanet("jupiter", choleskyMethod, [-3e9, 3e9])
solveForPlanet("saturn", choleskyMethod, [-1e10, 1e10])

print("\n\nUsing Householder method")
solveForPlanet("earth", householderMethod, [-1e8, 1e8])
solveForPlanet("jupiter", householderMethod)
solveForPlanet("saturn", householderMethod, [-1e10, 1e10])

plt.show()
