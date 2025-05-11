import numpy as np

from getPlanetInfo import getCoordinatesFromFile


# TODO: Implement the Householder Method
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
    pass


# TODO: Implement in a way that avoids the trivial case
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

    b = np.zeros(len(x))

    ATA = A.T @ A
    ATB = A.T @ b

    try:
        L = np.linalg.cholesky(ATA)
        y = np.linalg.solve(L, ATB)
        x = np.linalg.solve(L.T, y)
        return x
    except np.linalg.LinAlgError:
        print("Matrix is not positive definite")
        return None


'''
jupiterCoordinates: list[list[float]] | None = getCoordinatesFromFile(
    fileName="./public/planetOrbitData/horizons_results_monthly_jupiter.txt",
    startLine="$$SOE",
    endLine="***"
)

saturnCoordinates: list[list[float]] | None = getCoordinatesFromFile(
    fileName="./public/planetOrbitData/horizons_results_monthly_saturn.txt",
    startLine="$$SOE",
    endLine="***"
)
'''
earthCoordinates: list[list[float]] = []
try:
    earthCoordinates = getCoordinatesFromFile(
        fileName="./public/planetOrbitData/horizons_results_monthly_earth.txt",
        startLine="$$SOE",
        endLine="***"
    )
except EOFError:
    print("Error reading file")
    exit(1)

coefficients: np.ndarray | None = choleskyMethod(earthCoordinates)
if coefficients is None:
    exit(1)

print(coefficients)

# TODO: Make the function using the coefficients and plot it
