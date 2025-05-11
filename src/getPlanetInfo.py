# File must be TextIOWrapper
# Can't put the type for the parameter because it needs to be imported
# The only packages that are allowed to be imported are:
# Numpy, Scipy and Matplotlib
def findStartLine(file, startLine: str) -> bool:
    for line in file:
        if startLine in line:
            return True

    return False


def makeCoordinateVector(line: str) -> list[float]:
    equalIndex = line.find('=')
    x = 1
    if line[equalIndex + 1] == '-':
        x = -1
    x *= float(line[equalIndex + 2: line.find(' ', equalIndex + 2)])

    equalIndex = line.find('=', equalIndex + 1)
    y = 1
    if line[equalIndex + 1] == '-':
        y = -1
    y *= float(line[equalIndex + 2: line.find(' ', equalIndex + 2)])

    equalIndex = line.find('=', equalIndex + 1)
    z = 1
    if line[equalIndex + 1] == '-':
        z = -1
    z *= float(line[equalIndex + 2:])

    vector = [x, y, z]
    return vector


def getCoordinatesFromFile(fileName: str, startLine: str, endLine: str) -> list[list[float]]:
    coordinates: list[list[float]] = []
    with open(fileName, encoding='utf-8') as file:
        startFound = findStartLine(file, startLine)

        if not startFound:
            print("Start line not found")
            # TODO: Check if this is the right exception to raise
            raise EOFError

        next(file)

        while True:
            try:
                line = next(file)

                if endLine in line:
                    break

                # Processing coordinates and adding them to the list
                coordinates.append(makeCoordinateVector(line.strip()))

                for _ in range(3):
                    next(file)

            except StopIteration:
                break

    return coordinates
