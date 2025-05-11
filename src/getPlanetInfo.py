def findStartLine(file, startLine):
    for line in file:
        if startLine in line:
            return True

    return False


def makeCoordinateVector(line):
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


def getCoordinatesFromFile(fileName, startLine, endLine):
    coordinates = []
    with open(fileName, encoding='utf-8') as file:
        startFound = findStartLine(file, startLine)

        if not startFound:
            print("No se encontro la linea de inicio")
            return

        next(file)

        while True:
            try:
                line = next(file)

                if endLine in line:
                    break

                # Procesamiento de las coordenadas
                coordinates.append(makeCoordinateVector(line.strip()))

                for _ in range(3):
                    next(file)

            except StopIteration:
                break

    return coordinates


earthCoordinates = getCoordinatesFromFile(
    fileName="public/planetOrbitData/horizons_results_monthly_earth.txt",
    startLine="$$SOE",
    endLine="***"
)

print(earthCoordinates)

'''
jupiterCoordinates = getCoordinatesFromFile(
    fileName="./planetOrbitData/horizons_results_monthly_jupiter.txt",
    startLine="$$SOE",
    endLine="***"
)

saturnCoordinates = getCoordinatesFromFile(
    fileName="./planetOrbitData/horizons_results_monthly_saturn.txt",
    startLine="$$SOE",
    endLine="***"
)
'''
