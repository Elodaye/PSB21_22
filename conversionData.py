import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def conversionData():
    try:
        file = open('test2.txt', "r")
    except:
        print("Erreur ecriture fichier.")

    p = []

    line = file.readline()
    while line:
        data = line.split(", ")
        flag = 0

        x = data[0]
        y = data[1]
        label = data[3]
        z = float(data[-1][:-1])

        print(z)

        for i in range(len(p)):
            if p[i][0] == x and p[i][1] == y and p[i][2] == label:
                p[i][3] += z
                p[i][4] += 1
                flag = 1

        if flag == 0:
            p.append([x, y, label, z, 1])

        for i in range(len(p)):
            p[i][3] /= p[i][4]



        line = file.readline()

    file.close()

    print(p)



    for i in range(len(p)):
        try:
            file = open(p[i][2] + "_" + p[i][0] + "_" + p[i][1], "w")

            s = p[i][0] + ", " + p[i][1] + ", " +  str(p[i][3])
            file.write(s)
            file.write("\n")

            file.close()
        except:
            print("Erreur ecriture fichier.")

    return p

