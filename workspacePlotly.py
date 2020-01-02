import numpy as np
import matplotlib.pyplot as plt 
import random
import plotly.graph_objects as go
import numpy as np

# Fungsi Matriks D-H
def matrixDH(a, al, d, t):
    """Generate D-H Matrix with parameter a, alpha, d, tetha.
    Returning the matrix
    """
    return np.matrix([[np.cos(t), -np.sin(t)*np.cos(al), np.sin(t)*np.sin(al), a*np.cos(t)],
                      [np.sin(t), np.cos(t)*np.cos(al), -np.cos(t)*np.sin(al), a*np.sin(t)],
                      [0, np.sin(al), np.cos(al), d],
                      [0, 0, 0, 1]])

#Declare Tmp variable
xStored = []
yStored = []
zStored = []



def generateWorkspace(sampleSize):
    # D-H Parameter
    a1, alpha1, d1 = 0, np.pi/2, 0.251
    a2, alpha2, d2 = 0.197, 0, 0
    a3, alpha3, d3 = 0, np.pi/2, 0
    a4, alpha4, d4 = 0, -np.pi/2, 0.385
    a5, alpha5, d5 = 0, np.pi/2, 0
    a6, alpha6, t6 = 0, np.pi/2, 0

    #Joint Limit
    t1_min, t1_max = -1.57, 1.57
    t2_min, t2_max = -1.57, 1.57
    t3_min, t3_max = -1.57, 1.57
    t4_min, t4_max = -3.14, 3.14
    t5_min, t5_max = -1.57, 1.57
    d6_min, d6_max = 0 + 0.0716, 0.1 + 0.0716
    # Sampling Data Generate
    # N = 5000

    print("Generating Sampling Data")
    t1 = (t1_min + (t1_max-t1_min) * random.random() for i in range(sampleSize))
    t2 = (t2_min + (t2_max-t2_min) * random.random() for i in range(sampleSize))
    t3 = (t3_min + (t3_max-t3_min) * random.random() for i in range(sampleSize))
    t4 = (t4_min + (t4_max-t4_min) * random.random() for i in range(sampleSize))
    t5 = (t5_min + (t5_max-t5_min) * random.random() for i in range(sampleSize))
    d6 = (d6_min + (d6_max-d6_min) * random.random() for i in range(sampleSize))

    t1, t2, t3, t4, t5, d6 = list(t1), list(t2), list(t3), list(t4), list(t5), list(d6)
    print("Finish Generating Data")
    # zlist = []

    # Calculate Forward Kinematics
    print("Calculating Workspace on Progress")

    for i in range(sampleSize):
        A1 = matrixDH(a1, alpha1, d1, t1[i])
        A2 = matrixDH(a2, alpha2, d2, t2[i])
        A3 = matrixDH(a3, alpha3, d3, t3[i])
        A4 = matrixDH(a4, alpha4, d4, t4[i])
        A5 = matrixDH(a5, alpha5, d5, t5[i])
        A6 = matrixDH(a6, alpha6, d6[i], t6)

        T = A1 * A2 * A3 * A4 * A5 * A6
        x = T[0,3]
        y = T[1,3]
        z = T[2,3]

        # zlist.append(z)
        xStored.append(x)
        yStored.append(y)
        zStored.append(z)
        


    print("Finish Calculating")
    # Debug nilai zmax
    # print(max(zlist))


generateWorkspace(30000)

print("Showing Workspace")

# Initiate Graph With Plotly
fig = go.Figure()


fig.add_trace(go.Scatter3d(
    x=xStored,
    y=yStored,
    z=zStored,
    mode="markers",
    marker=dict(
        size=4,
        color=zStored,
        colorscale='Viridis',
        opacity=0.5
    )
))

fig.update_layout(
    title_text = "3D Workspace Arm Robot"
)

fig.show()