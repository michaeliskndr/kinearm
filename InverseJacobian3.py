
import numpy as np
import matplotlib.pyplot as plt
import math
from helper.helperFunction import matrixDH


# D-H Parameter
a1, alpha1, d1 = 0, np.pi/2, 0.251
a2, alpha2, d2 = 0.197, 0, 0
a3, alpha3, d3 = 0, np.pi/2, 0
a4, alpha4, d4 = 0, -np.pi/2, 0.385
a5, alpha5, d5 = 0, np.pi/2, 0
a6, alpha6, t6 = 0, np.pi/2, 0


# Define x y z
x = 0
y = 0
z = 0


# Define xTarget, yTarget, zTarget


def setTargetPosition(xTarget, yTarget, zTarget):
    """

    Considering it's negligible, by using it in the inverse kinematics

    xTarget = Inputting X position
    yTarget = Inputting Y position
    zTarget = Inputting Z position
    """
    targetPosition = np.matrix([xTarget, yTarget, zTarget]).T

    return targetPosition


# Forward Kinematics Methods
def forwardKinematics(Q):
    """
    Input: Angle of each joints
    Output: returning transformation Matrix from T01, T02, T03, T04, T05, T06
    """

    t1, t2, t3, t4, t5, d6 = Q[0, 0], Q[0, 1], Q[0, 2], Q[0, 3], Q[0, 4], Q[0, 5]
    T01 = matrixDH(a1, alpha1, d1, 1.07 + t1)
    T12 = matrixDH(a2, alpha2, d2, np.pi/2 + t2)
    T23 = matrixDH(a3, alpha3, d3, np.pi/2 + t3)
    T34 = matrixDH(a4, alpha4, d4, 0 + t4)
    T45 = matrixDH(a5, alpha5, d5, 0 + t5)
    T56 = matrixDH(a6, alpha6, d6 + 0.0716, t6)

    T02 = T01 * T12
    T03 = T02 * T23
    T04 = T03 * T34
    T05 = T04 * T45
    T06 = T05 * T56

    x = T06[0, 3]
    y = T06[1, 3]
    z = T06[2, 3]

    return T01, T02, T03, T04, T05, T06


def updatePosition(Q):
    """
    Q parameter  is states of robots 
    returning x,y,z position
    """
    T01, T02, T03, T04, T05, T06 = forwardKinematics(Q)

    x = T06[0, 3]
    y = T06[1, 3]
    z = T06[2, 3]

    return x, y, z


def getJacobian(Q):
    """Q parameter is initial states of robots, delta is the robot step 0.005
        Matriks Jacobian = [Jv, Jw].T

        Linear Velocity
        Jv  = np.cross(zn-1,(p-pn-1))

        Angular Velocity
        Jw = zn-1
    """

    # Getting zn-1
    T01, T02, T03, T04, T05, T06 = forwardKinematics(Q)
    z0 = np.matrix([0, 0, 1]).T
    z1 = T01[:3, 2]
    z2 = T02[:3, 2]
    z3 = T03[:3, 2]
    z4 = T04[:3, 2]
    z5 = T05[:3, 2]

    # Getting p, pn-1
    p0 = np.matrix([0, 0, 0]).T

    p1 = T01[:3, 3]
    p2 = T02[:3, 3]
    p3 = T03[:3, 3]
    p4 = T04[:3, 3]
    p5 = np.matrix([0, 0, 0]).T
    p = T06[:3, 3]

    # Getting Linear Velocity
    jv1 = np.cross(z0, (p - p0), axis=0)
    jv2 = np.cross(z1, (p - p1), axis=0)
    jv3 = np.cross(z2, (p - p2), axis=0)
    jv4 = np.cross(z3, (p - p3), axis=0)
    jv5 = np.cross(z4, (p - p4), axis=0)
    jv6 = z5

    # Getting Angular Velocity
    jw1 = z0
    jw2 = z1
    jw3 = z2
    jw4 = z3
    jw5 = z4
    jw6 = p5

    J = np.bmat([[jv1, jv2, jv3, jv4, jv5, jv6],
                 [jw1, jw2, jw3, jw4, jw5, jw6]])

    pseudoJacobian = np.linalg.pinv(J)

    return pseudoJacobian


def achievedPosition(Q, xTarget, yTarget, zTarget):
    """
    Input: Target Position
           Current Position
           Transformation Matrix to get Orientation

    Output: Checking whether the position is already reached or not

    """
    T01, T02, T03, T04, T05, T06 = forwardKinematics(Q)

    # Position Error
    currentX = T06[0, 3]
    currentY = T06[1, 3]
    currentZ = T06[2, 3]

    positionError = np.matrix([[xTarget - currentX],
                               [yTarget - currentY],
                               [zTarget - currentZ]])

    tolerance = 0.00005

    if abs(positionError[0, 0]) <= tolerance and abs(positionError[1, 0]) <= tolerance and abs(positionError[2, 0]) <= tolerance:
        return True


def updateError(Q, xTarget, yTarget, zTarget):
    """
    Input: Target Position
           Current Position
           Transformation Matrix to get Orientation

    Output: Returning 6 x 1 Matrix consisting Error Position and Orientation Error

    """
    T01, T02, T03, T04, T05, T06 = forwardKinematics(Q)

    # Position Error
    currentX = T06[0, 3]
    currentY = T06[1, 3]
    currentZ = T06[2, 3]

    positionError = np.matrix([[xTarget - currentX],
                               [yTarget - currentY],
                               [zTarget - currentZ]])

    # Orientation Error
    r11 = T06[0, 0]
    r22 = T06[1, 1]
    r33 = T06[2, 2]

    r32 = T06[2, 1]
    r23 = T06[1, 2]
    r13 = T06[0, 2]
    r31 = T06[2, 0]
    r21 = T06[1, 0]
    r12 = T06[0, 1]

    nue = np.arccos((r11+r22+r33-1)/2)

    tempErrorMatrix = np.matrix([[r32 - r23],
                                 [r13 - r31],
                                 [r21 - r12]])

    orientationError = (1 / 2 * np.sin(nue)) * tempErrorMatrix

    errorMatrix = np.bmat([[positionError],
                           [orientationError]])

    return errorMatrix


xTarget = -0.16
yTarget = -0.3
zTarget = 0.72

# Q Initial States (theta1, theta2, theta3, theta4, theta5, d6)
# Q = np.matrix([np.radians(0), np.radians(0), np.radians(
#     0), np.radians(0), np.radians(0), 0])


Q = np.matrix([0, 0, 0, 0, 0, 0])

step = 0.0001

if __name__ == "__main__":

    while True:
        J = getJacobian(Q)
        dE = updateError(Q, xTarget, yTarget, zTarget)
        dQ = J * dE * step
        Q = Q + dQ.T 

        if achievedPosition(Q, xTarget, yTarget, zTarget):
            print("Solution Found")
            print(f"The joint state is {Q}")
            break
