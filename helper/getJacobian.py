def Jacobian(q):

 Matrix4d tempMat; //(4x4) matrix

 for i in range(0,dof):

    tempMat = get_T(i)  //transformation matrix from i-1 to i

    Ri    = R[i-1]*tempMat[0:3;0:3] //orientation Matrix
    Ti    = T[i-1]*tempMat

    zi  = Ri[:,2]
    Oi  = Ti[0:3,3]


    for i in range(0,dof):  
    Vector3d tempVec = zi.cross( Oi[dof] - Oi[i] )
    Jmat[0:3,i] = tempVec
    Jmat[3:6,i] = zi

    return Jmat