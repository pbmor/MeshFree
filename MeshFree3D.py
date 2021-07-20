import numpy as np
import os
import sys
import math
from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa
import vtk

def NN_BF(Pt,Pts,r):
    N = len(Pts)

    SED = np.zeros(N)

    l = []
    for i in range(N):
        if np.array_equal(Pt[:], Pts[i][:]):
            SED[i]=10000000
        else:
            SED[i] = np.sqrt(sum((Pt[k]-Pts[i][k])**2 for k in range(3)))
        if SED[i]<=r:
            l.append(Pts[i][:])

    return l

def Pt_IDs(Pts,Ranges,N,r):

    Diff = np.zeros(3)
    Mins = np.zeros((1,3))
    Mins[0,0] = Ranges[0]
    Mins[0,1] = Ranges[2]
    Mins[0,2] = Ranges[4]

    for i in range(3):
        Diff[i] = Ranges[2*i+1] - Ranges[2*i]

    bN = int(np.max(Diff)/r)+1
    N = len(Pts[:,0])

    bIDi = np.zeros(N)
    bIDj = np.zeros(N)
    bIDk = np.zeros(N)
    b = [[[[] for i in range(bN)] for j in range(bN)] for k in range(bN)]

    for id in range(N):
        i = int((Pts[id,0]-Mins[0,0])/r)
        j = int((Pts[id,1]-Mins[0,1])/r)
        k = int((Pts[id,2]-Mins[0,2])/r)

        b[i][j][k].append(Pts[id,:])

    return b, Mins, bN

def Local_Pts(Pts,b,Mins,bN,id):

    Points = []
    i = int((Pts[id,0]-Mins[0,0])/r)
    j = int((Pts[id,1]-Mins[0,1])/r)
    k = int((Pts[id,2]-Mins[0,2])/r)
    for I in [-1,0,1]:
        for J in [-1,0,1]:
            for K in [-1,0,1]:
                Iid = int(i+I)
                Jid = int(j+J)
                Kid = int(k+K)
                if (0<= Iid <bN) and (0<=Jid <bN) and (0<=Kid<bN):
                    Points += b[Iid][Jid][Kid]
    return Points

def ModRK3D(nodes,supp,supphat,order,weight):
    
    nNodes = len(nodes)
    shape = ModRK3DShape(nodes, points)
    
    #Create empty dshape array
    dshape = np.zeros((nNodes,nNodes))
    
    dshape[0,:] = np.zeros((1,nNodes))

    for i in range(1,nNodes):
        dhsape[i,:] = np.zeros((1,nNodes))

    dshape[nNodes-1,:] = np.zeros((1,nNodes))

    return shape, dshape

def ModRK3DShape(nodes, points,supp,supphat,order):
    
    nNodes = len(nodes)
    nPoints = len(points)
    shape = np.zeros((nPoints,nNodes))
    nb = order+1
    for i in range(nPoints):
        X=points[i]
        M = np.zeros((nb,nb))
        Fhat = np.zeros((nb,1))
        for j in range(Nodes):
            s=nodes[j]
            if(abs(X-s)>supp):
                phi=calc_phi(X-s,supp)
                phihat=calc_phi(X-s,supphat)
                # Calculation of moment matrix M
                [H,dH]=basis(X-s,order)
                M += np.transpose(H)*H*phi
                Fhat += (np.transpose(H))*phihat
    
        invM = inv(M)
    
    for j in range(nNodes):
        # Calculate different shape functions here
        s=nodes(j)
        phi=calc_phi(X-s,supp)
        phihat=calc_phi(X-s,supphat)
        [H,dH] = basis(X-s,order)
        H0 = basis(0,order)
        
        shape[i,j] = np.matmul(H,np.matmul(invM,(np.transpose(H0)-Fhat)))*phi+phihat

    return shape

def basis(x,order):
    # Returns the basis and it's derivative at a point x based upon the order
    if order==1:
        H = [1,x]
        dH = [0,1]

    elif order==2:
        H = [1,x,x**2]
        dH = [0,1,2*x]

    return H, dH

def calc_phi(X,a):
    # Returns the weight function at point X with support size a
    z=abs(X)/a

    if (z>1):
        phi = 0
    
    if (z>0.5):
        phi=2*(1-z)**3

    phi = 1-6*z^2+6*z**3;

    return phi


if __name__=='__main__':

    #Get points
    Fname = '../RunVTK/Point_Clouds/bav02/propagated point clouds/seg05_to_01_bav02_root_pointcloud.vtk'
 
    # Read the source file.
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(Fname)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    ######################################
    # Define File Data
    polydata  = reader.GetOutput()
    Pts = vtk_to_numpy(polydata.GetPoints().GetData())
    Ranges = np.array(polydata.GetPoints().GetBounds())
    N = len(Pts[:,0])
    print(N)
    #Define search radius
    r = 0.5
    
    b, Mins, bN = Pt_IDs(Pts,Ranges,N,r)

    #Create empty array
    NN = []
    for i in range(N):
        NN.append([])

    #Find nearest neighbours of each point
    for id in range(N):
        Points = Local_Pts(Pts,b,Mins,bN,id)
        l =  NN_BF(Pts[id][:],Points,r)
        NN[id].append(l)

    nNodes = N
    nDOF = nNodes
    PtDiff = np.zeros(N-1)
    for i in range(2,N-1):
        PtDiff[i] = np.abs(np.sqrt((Pts[0,0]-Pts[i,0])**2+(Pts[0,1]-Pts[i,1])**2+(Pts[0,2]-Pts[i,2])**2))
        print(PtDiff[i])
    dx = np.min(PtDiff)
    
    print(dx)
    
    # Order of the basis
    order = 1
    # Support size
    fac = 2.001
    supp = fac*order*dx
    supphat = 0.9*dx
    smooth=1


    shape, dshape = ModRK3D(Pts,supp,supphat,order,weight)


