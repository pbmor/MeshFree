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

def NearestNeighbour(Pts,Ranges,N,r):
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
        print(id)

    return NN


def ModRK3D(nodes,r,order,weight,Ranges):
    
    
    supp = 3*r
    supphat = 0.9*r

    nNodes = len(nodes)
    b, Mins, bN = Pt_IDs(nodes,Ranges,nNodes,r)

    shape_dx = np.zeros(nNodes)
    shape_dy = np.zeros(nNodes)
    shape_dz = np.zeros(nNodes)
    
    phi=0.5+np.sqrt(5./4.)
    Pos = np.zeros((20,3))
    Pos[0,:]=[1.,1.,1.]
    Pos[1,:]=[1.,1.,-1.]
    Pos[2,:]=[1.,-1.,1.]
    Pos[3,:]=[1.,-1.,-1.]
    Pos[4,:]=[-1.,1.,1.]
    Pos[5,:]=[-1.,1.,-1.]
    Pos[6,:]=[-1.,-1.,1.]
    Pos[7,:]=[-1.,-1.,-1.]

    Pos[8,:]=[0.,1./phi,phi]
    Pos[9,:]=[0.,1./phi,-phi]
    Pos[10,:]=[0.,-1./phi,phi]
    Pos[11,:]=[0.,-1./phi,-phi]

    Pos[12,:]=[1./phi,phi,0.]
    Pos[13,:]=[1./phi,-phi,0.]
    Pos[14,:]=[-1./phi,phi,0.]
    Pos[15,:]=[-1./phi,-phi,0.]

    Pos[16,:]=[phi,0.,1./phi]
    Pos[17,:]=[-phi,0.,1./phi]
    Pos[18,:]=[phi,0.,-1./phi]
    Pos[19,:]=[-phi,0.,-1./phi]
    

    for i in range(nNodes):
        print('Node:',i)
        for j in range(20):
            X = (nodes[i,:]+(Pos[j,:]/np.sqrt(sum((Pos[j,k]**2 for k in range(3)))))*r)
            shape = ModRK3DShape(X, nodes,supp,supphat,order,b,Mins,bN,i)
            for k in range(len(shape)):
                shape_dx[i] += shape[k]*Pos[j,0]
                shape_dy[i] += shape[k]*Pos[j,1]
                shape_dz[i] += shape[k]*Pos[j,2]
        for j in range(nNodes):
            shape_dx[i] *= 3/20
            shape_dy[i] *= 3/20
            shape_dz[i] *= 3/20
    
    
    return shape, shape_dx, shape_dy,shape_dz

def ModRK3DShape(node, nodes,supp,supphat,order,b,Mins,bN,i):
    
    nNodes = len(nodes)
    shape = np.zeros((nNodes,nNodes))
    nb = 3+1
    
    X = node
    M = np.zeros((nb,nb))
    Fhat = np.zeros((4,1))
    Local_Points = Local_Pts(nodes,b,Mins,bN,i)
    nLP = len(Local_Points)
    print('NearestNeighbour Length',nLP)
    shape = np.zeros(nLP)
    for j in range(nLP):
        s=Local_Points[j][:]
        Diff = X-s
        DiffMag = np.sqrt(sum((Diff[k]**2 for k in range(3))))
        phi=calc_phi(DiffMag,supp)
        phihat=calc_phi(DiffMag,supphat)
        # Calculation of moment matrix M
        [H,dH,HT]=basis(Diff,order)
        M += (HT.dot(H))*phi
        Fhat += HT*phihat
 
    invM = np.linalg.inv(M)
    
    for j in range(nLP):
        # Calculate different shape functions here
        s=Local_Points[j][:]
        Diff = X-s
        DiffMag = np.sqrt(sum((Diff[k]**2 for k in range(3))))
        phi=calc_phi(DiffMag,supp)
        phihat=calc_phi(DiffMag,supphat)
        [H,dH,HT] = basis(Diff,order)
        [H0,dH0,HT0] = basis(np.zeros(3),order)
        shape[j] = H.dot(invM.dot((HT0-Fhat)))*phi+phihat
    
    return shape

def basis(x,order):
    # Returns the basis and it's derivative at a point x based upon the order
    if order==1:
        
        H  = np.array([[1,x[0],x[1],x[2]]])
        dH = np.array([[0,1,0,0]])

    elif order==2:
        H  = np.array([[1,x[0],x[1]**2]])
        dH = np.array([[0,1,2*x[1]]])
    
    elif order==3:
        H  = np.array([[1,x[0],x[1]**2,  x[2]**3]])
        dH = np.array([[0,   1, 2*x[1],3*x[2]**2]])
    HT = H.transpose()
    return H, dH, HT

def calc_phi(z,a):

    if (z<a):
        phi = 1
    else:
        phi = 0

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
    
    #Define search radius
    r = 0.5

    #Create empty array
    #NN = NearestNeighbour(Pts,Ranges,N,r)

    nNodes = N
    nDOF = nNodes
    PtDiff = np.zeros(N-1)

    dx = 0.5
    # Order of the basis
    order = 1 
    # Support size
    fac = 2.001
    
    smooth=1

    weight = np.ones((len(Pts),1))

    shape, shape_dx, shape_dy,shape_dz = ModRK3D(Pts,r,order,weight,Ranges)


