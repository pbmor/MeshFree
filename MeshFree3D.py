import numpy as np
import os
import sys
import math
from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa
import vtk
from statistics import mode
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from math import pi

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
    bIds = [[[[] for i in range(bN)] for j in range(bN)] for k in range(bN)]

    for id in range(N):
        i = int((Pts[id,0]-Mins[0,0])/r)
        j = int((Pts[id,1]-Mins[0,1])/r)
        k = int((Pts[id,2]-Mins[0,2])/r)

        b[i][j][k].append(Pts[id,:])
        bIds[i][j][k].append(id)

    return b, Mins, bN, bIds

def Local_Pts(Pts,b,bIds,Mins,bN,id,r):

    Points = []
    Ids = []
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
                    Ids += bIds[Iid][Jid][Kid]

    return Points, Ids

def NearestNeighbour(Pts,Ranges,N,r):
    b, Mins, bN, bIds = Pt_IDs(Pts,Ranges,N,r)

    #Create empty array
    NN = []
    for i in range(N):
        NN.append([])

    #Find nearest neighbours of each point
    for id in range(N):
        Points, Ids = Local_Pts(Pts,b,bIds,Mins,bN,id,r)
        l =  NN_BF(Pts[id][:],Points,r)
        NN[id].append(l)
        print(id)

    return NN


def ModRK3D(nodes,r,order,weight,Ranges):
    
    
    supp = 3*r
    supphat = 0.9*r

    nNodes = len(nodes)
    b, Mins, bN, bIds = Pt_IDs(nodes,Ranges,nNodes,r)

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
    Local_Points, LP_Ids = Local_Pts(nodes,b,bIds,Mins,bN,i,r)
    nLP = len(Local_Points)
    print('NearestNeighbour Length',nLP)
    shape = np.zeros(nLP)
    for j in range(nLP):
        s = Local_Points[j][:]
        Diff  = X-s
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
    
    N = len(Pts)
    Diff = 200*np.ones((N,N))
    Dx = 200*np.ones(N)
    Dy = 200*np.ones(N)
    Dz = 200*np.ones(N)
    b, Mins, bN, bIds = Pt_IDs(Pts,Ranges,N,1.5)
    
    for i in range(N):
        Points, PointIds = Local_Pts(Pts,b,bIds,Mins,bN,i,1.5)
        
        for j in PointIds:
            print('i=',i,'j=',j)
            Diff[i,j] = np.sqrt(sum(((Pts[i,k]-Pts[j,k])**2 for k in range(3))))
        
        ClosestIds = np.argpartition(Diff[i,:],27)[:27]
        Closest = Diff[i,np.argpartition(Diff[i,:],27)[:27]]
        print(ClosestIds)
        print(Pts[ClosestIds,:])        
        for j in ClosestIds:
            #print('i=',i,'j=',j)
            if (Pts[i,1]-Pts[j,1]<0.4) and (Pts[i,2]-Pts[j,2]<0.4) and not (i==j):
                if (Dx[i]>abs(Pts[i,0]-Pts[j,0])) and (abs(Pts[i,0]-Pts[j,0])>0.2):
                    Dx[i] = round(abs(Pts[i,0]-Pts[j,0]),4)
                    dx =Dx[i]
            if (Pts[i,0]-Pts[j,0]<0.4) and (Pts[i,2]-Pts[j,2]<0.4) and not (i==j):
                if (Dy[i]>abs(Pts[i,0]-Pts[j,1])) and (abs(Pts[i,1]-Pts[j,1])>0.2):
                    Dy[i] = round(abs(Pts[i,1]-Pts[j,1]),4)
                    dy = Dy[i]
            if (Pts[i,0]-Pts[j,0]<0.4) and (Pts[i,1]-Pts[j,1]<0.4) and not (i==j):
                if (Dz[i]>abs(Pts[i,2]-Pts[j,2])) and (abs(Pts[i,2]-Pts[j,2])>0.2):
                    Dz[i] = round(abs(Pts[i,2]-Pts[j,2]),4)
                    dz = Dz[i]
        if i in [1,50,199,400,5000,60000]:
            print('dx =',dx)
            print('dy =',dy)
            print('dz =',dz)
            t = np.linspace(0, 2*pi, 100)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(Pts[ClosestIds,0],Pts[ClosestIds,1],Pts[ClosestIds,2],'*')
            ax.plot3D(Pts[i,0],Pts[i,1],Pts[i,2],'r.')
            ax.plot3D(Pts[i,0]+dx*np.cos(t),Pts[i,1]+dy*np.sin(t),Pts[i,2]*np.ones(100))
            ax.plot3D(Pts[i,0]+dx*np.cos(t),Pts[i,1]*np.ones(100),Pts[i,2]+dz*np.sin(t))
            ax.plot3D(Pts[i,0]*np.ones(100),Pts[i,1]+dy*np.cos(t),Pts[i,2]+dz*np.sin(t))
            ax.set_xlabel('x',size=14)
            ax.set_ylabel('y',size=14)
            ax.set_zlabel('z',size=14)
            #ax.title('Point distribution',size=14)
            plt.show()

    
    ClosestIds = [1220,  159,  197,  198,  199,  237,  201,  158, 1258,  196, 1219,  200,  236,  1221,  161,  157, 2356, 1188,  235,  239, 1257, 1259,  160,  156,  238,  234, 1187]
    #print(ClosestIds)
    #print(Pts[ClosestIds,:])
    t = np.linspace(0, 2*pi, 100)
    dx = 0.5
    dy = 0.5
    dz = 0.5
    ax = plt.axes(projection='3d')
    ax.plot3D(Pts[ClosestIds,0],Pts[ClosestIds,1],Pts[ClosestIds,2],'*')
    ax.plot3D(Pts[199,0],Pts[199,1],Pts[199,2],'r.')
    ax.plot3D(Pts[199,0]+dx*np.cos(t),Pts[199,1]+dy*np.sin(t),Pts[199,2]*np.ones(100))
    ax.plot3D(Pts[199,0]+dx*np.cos(t),Pts[199,1]*np.ones(100),Pts[199,2]+dz*np.sin(t))
    ax.plot3D(Pts[199,0]*np.ones(100),Pts[199,1]+dy*np.cos(t),Pts[199,2]+dz*np.sin(t))
    ax.set_xlabel('x',size=14)
    ax.set_ylabel('y',size=14)
    ax.set_zlabel('z',size=14)

    #ax.title('Point distribution',size=14)
    plt.show()
    ''' 
    for i in range(5000):
        #if (Pts[i,0]<=0.1*Ranges[0])and(Pts[i,1]<=0.1*Ranges[2]):
        figure(num=1)
        plt.plot(Pts[i,0],Pts[i,1],'*')
        plt.title('Point distribution',size=14)
        plt.xlabel('x',size=14)
        plt.ylabel('y',size=14)

        figure(num=2)
        plt.plot(Pts[i,0],Pts[i,2],'*')
        plt.title('Point distribution',size=14)
        plt.xlabel('x',size=14)
        plt.ylabel('z',size=14)

        figure(num=3)
        plt.plot(Pts[i,1],Pts[i,2],'*')
        plt.title('Point distribution',size=14)
        plt.xlabel('y',size=14)
        plt.ylabel('z',size=14)
    
    plt.show()
    '''
    #Define search radius
    r = 0.5

    #Create empty array
    #NN = NearestNeighbour(Pts,Ranges,N,r)

    nNodes = len(Pts[:,0]) 
    nDOF = nNodes
    PtDiff = np.zeros(nNodes-1)

    dx = 0.5
    # Order of the basis
    order = 1 
    # Support size
    fac = 2.001
    
    smooth=1

    weight = np.ones((len(Pts),1))

    #shape, shape_dx, shape_dy,shape_dz = ModRK3D(Pts,r,order,weight,Ranges)


