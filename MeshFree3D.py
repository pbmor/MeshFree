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
    '''
    A function that uses the brute force approach to find the nearest neighbours 
    of a given point, i.e. the points within a given radius.

    Inputs:  Pt - The point of interest
             Pts - An array of all the nodes in the mesh
             r - The radius of region in which to find the points

    Outputs: l - A list of the nearest points

    Relies on a reasonable estimation of the search radius.
    '''

    #The number of points
    N = len(Pts)

    l = []
    for i in range(N):
        #Find euclidean distance 
        ED = np.sqrt(sum((Pt[k]-Pts[i,k])**2 for k in range(3)))
        if (ED<=r) and not (Pt[:] == Pts[i,:]):
            #Add point to the list if the the ED is within the rdius and it is not 
            #the same point
            l.append(Pts[i][:])

    return l

def Pt_IDs(Pts,Ranges,r):
    '''
    This function assigns each point to a 'box' in the mesh region.

    Inputs: Pts - Array of the nodes of size N x 3, where N is the number of nodes
            Ranges - the maximums and minimums in the 3 directions, as a 1 x 6 array,
                    where the format is [x_min, x_max, y_min, y_max, z_min, z_max]
            r - Size of the box side length that will divide up the region. Requires 
                    an estimate that will create boxes that encompass some point 
                    locations but not too many. 

    Outputs: b - A list of the points in each 'box'
             Mins - The minimum value in the three directions
             bN - The number of boxes in a given direction(the same in every direction)
             bIds - A list of the point ids in each 'box' that will relate to the 
                    original array of points in Pts

    Relies on a reasonable estimation of the point spacing
    '''

    N = len(Pts)

    Diff = np.zeros(3)
    Mins = np.zeros(3)
    Mins[0] = Ranges[0]
    Mins[1] = Ranges[2]
    Mins[2] = Ranges[4]
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
        i = int((Pts[id,0]-Mins[0])/r)
        j = int((Pts[id,1]-Mins[1])/r)
        k = int((Pts[id,2]-Mins[2])/r)

        b[i][j][k].append(Pts[id,:])
        bIds[i][j][k].append(id)

    return b, Mins, bN, bIds

def Local_Pts(Pts,b,bIds,Mins,bN,pt_id,r):
    '''
    Find the points and the respective ids in the surrounding boxes of a given point

    Inputs: Pts - All points in mesh, dimensions N x 3
            b - list of the points assigned to each box
            bIds - list of point ids assigned to each box
            Mins - Minimums of the mesh in x, y, z directions
            pt_id - id of the point of interest
            r - search radius

    Outputs: Points - list of point in the box of the point and surrounding boxes
             Ids - list of point ids in the box of the point and surrounding boxes

    Relies on a reasonable estimation of the point spacing
    '''
    Points = []
    Ids = []
    # Find index for the box that the point is in
    i = int((Pts[pt_id,0]-Mins[0])/r)
    j = int((Pts[pt_id,1]-Mins[1])/r)
    k = int((Pts[pt_id,2]-Mins[2])/r)
    for I in [-1,0,1]:
        for J in [-1,0,1]:
            for K in [-1,0,1]:
                #Get indices of surrounding boxes
                Iid = int(i+I)
                Jid = int(j+J)
                Kid = int(k+K)
                if (0<= Iid <bN) and (0<=Jid <bN) and (0<=Kid<bN):
                    #Save points and the associated indices
                    Points += b[Iid][Jid][Kid]
                    Ids += bIds[Iid][Jid][Kid]

    return Points, Ids

def NearestNeighbour(Pts,Ranges,r):
    '''
    A function that creates a list of the neighbours within a radius for each 
    point in the mesh.

    Inputs: Pts - All points in mesh, dimensions N x 3
            Range - The maximums and minimums in the 3 directions, as a 1 x 6 array,
                    where the format is [x_min, x_max, y_min, y_max, z_min, z_max]
            r - search radius

    Outputs: NN - List of list of neighbours for each point.

    Relies on a reasonable estimation of the point spacing
    '''

    #Number of points
    N = len(Pts[:,0])

    #Assign boxes to the points
    b, Mins, bN, bIds = Pt_IDs(Pts,Ranges,r)

    #Create empty array
    NN = (NN.append[] for i in range(N))

    #Find nearest neighbours of each point
    for id in range(N):
        #Find surrounding points and the associated ids to narrow search
        Points, Ids = Local_Pts(Pts,b,bIds,Mins,bN,id,r)i
        #Find neighbour within radius
        l =  NN_BF(Pts[id,:],Points,r)
        #Add list of nearest points to overall list
        NN[id].append(l)
        print(id)

    return NN

def FindD(Pts,Ranges,r):
    '''
    A function to find the standard distance between node points in the three 
    principle directions.

    Inputs: Pts - Array of the nodes of size N x 3, where N is the number of nodes
            Ranges - the maximums and minimums in the 3 directions, as a 1 x 6 array,
                    where the format is [x_min, x_max, y_min, y_max, z_min, z_max]
            r - box size

    Outputs: Dx, Dy, and Dz -  arrays of the distance to the nearest point of each 
            point, for the 3 directions. All of size Nx1.

    This function relies on the Pt_Ids function (which assigns each point to a 'box' 
    in the mesh region) and the Local_Pts (which finds the points in its own and 
    the surrounding boxes).

    Relies on box size estimation
    '''
    # Number of Points
    N = len(Pts)

    #Find greatest directional difference
    MeshD = np.zeros(3)
    (MeshD[i] = Ranges[2*i+1]-Ranges[2*i] for i in range(3))
    MaxD = np.max(MeshD)

    #Create 'empty' arrays filled with overestimations for differences between 
    #the points and the minimum (non-zero) between 'adjacent' points
    Diff = (1.1*MaxD)*np.ones((N,N))
    Dx = (1.1*MaxD)*np.ones(N)
    Dy = (1.1*MaxD)*np.ones(N)
    Dz = (1.1*MaxD)*np.ones(N)

    #Get boxes of mesh
    b, Mins, bN, bIds = Pt_IDs(Pts,Ranges,r)

    for i in range(N):
        # Find local points to reduce
        Points, PointIds = Local_Pts(Pts,b,bIds,Mins,bN,i,r)
        # Find directional distances between each point with their 'local points' 
        for j in PointIds:
            Diff[i,j] = np.sqrt(sum(((Pts[i,k]-Pts[j,k])**2 for k in range(3))))
        
        # Get the 27 closest points and respective ids (with itself excluded)
        ClosestIds = np.argpartition(Diff[i,:],27)[:27]
        Closest = Diff[i,np.argpartition(Diff[i,:],27)[:27]]
        #Find minimum non-zero distance in each direction, when the point is 
        #approxiamtely aligned with orthogonally adjacent points, to 4 dp
        #For example the closest point in the x direction when the y and z are 
        #approximately equal
        for j in ClosestIds:
            if (Pts[i,1]-Pts[j,1]<0.4) and (Pts[i,2]-Pts[j,2]<0.4) and not (i==j):
                #overwrite the existing 'closest point' distance'
                if (Dx[i]>abs(Pts[i,0]-Pts[j,0])) and (abs(Pts[i,0]-Pts[j,0])>0.2):
                    Dx[i] = round(abs(Pts[i,0]-Pts[j,0]),4)
            if (Pts[i,0]-Pts[j,0]<0.4) and (Pts[i,2]-Pts[j,2]<0.4) and not (i==j):
                if (Dy[i]>abs(Pts[i,0]-Pts[j,1])) and (abs(Pts[i,1]-Pts[j,1])>0.2):
                    Dy[i] = round(abs(Pts[i,1]-Pts[j,1]),4)
            if (Pts[i,0]-Pts[j,0]<0.4) and (Pts[i,1]-Pts[j,1]<0.4) and not (i==j):
                if (Dz[i]>abs(Pts[i,2]-Pts[j,2])) and (abs(Pts[i,2]-Pts[j,2])>0.2):
                    Dz[i] = round(abs(Pts[i,2]-Pts[j,2]),4)
        
        #Plot examples of the point, with the closest points and approximate directional
        #ellipses, from dx, dy and dz estimations
        if i in [1,50,199]:
            print('dx =',dx)
            print('dy =',dy)
            print('dz =',dz)
            t = np.linspace(0, 2*pi, 100)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(Pts[ClosestIds,0],Pts[ClosestIds,1],Pts[ClosestIds,2],'*')
            ax.plot3D(Pts[i,0],Pts[i,1],Pts[i,2],'r.')
            ax.plot3D(Pts[i,0]+D[i]x*np.cos(t),Pts[i,1]+Dy[i]*np.sin(t),Pts[i,2]*np.ones(100))
            ax.plot3D(Pts[i,0]+D[i]x*np.cos(t),Pts[i,1]*np.ones(100),Pts[i,2]+Dz[i]*np.sin(t))
            ax.plot3D(Pts[i,0]*np.ones(100),Pts[i,1]+Dy[i]*np.cos(t),Pts[i,2]+Dz[i]*np.sin(t))
            ax.set_xlabel('x',size=14)
            ax.set_ylabel('y',size=14)
            ax.set_zlabel('z',size=14)
            ax.set_title('Closest Points',size=14)
            plt.show()

    return Dx, Dy, Dz

def ModRK3D(nodes,r,order,weight,Ranges):
    '''
    A function to find the shape functions and derivatives of the points in a mesh

    Inputs: nodes - Nodes of the Mesh
            r - radius to define support radii
            order - 
            weight - weighting of nodes (assumed to be constant)
            Ranges - Maximum and Minimum values of the mesh, in the 3 directions
    '''
    
    #Define support radii
    supp = 3*r
    supphat = 0.9*r

    #Get number of nodes
    nNodes = len(nodes)
    
    #Assign boxes of mesh
    b, Mins, bN, bIds = Pt_IDs(nodes,Ranges,r)
    
    #Define empty array to save derivatives of shape function
    shape_dx = np.zeros(nNodes)
    shape_dy = np.zeros(nNodes)
    shape_dz = np.zeros(nNodes)
    
    #Define points of dodecahedron with centre (0,0,0)
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
            #Get the dodecahedron point around the point
            X = (nodes[i,:]+(Pos[j,:]/np.sqrt(sum((Pos[j,k]**2 for k in range(3)))))*r)
            #Get the shape function of the point
            shape = ModRK3DShape(X, nodes,supp,supphat,order,b,bIds,Mins,bN,i)
            #Sum shape points for each dodecahedron point to find finite 
            #difference definition of derivatives
            for k in range(len(shape)):
                shape_dx[i] += shape[k]*Pos[j,0]
                shape_dy[i] += shape[k]*Pos[j,1]
                shape_dz[i] += shape[k]*Pos[j,2]
        #Get derivatives
        shape_dx[i] *= 3/20
        shape_dy[i] *= 3/20
        shape_dz[i] *= 3/20
    
    return shape, shape_dx, shape_dy,shape_dz

def ModRK3DShape(node,nodes,supp,supphat,order,b,bIds,Mins,bN,i):
    '''
    Get shape function for a node

    Inputs: node - node of interest
            nodes - all nodes
            supp - support radius
            supphat - support radius for phi_hat
            order -  function
            b - list of points in box locations
            bIds - list of point ids in box locations
            Mins - Minimums of 3 directions
            bN - Number of boxes in a direction
            i - index of node of interest

    Outputs: shape function
    '''

    #Create empty arrays
    nNodes = len(nodes)
    shape = np.zeros((nNodes,nNodes))
    M = np.zeros((nb,nb))
    Fhat = np.zeros((4,1))
    nb = 3+1
    
    #Get local points and the respective ids
    Local_Points, LP_Ids = Local_Pts(nodes,b,bIds,Mins,bN,i,r)

    #Number of local points
    nLP = len(Local_Points)
    shape = np.zeros(nLP)
    for j in range(nLP):
        s = Local_Points[j][:]
        #Define difference vector
        Diff  = node-s
        #Define difference magnitude
        DiffMag = np.sqrt(sum((Diff[k]**2 for k in range(3))))
        #Define phi and phi_hat
        phi=calc_phi(DiffMag,supp)
        phihat=calc_phi(DiffMag,supphat)
        #Calculation of moment matrix M
        [H,dH,HT]=basis(Diff,order)
        M += (HT.dot(H))*phi
        Fhat += HT*phihat
 
    #Get inverse of M
    invM = np.linalg.inv(M)
    
    for j in range(nLP):
        # Calculate different shape functions 
        s=Local_Points[j][:]
        #Define difference vector
        Diff = node-s
        #Get difference magnitude
        DiffMag = np.sqrt(sum((Diff[k]**2 for k in range(3))))
        #Define phi and phi_hat
        phi=calc_phi(DiffMag,supp)
        phihat=calc_phi(DiffMag,supphat)
        #Define H matrices
        [H,dH,HT] = basis(Diff,order)
        [H0,dH0,HT0] = basis(np.zeros(3),order)
        #Save shape function
        shape[j] = H.dot(invM.dot((HT0-Fhat)))*phi+phihat
    
    return shape

def basis(x,order):
    '''
    Define basis H and associated derivative   

    Inputs: x - difference vector
            order - order of the basis

    Outputs: The basis H, its derivative and its transpose
    '''
    # Returns the basis and it's derivative at a point x based upon the order
    if order==1:    
        H  = np.array([[1,x[0],x[1],x[2]]])
        dH = np.array([[0,1,0,0]])
    else:
        print('Warning orders that are greater than 1 are not defined')
        H, dH = np.zeros(4)
    HT = H.transpose()
    return H, dH, HT

def calc_phi(z,a):
    '''
    Calculate phi

    Inputs: z - difference vector magnitude
            a - supp radius

    Outputs: phi
    '''
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
    
    Dx, Dy, Dz = FindD(Pts,Ranges,r)


    #Define search radius
    r = 0.5

    #Create empty array
    #NN = NearestNeighbour(Pts,Ranges,r)

    nNodes = len(Pts[:,0]) 
    nDOF = nNodes
    PtDiff = np.zeros(nNodes-1)

    # Order of the basis
    order = 1 
    # Support size
    fac = 2.001

    weight = np.ones((len(Pts),1))

    shape, shape_dx, shape_dy,shape_dz = ModRK3D(Pts,r,order,weight,Ranges)


