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
import warnings

class BoxSearch:
    def __init__(self,Points,dx,dy=None,dz=None):
        if dy is None:
            dy,dz = dx, dx
        self.dx, self.dy, self.dz = dx,dy,dz
        self._minx,self._maxx,self._miny,self._maxy,self._minz,self._maxz = np.min(Points[:,0]),np.max(Points[:,0]),np.min(Points[:,1]),np.max(Points[:,1]),np.min(Points[:,2]),np.max(Points[:,2])

        self._minx -= dx
        self._miny -= dy
        self._minz -= dz
        self._maxx += dx
        self._maxy += dy
        self._maxz += dz

        self._nx,self._ny,self._nz = int((self._maxx-self._minx)/self.dx),int((self._maxy-self._miny)/self.dy),int((self._maxz-self._minz)/self.dz)
        self._N = (self._nx+2)*(self._ny+2)*(self._nz+2)
        self._Box = [[] for i in range(self._N)]
        self._PtIDs = [[] for i in range(self._N)]
        
        nPts = len(Points)
        
        for i in range(nPts):
            bid = self.BoxID(Points[i])
            self._Box[bid].append(Points[i])
            self._PtIDs[bid].append(i)

    def BoxID(self,X,dix=0,diy=0,diz=0):
        ix,iy,iz = (int((X[0]-self._minx)/self.dx)+1),(int((X[1]-self._miny)/self.dy)+1),(int((X[2]-self._minz)/self.dz)+1)
        ix += dix
        iy += diy
        iz += diz

        return ix*self._ny*self._nz + iy*self._nz + iz

    def neighb(self,x):
        if x[0]<self._minx or x[0]>self._maxx or x[1]<self._miny or x[1]>self._maxy or x[2]<self._minz or x[2]>self._maxz:
            warnings.warn("Point x is out of range of the box")
            return []

        ids = []
        for dix in [-1,0,1]:
            for diy in [-1,0,1]:
                for diz in [-1,0,1]:
                    bid = self.BoxID(x,dix,diy,diz)
                    ids.extend(self._PtIDs[bid])

        return ids


def FindD(Pts,Ranges,r,boxes):
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
    for i in range(3):
        MeshD[i] = Ranges[2*i+1]-Ranges[2*i]
    MaxD = np.max(MeshD)

    #Create 'empty' arrays filled with overestimations for differences between 
    #the points and the minimum (non-zero) between 'adjacent' points
    Diff = (1.1*MaxD)*np.ones((N,N))
    Dx = (1.1*MaxD)*np.ones(N)
    Dy = (1.1*MaxD)*np.ones(N)
    Dz = (1.1*MaxD)*np.ones(N)

    #Get boxes of mesh
    #b, Mins, bN, bIds = Pt_IDs(Pts,Ranges,r)

    for i in range(N):
        # Find local points to reduce
        PointIds =  boxes.neighb(Pts[i,:])
        #Points, PointIds = Local_Pts(Pts,b,bIds,Mins,bN,i,r)
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
        
    return Dx, Dy, Dz

def ModRK3D(nodes,r,order,boxes):
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
    
    #Define empty array to save derivatives of shape function
    Shape = np.zeros((nNodes,nNodes))
    shape_dx = np.zeros((nNodes,nNodes))
    shape_dy = np.zeros((nNodes,nNodes))
    shape_dz = np.zeros((nNodes,nNodes))
    
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

    Pos = np.zeros((6,3))
    Pos[0,:]=[0.,0.,1.]
    Pos[1,:]=[0.,0.,-1.]
    Pos[2,:]=[1.,0.,0.]
    Pos[3,:]=[-1.,0.,0.]
    Pos[4,:]=[0.,1.,0.]
    Pos[5,:]=[0.,-1.,0.]
    
    for j in range(len(Pos)):
        Pos[j,:] = (Pos[j,:]/np.sqrt(sum((Pos[j,k]**2 for k in range(3)))))

    for i in range(nNodes):        
        print('Finding shape derivatives of node: ',i,' ',nodes[i,:])

        #Get the shape function of the point
        shape_pt = ModRK3DShape(nodes[i,:],nodes,supp,supphat,order,boxes)
            
        for j in range(len(Pos)):
            X = (nodes[i,:]+(Pos[j,:])*r)
            shape = ModRK3DShape(X,nodes,supp,supphat,order,boxes)
    
            #Sum shape points for each dodecahedron point to find finite 
            #difference definition of derivatives
        
            shape_dx[i,:] += shape*Pos[j,0]
            shape_dy[i,:] += shape*Pos[j,1]
            shape_dz[i,:] += shape*Pos[j,2]

        #Get derivatives
        Shape[i,:] = shape_pt
        shape_dx[i,:] *= 3/(r*len(Pos))
        shape_dy[i,:] *= 3/(r*len(Pos))
        shape_dz[i,:] *= 3/(r*len(Pos))
    
    return Shape, shape_dx, shape_dy,shape_dz

def ModRK3DShape(node,nodes,supp,supphat,order,boxes):
    '''
    Get shape function for a node

    Inputs:  node -- node of interest
             nodes -- all nodes
             supp -- support radius
             supphat -- support radius for phi_hat
             order --  function
             b -- list of points in box locations
             bIds -- list of point ids in box locations
             Mins -- Minimums of 3 directions
             bN -- Number of boxes in a direction
             i -- index of node of interest

    Outputs: shape -- shape function
             nLP -- Number of Local points
    '''

    #Create empty arrays
    nNodes = len(nodes)
    
    nb = 3+1
    
    M = np.zeros((nb,nb))
    Fhat = np.zeros((4,1))

    #Get local points and the respective ids
    LP_Ids = boxes.neighb(node)
    
    shape = np.zeros(nNodes)
    for j in LP_Ids:
        s = nodes[j,:]
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
    
    for j in LP_Ids:
        # Calculate different shape functions 
        s = nodes[j,:]
        #Define difference vector
        Diff = node-s
        #Get difference magnitude
        DiffMag = np.sqrt(sum((Diff[k]**2 for k in range(3))))
        #if not DiffMag ==0:
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
    '''
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
    
    #Find Dx, Dy, Dz with local point search radius of 1.5
    #boxes = BoxSearch(Pts,1.5)
    #Dx, Dy, Dz = FindD(Pts,Ranges,1.5,boxes)
    '''

    x = np.linspace(0, 10,6)#[0.0,1.0]
    y = np.linspace(0, 10,6)
    z = np.linspace(0, 10,6)
    Pts = np.zeros((216,3))

    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                Pts[k+6*j+36*i,:] = [x[i], y[j], z[k]]

    

    #print(Dx, Dy, Dz)
    #print(np.max([Dx,Dy,Dz]))
    #print(np.mean([Dx,Dy,Dz]))
    r = 5.1#*0.7777777#*np.max([Dx,Dy,Dz])

    boxes = BoxSearch(Pts,r)

    #Create empty arrays
    nNodes = len(Pts[:,0]) 
    nDOF = nNodes
    PtDiff = np.zeros(nNodes-1)

    # Order of the basis
    order = 1 

    #print('Find Shape function and its derivatives...')
    shape, shape_dx, shape_dy, shape_dz = ModRK3D(Pts,r,order,boxes)
    #print('...Done')
    
    #Create empyt array for deformation gradient
    F = np.zeros((3,3))
    
    for i in range(nNodes):
        F = np.zeros((3,3))
        LP_Ids = boxes.neighb(Pts[i])
        #shape, shape_dx, shape_dy, shape_dz = ModRK3D(Pts[i,:],Pts,r,order,boxes,i)
        for j in LP_Ids:
            X = Pts[j,:]
            F[0,:] += shape_dx[i,j]*X
            F[1,:] += shape_dy[i,j]*X
            F[2,:] += shape_dz[i,j]*X
        print('Point_',i,':')
        print('F=',F)
    
    
