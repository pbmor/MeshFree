import numpy as np
import os
import sys
import math
import csv
import glob
import nibabel as nib
from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa
import vtk
from statistics import mode
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from math import pi
import warnings
#from paraview.simple import Contour

class BoxSearch:
    def __init__(self,Points,dx,dy=None,dz=None):

        #Define box resolution if not specified
        if dy is None:
            dy,dz = dx, dx

        #Save resolutions to class
        self.dx, self.dy, self.dz = dx,dy,dz
        
        #Find maximum and minimum locations in each direction and save to class
        self._Minx,self._Maxx,self._Miny,self._Maxy,self._Minz,self._Maxz = np.min(Points[:,0]),np.max(Points[:,0]),np.min(Points[:,1]),np.max(Points[:,1]),np.min(Points[:,2]),np.max(Points[:,2])

        #Find and save the centres in each direction
        self._centrex, self._centrey, self._centrez = ((self._Maxx+self._Minx)/2),((self._Maxy+self._Miny)/2),((self._Maxz+self._Minz)/2)

        #Find the number of boxes from the centre that are required to cover the point space
        Ndx = np.ceil(self._Maxx-self._centrex)
        Ndy = np.ceil(self._Maxy-self._centrey)
        Ndz = np.ceil(self._Maxz-self._centrez)

        #Redefine the maximums and minimums to be defined by the box resolution such that tgere is an integer number of boxes in each direction and there is an extra box beyond the last box containing the points
        self._minx = self._centrex - (Ndx+1)*dx
        self._miny = self._centrey - (Ndy+1)*dy
        self._minz = self._centrez - (Ndz+1)*dz
        self._maxx = self._centrex + (Ndx+1)*dx
        self._maxy = self._centrey + (Ndy+1)*dy 
        self._maxz = self._centrez + (Ndz+1)*dz 

        #Find and save the number of boxes in each direction
        self._nx,self._ny,self._nz = int((self._maxx-self._minx)/self.dx),int((self._maxy-self._miny)/self.dy),int((self._maxz-self._minz)/self.dz)
        
        #Find and save the total number of boxes
        self._N = (self._nx)*(self._ny)*(self._nz)
        
        #Create empty arrays for the the points and respective ids
        self._Box = [[] for i in range(self._N)]
        self._PtIDs = [[] for i in range(self._N)]
        
        #Save the number of points
        self.nPts = len(Points)
        
        #Find box id for each point using BoxID function and save the pointds and arrays to the assigned box array
        for i in range(self.nPts):
            bid = self.BoxID(Points[i])
            self._Box[bid].append(Points[i])
            self._PtIDs[bid].append(i)

    def BoxID(self,X,dix=0,diy=0,diz=0):
        '''
        A function to find the box id for a given point
        Input:
            - The point of interest.

        Output:
            - The respective box id
        '''
        #Find the location of the point in the box relative to the box resolution
        ix,iy,iz = (int((X[0]-self._minx)/self.dx)),(int((X[1]-self._miny)/self.dy)),(int((X[2]-self._minz)/self.dz))
        ix += dix
        iy += diy
        iz += diz

        #Return the appropriate box id noting the boxes are listed consecutively
        return ix*self._ny*self._nz + iy*self._nz + iz

    
    def neighb(self,x):
        '''
        A function to find the point ids of all the points in the box and surrounding boxes of a specific point.
        Input:
            - The point of interest.

        Output:
            - A list of the point ids
        '''
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


def ModRK3D(node,nodes,r,supp,supphat,order,boxes,i):
    '''
    A function to find the shape functions and derivatives of the a specific point

    Inputs: nodes - Nodes of the Mesh
            r - radius to define support radii
            order - 
            weight - weighting of nodes (assumed to be constant)
            Ranges - Maximum and Minimum values of the mesh, in the 3 directions
    '''

    #Get number of nodes
    nNodes = len(nodes)
    
    #Define empty array to save derivatives of shape function
    Shape = np.zeros(nNodes)
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
    
    #Normalise the dodecahedron locations
    for j in range(len(Pos)):
        Pos[j,:] = (Pos[j,:]/np.sqrt(sum((Pos[j,k]**2 for k in range(3)))))
    
    print('Finding shape derivatives of node: ',i,' ',node)

    #Get the shape function of the point
    shape_pt = ModRK3DShape(node,nodes,supp,supphat,order,boxes)
    
    #Get shape functions of the points of the dodecahdron surrounding the point of interest
    for j in range(len(Pos)):
        X = (node+(Pos[j,:])*r)
        shape = ModRK3DShape(X,nodes,supp,supphat,order,boxes)
    
        #Sum shape points for each dodecahedron point to find finite 
        #difference definition of derivatives
        shape_dx += shape*Pos[j,0]
        shape_dy += shape*Pos[j,1]
        shape_dz += shape*Pos[j,2]

    #Get derivatives
    Shape = shape_pt
    shape_dx *= 3/(r*len(Pos))
    shape_dy *= 3/(r*len(Pos))
    shape_dz *= 3/(r*len(Pos))
    
    return Shape, shape_dx, shape_dy, shape_dz


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
    nii = nib.load(Fname)
    nii_header = nii.header
    pixdim = nii.header['pixdim']
    Vol = np.prod(pixdim[1:4])

    DMax = np.max(pixdim[1:4])
    DMin = np.min(pixdim[1:4])

    
    os.system('/Applications/ITK-SNAP.app/Contents/bin/c3d ' + Fname + ' -trim 5vox -o temp_vtk.vtk')

    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(ref)
    reader.ReadAllScalarsOn()
    reader.Update()
    new_data = reader.GetOutput()

    thresh = vtk.vtkThreshold()
    thresh.SetInputData(new_data)
    thresh.Scalars = ['POINTS', 'scalars']
    thresh.ThresholdRange = [0.5, 5.0]
    thresh.Update()
    '''

    Fname = 'Point_Clouds/bav01/seg05_to_02_bav01_root_reslice.nii.gz'

    DMax = 0.55
    DMin = 0.48

    List_of_Subdirectories = sorted(glob.glob('./PropagatedPointClouds/*'))
    CommonOfDir = os.path.commonprefix(List_of_Subdirectories)
    for d in List_of_Subdirectories:
        DataDir = d.replace(CommonOfDir,'')
        
        with open('../RunVTK/echoframetime.csv') as csv_file:
            XLData = csv.reader(csv_file, delimiter=',')
            for row in XLData:
                if DataDir == row[0]:
                    DataInfo = row

        #Define Frame Time Length
        FT = DataInfo[1]
        #Define Open Frame and Close Frame
        OF = DataInfo[4]
        CF = DataInfo[5]

        #for 'medial_meshes' in List_of_subdirectories:
        fnames = sorted(glob.glob(d + '/propagated point clouds/*.vtk'))

        #Choose reference frame
        refN = int(DataInfo[4])-1 #Choose frame before valve opens

        common = os.path.commonprefix(fnames)
        for Fname in list(fnames):
            X = Fname.replace(common,'')
            X = X.replace('.vtk','')
            X = np.fromstring(X, dtype=int, sep=' ')
            X=X[0]
            if X==refN:
                ref=Fname
        NX = len(fnames)

        if not fnames:
            print(DataDir," is empty")
        else:
            fdir = os.path.dirname(fnames[0])
            # Check Directory
            if not os.path.exists(fdir):
                print('Error: Path does not exist:', fdir)
                sys.exit()
            if DataInfo[9] == 'n':
                print(DataDir,' is excluded')
            else:
                print(DataDir,' is included')

                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(ref)
                reader.ReadAllScalarsOn()
                reader.Update()
                polydata = reader.GetOutput()
                Pts_ref = vtk_to_numpy(polydata.GetPoints().GetData())
                #print(dir(polydata))
                #Spacing = polydata.GetSpacing()
                
                Vol  = DMin**3
                r    = (Vol*(3/(4*math.pi)))**(1/3)
                supp = 3.0*DMax
                supphat = 0.9*DMin
    
                print('DMax =',DMax)
                print('DMin =',DMin)
                print('Finding boxes...')
                boxes = BoxSearch(Pts_ref,r+supp)
                print('Found')
    
                #Create empty arrays
                nNodes = len(Pts_ref[:,0])
                nDOF = nNodes
                PtDiff = np.zeros(nNodes-1)

                # Order of the basis
                order = 1 

                #Cycle through frames
                for Fname in ref: #fnames:
                    reader = vtk.vtkPolyDataReader()
                    reader.SetFileName(ref)
                    reader.ReadAllScalarsOn()
                    reader.Update()
                    polydata = reader.GetOutput()  
                    Pts = vtk_to_numpy(polydata.GetPoints().GetData())
        
                    #Create empty array for deformation gradient
                    F = np.zeros((3,3))
        
                    for i in range(nNodes):
                        F = np.zeros((3,3))
                        #Get List of local point ids
                        LP_Ids = boxes.neighb(Pts[i])
                        #Get shape functions
                        shape, shape_dx, shape_dy, shape_dz = ModRK3D(Pts_ref[i,:],Pts_ref,r,supp,supphat,order,boxes,i)
                        for j in LP_Ids: 
                            X = Pts[j,:]
                            F[:,0] += shape_dx[j]*X
                            F[:,1] += shape_dy[j]*X
                            F[:,2] += shape_dz[j]*X
            
                        print('Point_',i,':')
                        print('F=',F)
    
