import numpy as np
import os
import sys
import math
import glob
import concurrent.futures
from itertools import repeat
from tqdm import tqdm
import nibabel as nib
from vtk.util.numpy_support import vtk_to_numpy
import vtk
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

def OrderList(flist,N,ref):
    '''
    Reorders the list so that the first file is the reference file, and returns new list of filenames and their IDs 
    Keyword arguments:
    flist -- list of filenames
    N -- number of files
    ref -- reference file name
    For example: for a list [file1.vtk, file2.vtk, file3.vtk, file4.vtk, file7.vtk, file8.vtk] and ref = file3.vtk
    it will return [file3.vtk file4.vtk file7.vtk file8.vtk file1.vtk file2.vtk], [3 4 7 8 1 2], and 3
    '''
    # Order filenames so that reference frame goes first
    Fno = np.zeros(N)

    FListOrdered = [None]*N
    common = os.path.commonprefix(flist)
    for i, Fname in enumerate(flist):
        X = Fname.replace(common,'')
        X = X.replace('.vtk','')
        X = np.fromstring(X, dtype=int, sep=' ')
        #Get list of frame labels
        Fno[i] = X
    # Sort fname labels
    Fno.sort()

    # Get list of file names in new order
    for i,F in enumerate(Fno):
        for Fname in flist:
            X = Fname.replace(common,'')
            X = X.replace('.vtk','')
            X = np.fromstring(X, dtype=int, sep=' ')
            if X ==F:
                FListOrdered[i] = Fname

    return FListOrdered

def ModRK3D(node,nodes,r,supp,supphat,order,boxes,Pos):
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
    
    #Get List of local point ids
    LP_Ids = boxes.neighb(node)
    
    #Define empty array to save derivatives of shape function
    shape_dx = np.zeros(nNodes)
    shape_dy = np.zeros(nNodes)
    shape_dz = np.zeros(nNodes)

    #Get the shape function of the point
    shape_pt = ModRK3DShape(node,nodes,supp,supphat,order,LP_Ids)
    
    #Get shape functions of the points of the dodecahdron surrounding the point of interest
    for j in range(len(Pos)):
        X = (node+(Pos[j,:])*r)
        shape = ModRK3DShape(X,nodes,supp,supphat,order,LP_Ids)
    
        #Sum shape points for each dodecahedron point to find finite 
        #difference definition of derivatives
        shape_dx += shape*Pos[j,0]
        shape_dy += shape*Pos[j,1]
        shape_dz += shape*Pos[j,2]

    #Get derivatives
    shape_dx *= 3/(r*len(Pos))
    shape_dy *= 3/(r*len(Pos))
    shape_dz *= 3/(r*len(Pos))
    
    return shape_pt, shape_dx, shape_dy, shape_dz, LP_Ids


def ModRK3DShape(node,nodes,supp,supphat,order,LP_Ids):
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


def FindFandInvariants(Pts, Shape_dx, Shape_dy, Shape_dz, LocalPoints_Ids, nNodes):

    #Create empty arrays
    F = np.zeros((nNodes,3,3))
    I1 = np.zeros(nNodes)
    I2 = np.zeros(nNodes)
    I3 = np.zeros(nNodes)
    J  = np.zeros(nNodes)

    for i in range(nNodes):
        for j in LocalPoints_Ids[i]:
            X = Pts[j,:]
            F[i,0,:] += Shape_dx[i][j]*X
            F[i,1,:] += Shape_dy[i][j]*X
            F[i,2,:] += Shape_dz[i][j]*X

            f = F[i,:,:]
            C   = np.dot(f.T,f)
            C2  = np.dot(C,C)

            # Define principle strains and J
            trC   = C.trace()
            trC2  = C2.trace()
            I1[i] = trC
            I2[i] = (trC**2-trC2)/2
            I3[i] = np.linalg.det(C)
            J[i] = np.linalg.det(f)

    return F, I1, I2, I3, J

def RunMain(flist,ref,RunChoice):
                
    if RunChoice == 'NIFTI':
        nii = nib.load(ref)
        pixdim = nii.header['pixdim']
        Vol = np.prod(pixdim[1:4])

        DMax = np.max(pixdim[1:4])
        DMin = np.min(pixdim[1:4])
        os.system('/Applications/ITK-SNAP.app/Contents/bin/c3d ' + ref + ' -trim 5vox -o temp_vtk.vtk')

        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName('temp_vtk.vtk')
        reader.ReadAllScalarsOn()
        reader.Update()
        new_data = reader.GetOutput()

        thresh = vtk.vtkThreshold()
        thresh.SetInputData(new_data)
        thresh.Scalars = ['POINTS', 'scalars']
        thresh.ThresholdRange = [0.5, 5.0]
        thresh.Update()
        polydata = thresh.GetOutput()
        Pts_ref = vtk_to_numpy(polydata.GetPoints().GetData())
        
    elif RunChoice == 'VTK':    
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(ref)
        reader.ReadAllScalarsOn()
        reader.Update()
        polydata = reader.GetOutput()
        Pts_ref = vtk_to_numpy(polydata.GetPoints().GetData())
        DMax = 0.55
        DMin = 0.48
    if RunChoice =='NIFTI' or RunChoice == 'VTK':    
        Vol  = DMin**3
        r    = (Vol*(3/(4*math.pi)))**(1/3)
        supp = 3.0*DMax
        supphat = 0.9*DMin
    
        print('Finding boxes...')
        boxes = BoxSearch(Pts_ref,r+supp)
        print('Found')
    
        #Create empty arrays
        nNodes = len(Pts_ref[:,0])
    
        # Order of the basis
        order = 1 
        
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
            
        print('Finding Shape Functions...')
        Shape, Shape_dx, Shape_dy, Shape_dz, LocalPoints_Ids = [],[],[],[],[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(ModRK3D, Pts_ref,repeat(Pts_ref),repeat(r),repeat(supp),repeat(supphat),repeat(order),repeat(boxes),repeat(Pos)),total = len(Pts_ref)))
            for result in results:    
                Shape.append(result[0])
                Shape_dx.append(result[1])
                Shape_dy.append(result[2])
                Shape_dz.append(result[3])
                LocalPoints_Ids.append(result[4])
        
        
        N = len(flist)
        # ReOrder list to be 
        FListOrdered = OrderList(flist, N, ref)
        for X,Fname in enumerate(FListOrdered):
            if RunChoice == 'NIFTI':
                nii = nib.load(Fname)
                pixdim = nii.header['pixdim']
                Vol = np.prod(pixdim[1:4])
        
                DMax = np.max(pixdim[1:4])
                DMin = np.min(pixdim[1:4])
                os.system('/Applications/ITK-SNAP.app/Contents/bin/c3d ' + ref + ' -trim 5vox -o temp_vtk.vtk')
        
                reader = vtk.vtkGenericDataObjectReader()
                reader.SetFileName('temp_vtk.vtk')
                reader.ReadAllScalarsOn()
                reader.Update()
                new_data = reader.GetOutput()
        
                thresh = vtk.vtkThreshold()
                thresh.SetInputData(new_data)
                thresh.Scalars = ['POINTS', 'scalars']
                thresh.ThresholdRange = [0.5, 5.0]
                thresh.Update()
                polydata = thresh.GetOutput()
                Pts = vtk_to_numpy(polydata.GetPoints().GetData())
                
            elif RunChoice == 'VTK':    
            
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(ref)
                reader.ReadAllScalarsOn()
                reader.Update()
                polydata = reader.GetOutput()
                Pts = vtk_to_numpy(polydata.GetPoints().GetData())
                DMax = 0.55
                DMin = 0.48
    
            F, I1, I2, I3, J = FindFandInvariants(Pts, Shape_dx, Shape_dy, Shape_dz, LocalPoints_Ids, nNodes)
            #Save deformation gradient (as rows) and invariants to vtk files
            F_flat = np.zeros((nNodes,9))
            for i in range(nNodes):
                F_flat[i,:] = F[i,:,:].ravel()
                
            VectorData = [F_flat,I1,I2,I3,J]
            VectorNames = ['F (flat)','I1','I2','I3','J']
            
            for i in range(len(VectorNames)):
                arrayVector = vtk.util.numpy_support.numpy_to_vtk(VectorData[i], deep=True)
                arrayVector.SetName(VectorNames[i])
                dataVectors = polydata.GetPointData()
                dataVectors.AddArray(arrayVector)
                dataVectors.Modified()
                
            #################################
            # Write data to vtp files
        
            fname = 'NewVTK' + os.path.splitext(Fname)[0] + '.vtk'
            directory = os.path.dirname(fname)
            if not os.path.exists(directory):
                os.makedirs(directory)
    
            writer = vtk.vtkDataSetWriter()
            writer.SetFileName(fname)
            writer.SetInputData(polydata)
            print('Writing ',fname)
            writer.Write()
    
    if RunChoice == 'TEST':
        x = np.linspace(0,10,6)
        y = np.linspace(0,10,6)
        z = np.linspace(0,10,6)
        Pts = np.zeros((6**3,3))
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    Pts[i+6*j+36*k,:] = [x[i],y[j],z[k]]
        DMax = 2.0
        DMin = 1.9

        Vol  = DMin**3
        r    = (Vol*(3/(4*pi)))**(1/3)
        supp = 3.0*DMax
        supphat = 0.9*DMin

        print('Finding boxes...')
        boxes = BoxSearch(Pts,r+supp)
        print('Found')

        #Create empty arrays
        nNodes = len(Pts[:,0])
        
        # Order of the basis
        order = 1
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
            
        print('Finding Shape functions...')
        Shape, Shape_dx, Shape_dy, Shape_dz, LocalPoints_Ids = [], [], [], [], []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(ModRK3D, Pts,repeat(Pts),repeat(r),repeat(supp),repeat(supphat),repeat(order),repeat(boxes),repeat(Pos)),total = len(Pts)))
            print(results)
            for result in results:
                Shape.append(result[0])
                Shape_dx.append(result[1])
                Shape_dy.append(result[2])
                Shape_dz.append(result[3])
                LocalPoints_Ids.append(result[4])

        TestF = np.array([[2.0,1.0,1.0],[0.0,3.0,1.0],[1.0,0.0,4.0]])
        TestC = np.array([10.0,20.0,15.0])
        pts = np.array(Pts)
        
        TestPts  = pts.dot(TestF)
        TestPts += TestC
        
        F, I1, I2, I3, J = FindFandInvariants(TestPts, Shape_dx, Shape_dy, Shape_dz, LocalPoints_Ids, nNodes)
        for i in range(nNodes):
            print(f'Deformation gradient of node {i} is:')
            print(F[i,:])


if __name__=='__main__':
    ''' 
    Run Script with one of three options
    'VTK' - Runs script with all of the propagated point clouds in the vtk format and save tham into a new directory called 'NewVTK
    'NIFTI' - Runs script with all of the point clouds in the nifti format and save tham into a new directory called 'VTKfromNifti'
    'Test' - Runs a test version of the script in which a new cube mesh is created. This mesh is then used to find the shape functions and then the mesh is deformed with the deformation gradient F=[[2,1,1][0,3,1][1,0,4]]. Then the deformation gradients and invariants are found, and the deformation gradient is returned to confirm the system is working.
    '''
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    WDIR = sys.argv[1]
    refN = int(sys.argv[2])
    RunChoice = sys.argv[3]
    
    print("Computing mesh-free root deformation gradients")
    
    if RunChoice == 'NIFTI':
        fnames = sorted(glob.glob(os.path.join(WDIR,'*root*.nii.gz')))
        fdir = os.path.dirname(fnames[0])
    elif RunChoice == 'VTK':
        fnames = sorted(glob.glob(os.path.join(WDIR,'*root*.vtk')))
        fdir = os.path.dirname(fnames[0])
    
    if RunChoice != 'TEST':
        if not fnames:
            print(WDIR," is empty")
        else:
            fdir = os.path.dirname(fnames[0])
            # Check Directory
            if not os.path.exists(fdir):
                print('Error: Path does not exist:', fdir)
                sys.exit()
                
        common = os.path.commonprefix(fnames)
        
        for Fname in list(fnames):
            if RunChoice == 'NIFTI':
                X = Fname.replace(common,'')
                # Get root name 
                WDIR.replace('./PointClouds/','')
                rootname = WDIR
                # Get file suffix
                common_suffix = '_'+rootname+'_root_reslice.nii.gz'
                # Get Frame number
                X = X.replace(common_suffix,'')
                X = np.fromstring(X, dtype=int, sep=' ')
                X=X[0]
                if X==refN:
                    ref=Fname
            elif RunChoice == 'VTK':
                X = Fname.replace(common,'')
                # Get root name 
                WDIR.replace('./PropagatedPointClouds/','')
                WDIR.replace('/propagated point clouds/','')
                rootname = WDIR
                # Get file suffix
                common_suffix = '_'+rootname+'_root_pointcloud.vtk'
                # Get Frame number
                X = X.replace(common_suffix,'')
                X = np.fromstring(X, dtype=int, sep=' ')
                X=X[0]
                if X==refN:
                    ref=Fname
    else:
        fnames = []
        ref = []
                
    RunMain(flist=fnames, ref=ref, RunChoice=RunChoice)
    
    if RunChoice == 'NIFTI':
        os.system("rm temp_vtk.vtk")
