#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 07:54:09 2021

@author: petermortensen
"""
import numpy as np
from math import pi
import concurrent.futures
from itertools import repeat
from tqdm import tqdm
from MeshFree3DMulti import ModRK3D
# from MeshFree3DMulti import ModRK3DShape
# from MeshFree3DMulti import basis
# from MeshFree3DMulti import calc_phi
from MeshFree3DMulti import FindFandInvariants
from MeshFree3DMulti import BoxSearch

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