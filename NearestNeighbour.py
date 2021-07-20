import numpy as np
import os
import sys
import vtk
import glob
import csv

from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa
from FileConversion import OrderList
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
'''
class Neighbours:

    def __init__(self, name):
        Ranges = np.array(polydata.GetPoints().GetBounds())
        self.FileName = name

    def add_Points(self,Pt):
        NP = len(Pts)
        self.Points = np.zeros((3,NP))
        for i in range(NP):
            self.Points(i).append(Pt)
'''

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


if __name__=='__main__':
    FixAndRotate=False
    List_of_Subdirectories = sorted(glob.glob('./Point_Clouds/*'))
    CommonOfDir = os.path.commonprefix(List_of_Subdirectories)
    for d in List_of_Subdirectories:
        DataDir = d.replace(CommonOfDir,'')

        with open('echoframetime.csv') as csv_file:
            XLData = csv.reader(csv_file, delimiter=',')
            for row in XLData:
                if DataDir == row[0]:
                    DataInfo = row

        fnames = sorted(glob.glob(d + '/propagated point clouds/*.vtk'))

        refN = int(DataInfo[4])-1

        common = os.path.commonprefix(fnames)
        for Fname in list(fnames):
            X = Fname.replace(common,'')
            X = X.replace('.vtk','')
            X = np.fromstring(X, dtype=int, sep=' ')
            X=X[0]
            if X==refN:
                ref=Fname

        N = len(fnames)
        FListOrdered, FId, refN = OrderList(fnames,N,ref)

        #Get reference frame
        Fname = FListOrdered[0]
        print('Reading Current Frame:', Fname)

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
        Diff = np.zeros(3)

        xmin = Ranges[0]
        ymin = Ranges[2]
        zmin = Ranges[4]

        for i in range(3):
            Diff[i] = Ranges[2*i+1] - Ranges[2*i]

        r = 0.5
        bN = int(np.max(Diff)/r)+1
        N = len(Pts[:,0])

        bIDi = np.zeros(N)
        bIDj = np.zeros(N)
        bIDk = np.zeros(N)
        b = [[[[] for i in range(bN)] for j in range(bN)] for k in range(bN)]
       
        for id in range(N):
            i = int((Pts[id,0]-xmin)/r)
            j = int((Pts[id,1]-ymin)/r)
            k = int((Pts[id,2]-zmin)/r)
        
            b[i][j][k].append(Pts[id,:])

        NN = []
        for i in range(N):
            NN.append([])

        for id in range(10): #range(N):
            Points = []
            print(id)
            i = int((Pts[id,0]-xmin)/r)
            j = int((Pts[id,1]-ymin)/r)
            k = int((Pts[id,2]-zmin)/r)
            for I in [-1,0,1]:
                for J in [-1,0,1]:
                    for K in [-1,0,1]:
                        Iid = int(i+I)
                        Jid = int(j+J)
                        Kid = int(k+K)
                        if (0<= Iid <bN) and (0<=Jid <bN) and (0<=Kid<bN):
                            Points += b[Iid][Jid][Kid]
            print('Surrounding Points = ',len(Points))

            l =  NN_BF(Pts[id][:],Points,r)
            NN[i].append(l)
            print('Nearest Points = ',len(l))
            print('l = ',l)
        ''' 
        #Initiate Class
        NN_class = Neighbours(DataDir)
        if i in range(len(Pts)):
            Pt = Pts(i)
            NN_class.add_Points(Pt)
        '''

