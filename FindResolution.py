import numpy as np

def FindD(Pts,Ranges,r,boxes):
    '''
    A function to find the standard distance between node points in the three
    principle directions.

    Inputs: Pts - Array of the nodes of size N x 3, where N is the number of nodes
            Ranges - the maximums and minimums in the 3 directions, as a 1 x 6 array,
                    where the format is [x_min, x_max, y_min, y_max, z_min, z_max]
            r - box size
            boxes - class of the results of the box search
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

    for i in range(N):
        # Find local points to reduce
        PointIds =  boxes.neighb(Pts[i,:])
        
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
