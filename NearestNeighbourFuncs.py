import numpy as np

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
    NN = []
    for i in range(N):
        NN = NN.append([])

    #Find nearest neighbours of each point
    for i in range(N):
        #Find surrounding points and the associated ids to narrow search
        Points, Ids = Local_Pts(Pts,b,bIds,Mins,bN,i,r)
        #Find neighbour within radius
        l =  NN_BF(Pts[i,:],Points,r)
        #Add list of nearest points to overall list
        NN[i].append(l)
        print(i)
    
    return NN



