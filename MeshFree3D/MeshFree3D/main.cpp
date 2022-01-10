//
//  main.cpp
//  MeshFree3D
//
//  Created by Peter Mortensen on 06/12/2021.
//
#include <iostream>
#include <array>
#include <vector>
#include <math.h>
//#include <vtkSmartPointer.h>
//#include <vtkGenericDataObjectReader.h>
#include <vnl_matrix.h>
//#include "/Users/petermortensen/opt/anaconda3/include/xtensor/xarray.hpp"
//#include "/Users/petermortensen/opt/anaconda3/include/xtensor/xio.hpp"
//#include "/Users/petermortensen/opt/anaconda3/include/xtensor/xview.hpp"

//std::vector<double> basis(float x,int order){
void basis(std::vector<double> x,int order){

    /*
    Define basis H and associated derivative

    Inputs: x - difference vector
            order - order of the basis

    Outputs: The basis H, its derivative and its transpose
    */
    // Returns the basis and its derivative at a point x based upon the order
    std::vector<double> H(4);
    std::vector<double> dH(4);
    if (order==1){
        H = {1,x[0],x[1],x[2]};
        dH = {0,1,0,0};
    }else{
        std::cout <<"Warning orders that are greater than 1 are not defined"<<std::endl;
        H = {0,0,0,0};
        dH = {0,0,0,0};
    }
    //HT = H.transpose()
    return ;//H,dH; // HT
}

float calc_phi(float z,float a){
    /*
        Calculate phi

        Inputs: z - difference vector magnitude
                a - supp radius

        Outputs: phi
    */
    float phi;
    if (z<a){
        phi = 1;
    } else {
        phi = 0;
    }
    
    return phi;
}
 
int main(int argc, const char * argv[]) {
    //vtkSmartPointer<vtkGenericDataObjectReader> reader =
    //      vtkSmartPointer<vtkGenericDataObjectReader>::New();
    //reader->SetFileName(inputFilename.c_str());
    //reader->Update();
    //reader = vtk.vtkPolyDataReader()
    //reader.SetFileName(ref)
    //reader.ReadAllScalarsOn()
    //reader.Update()
    //polydata = reader.GetOutput()
    //Pts_ref = vtk_to_numpy(polydata.GetPoints().GetData())
    double DMax = 0.55;
    double DMin = 0.48;
    double Vol  = pow(DMin,3);
    double r    = pow((Vol*(3/(4*M_PI))),(1/3));
    double supp = 3.0*DMax;
    double supphat = 0.9*DMin;
        
    std::cout <<"Finding boxes..."<<std::endl;
    //boxes = BoxSearch(Pts_ref,r+supp)
    std::cout <<"Found"<<std::endl;
        
    //Create empty arrays
    //nNodes = len(Pts_ref[:,0])
        
    //Order of the basis
    int order = 1;
            
    //Define points of dodecahedron with centre (0,0,0)
    float phi=0.5+pow((5./4.),1/2);
    std::array<std::array<double,3>,20> Pos;
          
    Pos[0]={1.,1.,1.};
    Pos[1]={1.,1.,-1.};
    Pos[2]={1.,-1.,1.};
    Pos[3]={1.,-1.,-1.};
    Pos[4]={-1.,1.,1.};
    Pos[5]={-1.,1.,-1.};
    Pos[6]={-1.,-1.,1.};
    Pos[7]={-1.,-1.,-1.};
        
    Pos[8]={0.,1./phi,phi};
    Pos[9]={0.,1./phi,-phi};
    Pos[10]={0.,-1./phi,phi};
    Pos[11]={0.,-1./phi,-phi};
    
    Pos[12]={1./phi,phi,0.};
    Pos[13]={1./phi,-phi,0.};
    Pos[14]={-1./phi,phi,0.};
    Pos[15]={-1./phi,-phi,0.};
          
    Pos[16]={phi,0.,1./phi};
    Pos[17]={-phi,0.,1./phi};
    Pos[18]={phi,0.,-1./phi};
    Pos[19]={-phi,0.,-1./phi};

    //Normalise the dodecahedron locations
    for (int j=0; j<20; j++){
        double Mag = pow((pow(Pos[j][0],2)+pow(Pos[j][1],2)+pow(Pos[j][2],2)),1/2);
        Pos[j] = {Pos[j,0]/Mag,Pos[j,1]/Mag,Pos[j,2]/Mag};
    }
    // insert code here...
    int z = 1;
    int a = 10;
    float Phi = calc_phi(z,a);
    std::cout << "Phi=" << Phi<<std::endl;
    
    std::vector<double> x(3);
    x = {2,3,4};
    basis(x,order);
    //std::vector<double>H,dH = basis(x,order);
    return 0;
}
