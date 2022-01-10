import vtk
import nibabel as nib
from vtk.util.numpy_support import vtk_to_numpy
import os
import numpy as np

Fname = 'Point_Clouds/bav01/seg05_to_01_bav01_root_reslice.nii.gz'
os.system('/Applications/ITK-SNAP.app/Contents/bin/c3d ' + Fname + ' -trim 5vox -o temp_vtk.vtk')

#reader = vtk.vtkStructuredPointsReader()
#reader = vtk.vtkPolyDataReader()
reader = vtk.vtkGenericDataObjectReader()
reader.SetFileName('temp_vtk.vtk')
reader.ReadAllScalarsOn()
#reader.ReadAllVectorsOn()
reader.Update()
new_data = reader.GetOutput()
#print(np.array( reader.GetOutput().GetPoints().GetData() ))
print(new_data)
print(dir(new_data))
print(dir(new_data.GetPointData()))
print(vtk_to_numpy(new_data.GetPointData().GetArray(0)))

thresh = vtk.vtkThreshold()
thresh.SetInputData(new_data)
thresh.Scalars = ['POINTS', 'scalars']
thresh.ThresholdRange = [0.5, 5.0]
thresh.Update()
Thresh = thresh.GetOutput()

print(Thresh)
print(dir(Thresh.GetPoints()))
print(Thresh.GetPoints())
print(vtk_to_numpy(Thresh.GetPoints().GetData()))

writer = vtk.vtkDataSetWriter()
writer.SetFileName('TestThresh.vtk')
writer.SetInputData(Thresh)
print('Writing ','TestThresh.vtk')
writer.Write()
#print(thresh.GetOutput)
#print(dir(thresh.GetOutput()))
#print(thresh.GetOutput().GetPointData())
#print(thresh.GetOutput().GetPoints())
#print(dir(thresh.GetOutput().GetCells()))
#print(thresh.GetOutput().GetCells())
#print(thresh.GetOutput().GetCells().GetData())
#print(vtk_to_numpy(thresh.GetOutput().GetCells().GetData()))

'''
img = nib.load(Fname)
img_data = img.get_data()
img_data_shape = img_data.shape
'''
