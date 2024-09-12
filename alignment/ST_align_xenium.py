import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from STalign import STalign
import torch
import os
import tifffile as tf
from tifffile import imread 

#=============================================================================================================#
#import xenium data

xen_path1='/path/to/xenium/folder/cells.parquet'
xen_path2='/path/to/metabolomics/coordinates.csv'

z1=pd.read_csv('xen_path1')
z2=pd.read_csv('xen_path2')

#asign x and y coordinates to numpy arrays

xA=np.array(z1['x_new'])
yA=np.array(z1['y_new'])
xB=np.array(z2['x_new'])
yB=np.array(z2['y_new'])

#=============================================================================================================#

#do initial translation and rotation
#this might have to be done in a loop to get the best fit
yA1=yA
xA1=xA
yB1=yB
xB1=xB

xcoord= 0
ycoord= 0
theta_deg = -8
theta0 = (np.pi/180) * -theta_deg

# Rotation matrix
L = np.array([[np.cos(theta0), -np.sin(theta0)],
              [np.sin(theta0), np.cos(theta0)]])

# Apply rotation to all sources
source_L = np.matmul(L, np.array([xB1, yB1]))
xI_L = source_L[0]
yI_L = source_L[1]

#translation matrix
#effectively makes the rotation about the centroid of I (i.e the means of xI and yI])
#and also moves the centroid of I to the centroid of J
T = np.array([ np.mean(xA1)- np.cos(theta0)*np.mean(xA1) +np.sin(theta0)*np.mean(yA1) - (np.mean(xA)-np.mean(xB1)),
              np.mean(yA1)- np.sin(theta0)*np.mean(xA1) -np.cos(theta0)*np.mean(yA1) - (np.mean(yA)-np.mean(yB1))])

xI_L_T = xI_L + T[0] - xcoord
yI_L_T = yI_L + T[1] - ycoord

# Plot all sources
fig, ax = plt.subplots(1,2,figsize=(20,10))

ax[0].scatter(xA1,yA1, s=0.2, alpha=0.3, label='Xenium')
ax[0].set_xticks(np.arange(0, 10000, 500))
ax[0].set_yticks(np.arange(0, 10000, 500))
ax[0].grid()
ax[0].legend(markerscale=10)
#ax.scatter(xD,yD, s=1, alpha=0.1, label='z2_orig')
ax[1].scatter(xI_L_T, yI_L_T, s=1, alpha=0.1, label='Metab')
ax[1].set_xticks(np.arange(0, 10000, 500))
ax[1].set_yticks(np.arange(0, 10000, 500))
ax[1].grid()
ax[1].legend(markerscale=10)
plt.tight_layout()

plt.show()

#=============================================================================================================#

#add landmark points
pointsXen_A = np.array([[2000,600], [8000, 7000], [6800,8850],[500,5500],[900,6000],[6250,4250]])
pointsXen_B= np.array([[400,150] , [4000,2750],[3250,3750],[100,2500],[400,2900], [2800,1550]])

pointsI=pointsXen_A 
pointsJ=pointsXen_B

#rasterize Xen Data for LDDMM
XA,YA,I,figA=STalign.rasterize(xA,yA, dx=30)
XB,YB,J,figB=STalign.rasterize(xI_L_T,yI_L_T, dx=30)

extentI= STalign.extent_from_x((YA,XA))
extentJ = STalign.extent_from_x((YB,XB))

fig,ax = plt.subplots(1,2)
ax[0].imshow((I.transpose(1,0,2).squeeze()), extent=extentI)
ax[1].imshow((J.transpose(0,1,2).squeeze()), extent=extentJ)

ax[0].scatter(pointsI[:,1],pointsI[:,0], c='red')
ax[1].scatter(pointsJ[:,1],pointsJ[:,0], c='red')
for i in range(pointsI.shape[0]):
    ax[0].text(pointsI[i,1],pointsI[i,0],f'{i}', c='red')
    ax[1].text(pointsJ[i,1],pointsJ[i,0],f'{i}', c='red')

#=============================================================================================================#

# compute initial affine transformation from points
L,T = STalign.L_T_from_points(pointsJ,pointsI)

affine = np.dot(np.linalg.inv(L), [xI_L_T - T[0], yI_L_T - T[1]])
print(affine.shape)
xMaffine = affine[0,:]
yMaffine = affine[1,:]

# plot
fig,ax = plt.subplots()
ax.scatter(xMaffine,yMaffine,s=1,alpha=0.8)
ax.scatter(xA,xB,s=1,alpha=0.8)

#=============================================================================================================#

#compute LDDMM


# run LDDMM
# specify device (default device for STalign.LDDMM is cpu)
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# keep all other parameters default
params = {'L':L,'T':T,
          #'epL':2,
          #'epT':200,
          #'expand':1,
          #'p':1.5,
          'niter':100,
          'pointsI':pointsI,
          'pointsJ':pointsJ,
          #'sigmaM':0.8,
          'device':device,
          #'sigmaB':0.2,
          #'sigmaA':0.2,
          #'diffeo_start':50,
          #'sigmaP':0.10,
          #'sigmaR':5,
          'epV': 200,
          #'muB': torch.tensor([0,0,0]), # black is background in target
          #'muA': torch.tensor([1,1,1]) # use white as artifact
          }

outBA = STalign.LDDMM([YB,XB],BZ,[YA,XA],AZ,**params)

# set device for building tensors
if torch.cuda.is_available():
    torch.set_default_device('cuda:0')
else:
    torch.set_default_device('cpu')

A = outBA['A']
v = outBA['v']
xv = outBA['xv']
tpointsBA = STalign.transform_points_source_to_target(xv,v,A,np.stack([xI_L_T,yI_L_T],1))

tpointsBA=tpointsBA.cpu().numpy()
xB_LDDMM = tpointsBA[:,0]
yB_LDDMM = tpointsBA[:,1]
xB_n=xB_LDDMM + 0
yB_n=yB_LDDMM - 0

fig,ax = plt.subplots(figsize=(20,20))
#ax.imshow((I).transpose(1,2,0),extent=extentI)
ax.scatter(xA,yA,s=1,alpha=0.9, label='Z1',color='blue')
#ax.scatter(xB,yB,s=1,alpha=0.9, label='xenium',color='green')
ax.scatter(xB_n,yB_n,s=1,alpha=0.9, label='Z2',color='orange')

#=============================================================================================================#

#export transformed data as part of original xenium dataset 


x=xB_n
y=yB_n
z=z2

aligned = np.stack((x,y),1)
headers_df1 = z.columns.tolist()
new_headers= headers_df1 + ['x_new','y_new']
results= np.hstack((z, aligned))
results_df = pd.DataFrame(results, columns=new_headers)
results_df.to_csv('/stornext/Bioinf/data/lab_brain_cancer/projects/tme_spatial/venture_multi_omics/venture_pt2/aligned_coordinates/z2_coordinates.csv', index=False)
