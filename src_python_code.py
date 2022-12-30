# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 21:02:13 2022

@author: devad
"""

import numpy as np
import cv2
img = cv2.imread("Lenna.png")
cv2.imshow('image', img)
cv2.waitKey(0)
#annotation code begins
def edge_lines(event, x, y, flag, parameters):
    global index, vertices
    if event == cv2.EVENT_LBUTTONDOWN:
            vertices = [(x,y)]
    elif event == cv2.EVENT_LBUTTONUP:
            vertices.append((x,y))
            print(f"edge line starts at: {vertices[0]} edge line ends at: {vertices[1]}")
            cv2.line(img, vertices[0], vertices[1],(0,255,0),4) #green color lines drawn to get vertex coordinates
            cv2.imshow("image", img)
            
vertices = []
index = 0
img = cv2.imread("image_box.png")
cv2.imshow('image', img)
cv2.setMouseCallback('image', edge_lines)
cv2.waitKey(0)
cv2.imwrite("annotated_box_image.png",img)
cv2.destroyAllWindows()
#annotation code ends

#marking points and drawing lines of different colors for each axes
P3 = [596,404,1]
P4 = [877,261,1]
P1 = [612,587,1]
P2 = [875,405,1]
P5 = [248,284,1]
P6 = [550,192,1]
P7 = [281,428,1]

#x axis
cv2.line(img,(P1[0],P1[1]),(P2[0],P2[1]),(250,0,0),2)
cv2.line(img,(P3[0],P3[1]),(P4[0],P4[1]),(250,0,0),2)
cv2.line(img,(P5[0],P5[1]),(P6[0],P6[1]),(250,0,0),2)
#y axis
cv2.line(img,(P1[0],P1[1]),(P7[0],P7[1]),(0,255,0),2)
cv2.line(img,(P3[0],P3[1]),(P5[0],P5[1]),(0,255,0),2)
cv2.line(img,(P4[0],P4[1]),(P6[0],P6[1]),(0,255,0),2)
#z axis
cv2.line(img,(P1[0],P1[1]),(P3[0],P3[1]),(0,0,255),2)
cv2.line(img,(P2[0],P2[1]),(P4[0],P4[1]),(0,0,255),2)
cv2.line(img,(P7[0],P7[1]),(P5[0],P5[1]),(0,0,255),2)
cv2.imshow('box_image',img)
cv2.waitKey(0)
print(np.array(P1))

#defining edges for the marked points
edges= np.zeros((7,3))
edges[0,:] = P1
edges[1,:] = P2
edges[2,:] = P3
edges[3,:] = P4
edges[4,:] = P5
edges[5,:] = P6
edges[6,:] = P7

#defining reference co-ordinates and the distance between the reference and the world origin
x_reference = edges[1]
y_reference = edges[6]
z_reference = edges[2]
wo = edges[0]
length_x_wo = np.sqrt(np.sum(np.square(x_reference - wo)))
length_y_wo = np.sqrt(np.sum(np.square(y_reference - wo)))
length_z_wo = np.sqrt(np.sum(np.square(z_reference - wo)))
print(edges[0])

#vanishing point calculations:
#vanishing point for x axis

ax1,bx1,cx1 = np.cross(edges[0],edges[1])
ax2,bx2,cx2 = np.cross(edges[2],edges[3])
ax3,bx3,cx3 = np.cross(edges[4],edges[5])

Vx1_2 = np.cross([ax1,bx1,cx1],[ax2,bx2,cx2])
Vx2_3 = np.cross([ax3,bx3,cx3],[ax2,bx2,cx2])
Vx1_3 = np.cross([ax1,bx1,cx1],[ax2,bx2,cx2])

Vx1_2 = Vx1_2/Vx1_2[2]
Vx2_3 = Vx2_3/Vx2_3[2]
Vx1_3 = Vx1_3/Vx1_3[2]

#vanishing point for y axis
ay1,by1,cy1 = np.cross(edges[0],edges[6])
ay2,by2,cy2 = np.cross(edges[2],edges[4])
Vy = np.cross([ay1,by1,cy1],[ay2,by2,cy2])
Vy = Vy/Vy[2]

#vanishing point for z axis
az1,bz1,cz1 = np.cross(edges[0],edges[2])
az2,bz2,cz2 = np.cross(edges[1],edges[3])
Vz = np.cross([az1,bz1,cz1],[az2,bz2,cz2])
Vz = Vz/Vz[2]
print(Vx1_2,Vy,Vz)

#calculating projection matrix
Vx1_2 = np.array([Vx1_2])
Vy = np.array([Vy])
Vz = np.array([Vz])
wo = np.array([wo])

x_reference=np.array([x_reference])
y_reference=np.array([y_reference])
z_reference=np.array([z_reference])

ax,resid,rank,s = np.linalg.lstsq( (Vx1_2-x_reference).T , (x_reference - wo).T )
ax = ax[0][0]/length_x_wo
ay,resid,rank,s = np.linalg.lstsq( (Vy-y_reference).T , (y_reference - wo).T )
ay = ay[0][0]/length_y_wo
az,resid,rank,s = np.linalg.lstsq( (Vz-z_reference).T , (z_reference - wo).T )
az = az[0][0]/length_z_wo

projection_matrix_x = ax*Vx1_2
projection_matrix_y = ay*Vy
projection_matrix_z = az*Vz

Projection_matrix = np.empty([3,4])
Projection_matrix[:,0] = projection_matrix_x
Projection_matrix[:,1] = projection_matrix_y
Projection_matrix[:,2] = projection_matrix_z
Projection_matrix[:,3] = wo

#Calculating homograph matrix
Hxy = np.zeros((3,3))
Hxy[:,0] = projection_matrix_x
Hxy[:,1] = projection_matrix_y
Hxy[:,2] = wo

Hyz = np.zeros((3,3))
Hyz[:,0] = projection_matrix_y
Hyz[:,1] = projection_matrix_z
Hyz[:,2] = wo

Hzx = np.zeros((3,3))
Hzx[:,0] = projection_matrix_x
Hzx[:,1] = projection_matrix_z
Hzx[:,2] = wo

print("The homography transformation matrices are given below for XY, YZ and ZX planes:")
print("XY plane: \n")
print(Hxy)
print("YZ plane: \n")
print(Hyz)
print("ZX plane: \n")
print(Hzx)

Hxy[0,2] = Hxy[0,2] +50
Hxy[1,2] = Hxy[1,2] -150

height,width,channel = img.shape
Txy = cv2.warpPerspective(img,Hxy,(width,height),flags=cv2.WARP_INVERSE_MAP)
cv2.imshow("Txy",Txy)
cv2.waitKey(0)
Tyz = cv2.warpPerspective(img,Hyz,(height,width),flags=cv2.WARP_INVERSE_MAP)
cv2.imshow("Tyz",Tyz)
cv2.waitKey(0)
Tzx = cv2.warpPerspective(img,Hzx,(height,width),flags=cv2.WARP_INVERSE_MAP)
cv2.imshow("Tzx",Tzx)
cv2.waitKey(0)
cv2.imwrite("box_image_XY_plane.jpg",Txy)
cv2.imwrite("box_image_YZ_plane.jpg",Tyz)
cv2.imwrite("box_image_ZX_plane.jpg",Tzx)
cv2.destroyAllWindows()