# from imutils import face_utils
# import numpy as np
# import imutils
# import dlib
# import cv2
# import math
# import os

# # image_path="C:\\Users\\Dev Atul Patel\\OneDrive\\Desktop\\detection\\Face-shape-detection-and-hairstyle-recommendation\\images\\test22.jpg"
# shape_predictor_path="C:\\Users\\Dev Atul Patel\\OneDrive\\Desktop\\try on final\\jweltryon\\shape_predictor_81_face_landmarks.dat"

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(shape_predictor_path)
# face_shape_printed = False


# def process_image(frame,selected_indices):
# 	global face_shape_printed
# 	image = imutils.resize(frame, width=500)
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 	rects = detector(gray, 1)
# 	count=0

# 	for (i, rect) in enumerate(rects):

# 		shape = predictor(gray, rect)
# 		shape = face_utils.shape_to_np(shape)
# 		for (x, y) in shape:
# 			cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
# 		for (x, y) in shape:
# 			count=count+1;
# 			if(count==1):
# 				(x1,y1)=(x,y)
# 			if(count==3):
# 				(x3,y3)=(x,y)
# 			if(count==5):
# 				(x5,y5)=(x,y)
# 			if(count==7):
# 				(x7,y7)=(x,y)
# 			if(count==9):
# 				(x9,y9)=(x,y)
# 			if(count==17):
# 				(x17,y17)=(x,y)
# 			if(count==28):
# 				(x28,y28)=(x,y)

# 		#print("1.",(x1,y1),"3.",(x17,y17),"5.",(x9,y9),"7.",(x28,y28))
# 		slope1=((y3-y1)*(1.0))/((x3-x1)*(1.0))
# 		slope2=((y5-y3)*(1.0))/((x5-x3)*(1.0))
# 		slope3=((y7-y5)*(1.0))/((x7-x5)*(1.0))
# 		slope4=((y9-y7)*(1.0))/((x9-x7)*(1.0))
# 		#print('s1:',slope1,'s2:',slope2,'s3:',slope3,'s4:',slope4)

# 		distx=math.sqrt(pow((x1-x17),2)+pow((y1-y17),2))
# 		disty=math.sqrt(pow((x9-x28),2)+pow((y9-y28),2))
# 		thresh=distx-disty

# 		lg=open('long','r+')
# 		rnd=open('round','r+')
# 		het=open('heart','r+')
# 		squ=open('square','r+')

# 		thresh_lg=float(lg.readline())
# 		thresh_rnd=float(rnd.readline())
# 		thresh_het=float(het.readline())
# 		thresh_squ=float(squ.readline())

# 		lg.seek(0)
# 		rnd.seek(0)
# 		het.seek(0)
# 		squ.seek(0)
# 		avg_hr=(thresh_rnd+thresh_het)/(2.0)
# 		avg_ls=(thresh_lg+thresh_squ)/(2.0)

# 		total_thresh=(avg_hr+avg_ls)/(2.0)

# 		detected_face=""
# 		if thresh<=total_thresh:
# 			# print("long or square")
# 			if slope1>=7.395:
# 				if slope3>=1.15:
# 					detected_face="long"
# 				else:
# 					detected_face="square"

# 			elif slope1<7.395:
# 				if slope3>=1.15:
# 					detected_face="square"

# 				else:
# 					detected_face="long"


# 		if thresh>total_thresh:
# 			# print("round or heart")
# 			if slope1>=11.75:
# 				if slope3<=1.1:
# 					detected_face="heart"
# 				else:
# 					detected_face="round"

# 			elif slope1<11.75:
# 				if slope3>1.1:
# 					detected_face="round"
# 				else:
# 					detected_face="heart"

# 		if not face_shape_printed and detected_face:
# 			print("Detected face shape:", detected_face)
# 			face_shape_printed = True



