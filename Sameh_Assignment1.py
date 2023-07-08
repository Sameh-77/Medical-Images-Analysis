# Sameh Algharabli --- CNG 1530 -- Assignment 1# 
#----------------------------------------------# 

# Libraries
import cv2
import os
import numpy as np

#from google.colab.patches import cv2_imshow

# ----------------------------------------------#

# ---------------Function for reading the info.txt file-----------------#
"""
The groundTruth list contains list of dictionaries, each dictionary is for one image
the dictionary has the name, x, y, radius, in another function 'avalue' will be 
added to the dictionary as the image itself 
"""


def read_groundTruth():
    groundTruth = []
    f = open("Dataset/Info.txt", "r")
    #f = open("data/Info.txt", "r")
    for line in f:
        imageDetails = {'name': "", 'x': 0, 'y': 0, 'radius': 0}
        line = line.split()
        imageDetails['name'] = line[0]
        imageDetails['x'] = int(line[4])
        imageDetails['y'] = 1024 - int(line[5])
        imageDetails['radius'] = int(line[6])
        groundTruth.append(imageDetails)
    return groundTruth


# ---------------------------#

# -----Function for reading the images---------#
"""
This function takes the groundtruth list, and append the image 
when the name match, it's added as the value of the image 
"""


def read_images(folder, groundTruth):
    for filename in os.listdir(folder):
        if (filename.split('.'))[-1] != 'pgm':
            continue
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            name = (filename.split('.'))[0]
            for x in range(len(groundTruth)):
                if name == groundTruth[x]['name']:
                    groundTruth[x]['value'] = img
    return groundTruth


# --------------------------------#
# ---------- GAMMA CORRECTION ---- #
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


# ===============================================#

def segmentation(groundTruthh):
    i = 0
    segmentedImages = []  # this list will be used to store the segmeneted images
    # A for loop to go through all the images
    for x in range(0,len(groundTruthh)):
        # this dictionary is used for each segmented image, it contains its name,
        # x,y,radius and the segmented image itself.

        segmentedImage = {}

        # image 3, image 5 , 7,9 , 11 (same problem as 7),15,16

        # (Lsstt workinggg ) 5,0,14,13,10

        #print(i)
        #i = i + 1
        # Reading the image and its name
        imgg = groundTruthh[x]['value']
        tempImg = imgg.copy()
        name = groundTruthh[x]['name']
        #print(name)
        # cv2.imshow('Original image ' + name, imgg)
        # cv2.waitKey(0)
        # -------------------------
        # Converting the image to GrayScale image
        greyscale = cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Gray scale",greyscale)
        # cv2.waitKey(0)
        #--------------------------------
        # Applying the gamma correction
        gammaImg = gammaCorrection(greyscale, 0.7)
        # cv2.imshow('Gamma corrected image', gammaImg)
        # cv2.waitKey(0)
        # -------------------------


        # =========================
        """ Here I'm doing thresholding to remove with a low value threshold so that 
        I will have to the breast area as the biggest area then I apply the controus 
        and sort them, and take the maximum controus, which is the breast area 
        which is the area that contains the tumors 
        """
        # Keeping all breast area as one area
        th, thresholedOriginal = cv2.threshold(gammaImg, 75, 255, cv2.THRESH_BINARY)

        # cv2.imshow("thresholed Image", thresholedOriginal)
        # cv2.waitKey(0)
        # -----------------------

        hh, ww = thresholedOriginal.shape[:2]
        # get largest contour
        contours, x = cv2.findContours(thresholedOriginal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        ours = contours[0]

        # draw largest contour as white filled on black background as mask
        mask = np.zeros((hh, ww), dtype=np.uint8)
        cv2.drawContours(mask, [ours], 0, 255, cv2.FILLED)

        areaOfSegmentation = cv2.bitwise_and(gammaImg, thresholedOriginal, mask=mask)
        # cv2.imshow("areaOfSegmentation", areaOfSegmentation)
        # cv2.waitKey(0)

        # -------------------------
        # kernel = np.ones((7, 7), np.uint8)
        # opened_image = cv2.morphologyEx(areaOfSegmentation, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("Opened Image", opened_image)
        # cv2.waitKey(0)
        # -----------------------------
        """ In this part, I'm getting rid of the pectoral muscle. 
        Here I'm doing thresholding again but with high value, I after applying the thresholding 
        I take the contours of the thresholded image and sort them depending on the area, afte that, 
        the area of pectoral muscle will be the highest so I just get it and build a mask from it
        then I revese it so that I can mask it with the original image and get the area we want 
        without the perctoral muscle 
        """
        th2, thresholded2 = cv2.threshold(areaOfSegmentation, 155, 255, cv2.THRESH_BINARY)

        #kernel = np.ones((10, 10), np.uint8)
        #erodedd = cv2.morphologyEx(thresholded2, cv2.MORPH_ERODE, kernel)
        # print("thresholded2 Image")
        # cv2_imshow(thresholded2)
        # ------------------------

        # ------------------------------
        hh, ww = thresholded2.shape[:2]

        # get largest contour
        contours, x = cv2.findContours(thresholded2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        ours = contours[0]

        # draw largest contour as white filled on black background as mask
        mask = np.zeros((hh, ww), dtype=np.uint8)
        cv2.drawContours(mask, [ours], 0, 255, cv2.FILLED)

        # Investing the mask
        mask = abs(255 - mask)


        # getting the image without the perctoral muscle
        ourpart = cv2.bitwise_and(areaOfSegmentation, areaOfSegmentation, mask=mask)

        # ---------------------------------------#

        th2, new = cv2.threshold(ourpart, 155, 255, cv2.THRESH_BINARY)

        # -------------------------
        # Here I'm applying morphological openning to my images so that
        # the extra parts are seperated, tumor will not be affected
        # because it does not have any holes

        kernel = np.ones((10, 10), np.uint8)
        lastOpening = cv2.morphologyEx(new, cv2.MORPH_OPEN, kernel)

        #kernel = np.ones((30, 30), np.uint8)
        #lastOpening = cv2.morphologyEx(lastOpening, cv2.MORPH_OPEN, kernel)


        # --------------------------
        """
        Here I'm getting all the contours after doing the openning, and then find the largest 
        contours, this contours is the tumor 
        """
        test = np.hstack((greyscale, gammaImg, thresholedOriginal, areaOfSegmentation, mask, ourpart, new, lastOpening))
        cv2.imwrite("test3/" + name + ".png", test)
        cnts, x = cv2.findContours(lastOpening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        if len(cnts) != 0:
            c = cnts[0]
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            # drawing a blue circle
            cv2.circle(tempImg, (int(cX), int(cY)), int(radius+8), 255, -1)

        segmentedImage['name'] = name
        segmentedImage['value'] = tempImg

        segmentedImages.append(segmentedImage)
        #test = np.hstack((greyscale, thresholedOriginal, areaOfSegmentation, thresholded2, ourpart, new, lastOpening))
        #cv2.imwrite("test3/" + name + ".png", test)
    return segmentedImages


# ----------------------------------------------------------------#
"""Here I'm drawing circles on the ground truth """
def segmentGroundTruth(groundTruthCopy):
    segmentedImagesGT = []
    for i in range(len(groundTruthCopy)):
        xgt = groundTruthCopy[i]['x']
        ygt = groundTruthCopy[i]['y']
        rgt = groundTruthCopy[i]['radius']
        name = groundTruthCopy[i]['name']
        groundTruthimage = groundTruthCopy[i]['value']
        newImg = groundTruthimage.copy()
        cv2.circle(newImg, (int(xgt), int(ygt)), int(rgt), (255,0,0), -1)

        segmentedGT = {}
        segmentedGT['name'] = name
        segmentedGT['value'] = newImg
        # segmentedImage['x'] = cX
        # segmentedImage['y'] = cY
        # segmentedImage['radius']=radius
        segmentedImagesGT.append(segmentedGT)

    return segmentedImagesGT

# ---------------------------------------------#
# In this function I calculate the performance of my segmentations by comparing with the ground truth

def perforamceMetrices(segmentedGroundtruth, segmentedImages):
    #z =0
    performance = {"Accuracy": [], "Recall": [], "Precision": [], "IOU": [],"f_score":[]}
    for trueimage in segmentedGroundtruth:
        truth_image = trueimage['value']
        name = trueimage['name']
        for segmented in segmentedImages:
            segmentedImage = trueimage['value']
            if (name==segmented['name']):
                segmentedImage = segmented['value']

                break
        #print(z)
        #z += 1
        test = np.hstack((truth_image, segmentedImage))
        cv2.imwrite("test4/"+name+".png",test)
        #cv2.waitKey(0)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        hh, ww = truth_image.shape[:2]

        for j in range(0,hh):
            for i in range(0,ww):
                #print(truth_image[j, i][0])
                if truth_image[j,i][0] == segmentedImage[j,i][0] == 255:
                    TP += 1
                if segmentedImage[j,i][0] == 255 and truth_image[j,i][0] != segmentedImage[j,i][0]:
                    FP += 1
                if truth_image[j,i][0] == segmentedImage[j,i][0] != 255:
                    TN += 1
                if segmentedImage[j,i][0] != 255 and truth_image[j,i][0] != segmentedImage[j,i][0]:
                    FN += 1


        accuracy = float((TP + TN)) / float((TP + FP + FN + TN))
        if((TP + FP)==0):
            precision=0
        else:
            precision = float((TP)) / float((TP + FP))

        if ((TP + FN) == 0):
            recall = 0
        else:
            recall = float((TP)) / float((TP + FN))

        if ((TP + FN + FP) == 0):
            recall = 0
        else:
            IOU = float((TP)) / float((TP + FN + FP))

        if (recall==0 or precision==0):
            f_score = 0
        else:
            f_score = 2 * ((precision*recall)/(precision+recall))

        print("Accuracy for image " + name + "=" + str(accuracy))
        print("precision for image " + name + "=" + str(precision))
        print("recall for image " + name + "=" + str(recall))
        print("IOU for image " + name + "=" + str(IOU))
        print("f_score for image " + name + "=" + str(f_score))
        print("---------------------------------------")

        performance['Accuracy'].append(accuracy)
        performance['Precision'].append(precision)
        performance['Recall'].append(recall)
        performance['IOU'].append(IOU)
        performance['f_score'].append(f_score)

    mean_accuracy = sum(performance['Accuracy']) / len(performance['Accuracy'])
    mean_precision = sum(performance['Precision']) / len(performance['Precision'])
    mean_recall = sum(performance['Recall']) / len(performance['Recall'])
    mean_IOU = sum(performance['IOU']) / len(performance['IOU'])
    if (len(performance['f_score']) == 0):
        mean_fscore=0
    else:
        mean_fscore = sum(performance['f_score']) / len(performance['f_score'])
    print("Mean Accuracy = " + str(mean_accuracy))
    print("Mean Precision = " + str(mean_precision))
    print("Mean Recall = " + str(mean_recall))
    print("Mean IOU = " + str(mean_IOU))
    print("Mean f_score = " + str(mean_fscore))
    print("---------------------------------------")

# -----------------------------------------#
def main():
    groundTruth = read_groundTruth()  # reading the Info file
    groundTruth = read_images("Dataset", groundTruth)  # reading the images and adding them to the list
    segmentedGroundtruth = segmentGroundTruth(groundTruth)
    segmentedImages = segmentation(groundTruth)

    perforamceMetrices(segmentedGroundtruth, segmentedImages)


    #---------------------------------------------#

# calling the main function
main()
