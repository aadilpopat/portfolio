import cv2
import numpy as np
import imutils

# get video
cap = cv2.VideoCapture('Videos/carpark_footage2.mp4')
ret, current_frame = cap.read()
# first_frame = current_frame
first_frame = cv2.imread('Images/refframe_footage2.png', cv2.IMREAD_COLOR)
car_cx = 0
car_cy = 0
allbays = []
allcars = []
cars = []
kernel2 = np.ones((7, 7), np.uint8)
kernel3 = np.ones((12, 12), np.uint8)
kernel4 = np.ones((20, 40), np.uint8)
kernel5 = np.ones((40, 40), np.uint8)


def FirstFrameProcessor():
    global first_frame_blank_gray
    global first_frame_dilate
    global first_frame_read

    first_frame_read = cv2.imread('Images/refframe_footage2.png', cv2.IMREAD_COLOR)
    first_frame_blank = first_frame_read.copy()
    first_frame_blank_gray = cv2.cvtColor(first_frame_blank, cv2.COLOR_BGR2GRAY)
    first_frame_gray = cv2.cvtColor(first_frame_read, cv2.COLOR_BGR2GRAY)
    first_frame_open = cv2.morphologyEx(first_frame_gray, cv2.MORPH_OPEN, kernel2)
    first_frame_close = cv2.morphologyEx(first_frame_open, cv2.MORPH_CLOSE, kernel4)
    first_frame_dilate = cv2.morphologyEx(first_frame_close, cv2.MORPH_DILATE, kernel4)


def currentframeprocessor():
    global current_frame_gray
    global first_frame_blank_gray
    global previous_frame
    global ret
    global current_frame
    global frame_diff
    global fgmaskdilate
    global fgmaskopen
    global fgmaskclose

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # used to get pb contours
    first_frame_blank = first_frame.copy()
    first_frame_blank_gray = cv2.cvtColor(first_frame_blank, cv2.COLOR_BGR2GRAY)
    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()
    frame_diff = cv2.absdiff(current_frame_gray, first_frame_blank_gray)
    retval, threshold = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
    fgmaskopen1 = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel4)
    fgmaskopen2 = cv2.morphologyEx(fgmaskopen1, cv2.MORPH_OPEN, kernel5)
    fgmaskclose = cv2.morphologyEx(fgmaskopen2, cv2.MORPH_CLOSE, kernel3)
    fgmaskdilate = cv2.morphologyEx(fgmaskclose, cv2.MORPH_DILATE, kernel2)


class DefineBay:
    """"Common base for all parking a_Bay"""
    bay_count = 0

    def __init__(self, point1, point2, point3, point4):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        self.point4 = point4
        self.allpoints = np.array([self.point1, self.point2, self.point3, self.point4])
        DefineBay.bay_count += 1

    def displaycount(self):
        print("Total Bays %d" % DefineBay.bay_count)

    def displaybaypoints(self):
        print("Point 1: ", self.point1,
              "Point 2: ", self.point2,
              "Point 3: ", self.point3,
              "Point 4: ", self.point4)


class BayDetection:
    """"Common base for centre of bay"""
    Xcentrepoints = []
    Ycentrepoints = []
    XcentrepointsLT = []
    XcentrepointsHT = []
    YcentrepointsLT = []
    YcentrepointsHT = []
    bayVac = ""
    hasChanged = []

    def __init__(self):
        pass

    def baycontours(self):
        global first_frame_thresh
        global parking_bay_contours
        for a_Bay in allbays:
            cv2.fillPoly(first_frame_dilate, pts=[a_Bay.allpoints], color=(255, 255, 255))
            ret, first_frame_thresh = cv2.threshold(first_frame_dilate, 254, 255, 0)
            parking_bay_contours = cv2.findContours(first_frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            parking_bay_contours = parking_bay_contours[0] if imutils.is_cv2() else parking_bay_contours[1]


    def baylocating(self):

        for each_bay in parking_bay_contours:
            M = cv2.moments(each_bay)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # for a_Bay in allbays:
            self.Xcentrepoints.append(cx)
            self.Ycentrepoints.append(cy)
            self.XcentrepointsLT.append(cx-25)
            self.XcentrepointsHT.append(cx+50)
            self.YcentrepointsLT.append(cy-25)
            self.YcentrepointsHT.append(cy+40)
            self.hasChanged.append(True)

    def findbays(self):
        self.baycontours()
        self.baylocating()

    def baydetection(self):
        baynumber = 0

        total = len(self.Xcentrepoints)

        # for bay in range(baynumber):


        while baynumber < total:
            for eachcar in cars:
                if self.YcentrepointsLT[baynumber] <= eachcar[1] <= self.YcentrepointsHT[baynumber] and \
                        self.XcentrepointsLT[baynumber] <= eachcar[0] <= self.XcentrepointsHT[baynumber]:
                    self.bayVac = "taken"
                    cv2.rectangle(previous_frame, (self.XcentrepointsLT[baynumber], self.YcentrepointsLT[baynumber]), (self.XcentrepointsHT[baynumber], self.YcentrepointsHT[baynumber]), (255, 0, 0), 1)
                    cv2.putText(previous_frame, "Occupied", (self.Xcentrepoints[baynumber], self.Ycentrepoints[baynumber]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 230), 1)
                    self.bayshouter(baynumber)
                else:
                    self.bayVac = "free"
                    cv2.putText(previous_frame, "Vacant", (self.Xcentrepoints[baynumber], self.Ycentrepoints[baynumber]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 200, 0), 1)
                    cv2.rectangle(previous_frame, (self.XcentrepointsLT[baynumber], self.YcentrepointsLT[baynumber]), (self.XcentrepointsHT[baynumber], self.YcentrepointsHT[baynumber]), (255, 0, 0), 1)
                    self.bayshouter(baynumber)
            baynumber += 1

    def bayshouter (self, bay_number):
        baynumber = bay_number + 1
        if self.bayVac == "free" and self.hasChanged[bay_number] is True:
            self.hasChanged[bay_number] = False
            print('bay ', baynumber, ' is available')
        elif self.bayVac == "taken" and self.hasChanged[bay_number] is False:
            self.hasChanged[bay_number] = True
            print('bay ', baynumber, ' is occupied')


class Cars:

    def carcontours(self):
        global car_contours
        # find contours of cars
        car_contours = cv2.findContours(fgmaskdilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        car_contours = car_contours[0] if imutils.is_cv2() else car_contours[1]


    def carlocating(self):
        global car_M
        global car_cx
        global car_cy


        for each_car in car_contours:
            if car_contours:
                car_M = cv2.moments(each_car)
                car_cx = int(car_M['m10']/car_M['m00'])
                car_cy = int(car_M['m01']/car_M['m00'])
                cars.append([car_cx, car_cy])
                # print(cars)
                if car_M["m00"] != 0:
                    floor_1.baydetection()
                else:
                    car_cx, car_cy = 0, 0


FirstFrameProcessor()

allbays.append(DefineBay([2, 355], [84, 334], [173, 441], [2, 477],))
allbays.append(DefineBay([90, 331], [209, 307], [344, 395], [179, 441],))
allbays.append(DefineBay([217, 308], [350, 276], [479, 355], [347, 394],))
allbays.append(DefineBay([357, 277], [445, 254], [580, 321], [484, 353],))

floor_1 = BayDetection()
floor_1.findbays()
carList = Cars()

# while video is playing

while cap.isOpened():
    cars = []

    currentframeprocessor()

    carList.carcontours()

    carList.carlocating()

    # cv2.drawContours(previous_frame, parking_bay_contours, -1, (0, 0, 255), 1)
    # cv2.drawContours(previous_frame, car_contours, -1, (0, 255, 0), 1)
    # cv2.putText(previous_frame, "X", (car_cx, car_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.rectangle(previous_frame, (420,0), (710,110), (0,0,0), -1)
    # cv2.putText(previous_frame, "KEY",(550, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.putText(previous_frame, "Red = Parking Bay",(500, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    # cv2.putText(previous_frame, "Green = Car",(500, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    # cv2.putText(previous_frame, "Blue = Bay Detection Proximity", (440, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # cv2.imshow('current_frame', current_frame)
    # cv2.imshow('framediff', frame_diff)
    # cv2.imshow('open', fgmaskopen)
    # cv2.imshow('close', fgmaskclose)
    # cv2.imshow('cars_dilate', fgmaskdilate)
    # cv2.imshow('first_frame_dilate', first_frame_dilate)
    cv2.imshow('prev', previous_frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
