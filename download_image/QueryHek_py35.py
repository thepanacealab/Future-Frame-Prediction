# @author Bhaskar Ray
# @version 11/26/2018

import urllib.request
import urllib.parse
import csv
import json as j
import pandas as pd
import datetime
from pgmagick import Image
import cv2
import os
import numpy as np
from PIL import Image as PILImage

def hek():
    startYear = ''
    startMonth = ''
    startDay = ''
    startHour = ''
    startMinute = ''
    startSecond = ''
    endYear = ''
    endMonth = ''
    endDay = ''
    endHour = ''
    endMinute = ''
    endSecond = ''
    coordinateValue = ''
    x1Value = ''
    x2Value = ''
    y1Value = ''
    y2Value = ''
    eventModuleString = ''
    filterCondition = []
    filterConditionValue = []
    filterOperators = []
    valuesLoop = []
    fileName = ''
    opDirectory = ''
    json = False
    xml = False
    filter = False
    url = ''
    EVENT_REGION = 'all'
    RESULT_LIMIT = '200'
    ENCODING_SCHEME = 'utf-8'

    try:

        inputFileName = input("Enter input file directory path with file name: ")
        opDirectory = input("Enter output directory path: ")
        dataType = input("Enter data type(JSON/XML): ")

        if "JSON" in dataType:
            url = 'https://www.lmsal.com/hek/her?cosec=2&&cmd=search&type=column'
            json = True
        elif "XML" in dataType:
            url = 'https://www.lmsal.com/hek/her?cosec=1&&cmd=search&type=column'
            xml = True

        with open(inputFileName, 'r') as f:
            contents = f.readlines()
            f.close()
        contents = [content.strip('\n') for content in contents]

        for token in contents:

            tokens = token.split(':')

            if 'Start Date' in tokens[0]:
                for tokenValue in tokens:
                    if '-' in tokenValue:
                        ymd = tokenValue.split('-')
                        startYear = ymd[0].strip()
                        startMonth = ymd[1].strip()
                        startDay = ymd[2].strip()

            elif 'End Date' in tokens[0]:

                for tokenValue in tokens:
                    if '-' in tokenValue:
                        ymd = tokenValue.split('-')
                        endYear = ymd[0].strip()
                        endMonth = ymd[1].strip()
                        endDay = ymd[2].strip()

            elif 'Start Time' in tokens[0]:
                for tokenValue in tokens:
                    if '-' in tokenValue:
                        hms = tokenValue.split('-')
                        startHour = hms[0].strip()
                        startMinute = hms[1].strip()
                        startSecond = hms[2].strip()

            elif 'End Time' in tokens[0]:
                for tokenValue in tokens:
                    if '-' in tokenValue:
                        hms = tokenValue.split('-')
                        endHour = hms[0].strip()
                        endMinute = hms[1].strip()
                        endSecond = hms[2].strip()

            elif 'Event Type' in tokens[0]:

                eventModule = [tokens[1].split(',')]
                eventModuleString = ",".join(eventModule[0]).strip()



            elif 'Spatial Region' in tokens[0]:
                for tokenValue in tokens:
                    if ',' in tokenValue:
                        event = tokenValue.split(',')
                        coordinateValue = event[0].strip()
                        x1Value = event[1].strip()
                        x2Value = event[2].strip()
                        y1Value = event[3].strip()
                        y2Value = event[4].strip()


            else:
                if tokens[0]:
                    filterCondition.append(tokens[0].strip())
                    opVal = tokens[1].split(',')
                    filterOperators.append(opVal[0].strip())
                    filterConditionValue.append(opVal[1].strip())
                    filter = True
                else:
                    filter = False






    except Exception as e:
        print("Input Error: ", str(e))

    try:

        values = {'event_type': eventModuleString, 'event_region': EVENT_REGION, 'event_coordsys': coordinateValue,
                  'x1': x1Value, 'x2': x2Value,
                  'y1': y1Value, 'y2': y2Value, 'result_limit': RESULT_LIMIT,
                  'event_starttime': startYear + "-" + startMonth + "-" + startDay + "T" + startHour + ":" + startMinute + ":" + startSecond,
                  'event_endtime': endYear + "-" + endMonth + "-" + endDay + "T" + endHour + ":" + endMinute + ":" + endSecond}

        values1 = values.copy()
        print(values1)

        if filter:
            for i in range(0, len(filterCondition)):
                valuesLoop.append({'sparam' + str(i): filterCondition[i], 'op' + str(i): filterOperators[i],
                                   'value' + str(i): filterConditionValue[i]})

            for i in range(0, len(valuesLoop)):
                values1.update(valuesLoop[i])
            print(values1)

        data = urllib.parse.urlencode(values1)
        data = data.encode(ENCODING_SCHEME)
        req = urllib.request.Request(url, data)
        resp = urllib.request.urlopen(req)
        resData = resp.read().decode('utf-8')

        if json:
            fileName = "'" + eventModuleString + "'" + "_event_startdate=" + startYear.strip() + "-" + startMonth.strip() + "-" + startDay.strip() + "T" + startHour + "-" + startMinute + "-" + startSecond + "_event_enddate=" + endYear.strip() + "-" + endMonth.strip() + "-" + endDay.strip() + "T" + endHour + "-" + endMinute + "-" + endSecond + ".json"
        elif xml:
            fileName = "'" + eventModuleString + "'" + "_event_startdate=" + startYear.strip() + "-" + startMonth.strip() + "-" + startDay.strip() + "T" + startHour + "-" + startMinute + "-" + startSecond + "_event_enddate=" + endYear.strip() + "-" + endMonth.strip() + "-" + endDay.strip() + "T" + endHour + "-" + endMinute + "-" + endSecond + ".xml"

        saveFile = open(opDirectory + fileName, 'w')
        saveFile.write(str(resData))
        saveFile.close()

    except Exception as e:
        print('Error:', str(e))

    inputCsvFilePath = input("Enter output directory path for csv file: ")
    with open(opDirectory + fileName, 'r') as f:
        result_parsedData = j.load(f)
        resultParsed = result_parsedData['result']
    result_data = open(inputCsvFilePath + fileName + "_result.csv", 'w')
    csvwriter = csv.writer(result_data)
    count = 0
    for rslt in resultParsed:

        if count == 0:
            header = rslt.keys()
            csvwriter.writerow(header)
            count += 1
        csvwriter.writerow(rslt.values())
    result_data.close()

    inputCsvConfigFilePath = input("Enter input config file directory path with config file name: ")

    with open(inputCsvConfigFilePath, 'r') as f:
        contentsCSV = f.readlines()
        f.close()
        contentsCSV = [contentData.strip('\n') for contentData in contentsCSV]

    df = pd.read_csv(inputCsvFilePath + fileName + "_result.csv", usecols=contentsCSV)
    df.to_csv(inputCsvFilePath + fileName + "_Extractedresult.csv", encoding='utf-8', index=False)


def helioviewer():
    # try:
    #
    #     inputCsvDataFile = input("Enter input csv data file directory path with csv data file name: ")
    #     data = pd.read_csv(inputCsvDataFile, usecols=['event_starttime', 'event_endtime'])
    #     event_startTime = data.event_starttime.tolist()
    #     event_endTime = data.event_endtime.tolist()
    #     print(event_startTime)
    #     print(event_endTime)
    #
    #     for event_startTimeData in event_startTime:
    #         uriHelioviewer = "http://legacy.helioviewer.org/api/v1/getJP2Image/?"
    #         valuesImageUri = { "date": event_startTimeData+"Z","observatory": "SDO", "instrument": "AIA", "detector": "AIA", "measurement": "193", "jpip": "true"}
    #         dataImageUri = urllib.parse.urlencode(valuesImageUri)
    #         # dataImageUri = dataImageUri.encode('utf-8')
    #         # print(""+uriHelioviewer+dataImageUri)
    #         reqImageUri = urllib.request.Request(uriHelioviewer+dataImageUri)
    #         respImageUri = urllib.request.urlopen(reqImageUri)
    #         resImageUriData = respImageUri.read()
    #         resImageUriData= str(resImageUriData).split("193", 1)[1]
    #         resImageUriData = resImageUriData.replace(".jp2'","")
    #         resImageUriData=resImageUriData.strip()
    #         print(resImageUriData)
    #         urlHelioviewer = "http://legacy.helioviewer.org/api/v1/getJP2Image/?"
    #         valuesImage = {"date":event_startTimeData+"Z","observatory":"SDO","instrument": "AIA", "detector": "AIA", "measurement": "193"}
    #         dataImage = urllib.parse.urlencode(valuesImage)
    #         # print(urlHelioviewer+dataImage)
    #         # dataImage = dataImage.encode('utf-8')
    #         # reqImage = urllib.request.Request(urlHelioviewer,dataImage)
    #         directory = "F:\QueryHekPy\output\images"
    #         urllib.request.urlretrieve(urlHelioviewer+dataImage, directory+resImageUriData+".jp2")
    #         # img = Image(urlHelioviewer+dataImage, directory+resImageUriData+".jp2")
    #         # img.write(urlHelioviewer+dataImage, directory+resImageUriData+'.jpeg')
    #         # respImage = urllib.request.urlopen(reqImage)
    #         # resImageData = respImage.read()
    #
    #         # saveFile = open("F:\QueryHekPy\output\images"+resImageUriData+".jp2", 'w')
    #         # saveFile = open("F:\QueryHekPy\output\images" + "img.jp2", 'w')
    #         # saveFile.write(str(resImageData))
    #         # saveFile.close()
    #
    # except Exception as e:
    #     print('Error:', str(e))

    # Jp2 Jpeg format solar image downloader--------------------------------------------------------------------------------------------------------------------------------------

    try:

        CDELT = float(0.600000023842)
        HPCCENTER = float(4096.0 / 2.0)
        # hpc_bboxPolygon = "POLYGON((-471.9 536.7,-315.3 536.7,-315.3 685.5,-471.9 685.5,-471.9 536.7))"
        # Coordinates= hpc_bboxPolygon.replace('POLYGON((', '?').replace(" ",'?').replace(",", '?').replace("))",'?').split('?')
        # x1=round(float(HPCCENTER+(float(Coordinates[1])/CDELT)))
        # y1=round(float(HPCCENTER-(float(Coordinates[2])/CDELT)))
        # x2=round(float(HPCCENTER+(float(Coordinates[3])/CDELT)))
        # y2=round(float(HPCCENTER-(float(Coordinates[4])/CDELT)))
        # x3 = round(float(HPCCENTER+(float(Coordinates[5])/CDELT)))
        # y3 = round(float(HPCCENTER-(float(Coordinates[6])/CDELT)))
        # x4 = round(float(HPCCENTER+(float(Coordinates[7])/CDELT)))
        # y4 = round(float(HPCCENTER-(float(Coordinates[8])/CDELT)))
        # x5 = round(float(HPCCENTER+(float(Coordinates[9])/CDELT)))
        # y5 = round(float(HPCCENTER-(float(Coordinates[10])/CDELT)))
        # print(x1)
        # print(y1)
        # print(x2)
        # print(y2)
        # print(x3)
        # print(y3)
        # print(x4)
        # print(y4)
        # print(x5)
        # print(y5)
        inputCsvDataFile = input("Enter input csv data file directory path with csv data file name: ")
        dataNew = pd.read_csv(inputCsvDataFile, usecols=['track_id', 'event_starttime', 'event_endtime', 'hpc_bbox'])
        track_id = dataNew.track_id.tolist()
        event_startTime = dataNew.event_starttime.tolist()
        event_endTime = dataNew.event_endtime.tolist()
        hpc_bboxPolygon = dataNew.hpc_bbox.tolist()
        directoryJp2 = input("Enter Jp2 image output directory path: ")
        directoryJpg = input("Enter Jpeg image output directory path: ")
        directoryJpgCrop = input("Enter cropped Jpeg image with bbox output directory path: ")
        #directoryJpgPoly = input("Enter Jpeg image with polygon output directory path: ")
        counter = 0
        for event_startTimeData in event_startTime:
            track_idData = track_id[counter]
            event_endTimeDate = event_endTime[counter]
            hpc_bboxPolygonData = hpc_bboxPolygon[counter]
            Coordinates = hpc_bboxPolygonData.replace('POLYGON((', '?').replace(" ", '?').replace(",", '?').replace(
                "))", '?').split('?')
            x1 = round(float(HPCCENTER + (float(Coordinates[1]) / CDELT)))
            y1 = round(float(HPCCENTER - (float(Coordinates[2]) / CDELT)))
            x2 = round(float(HPCCENTER + (float(Coordinates[3]) / CDELT)))
            y2 = round(float(HPCCENTER - (float(Coordinates[4]) / CDELT)))
            x3 = round(float(HPCCENTER + (float(Coordinates[5]) / CDELT)))
            y3 = round(float(HPCCENTER - (float(Coordinates[6]) / CDELT)))
            x4 = round(float(HPCCENTER + (float(Coordinates[7]) / CDELT)))
            y4 = round(float(HPCCENTER - (float(Coordinates[8]) / CDELT)))
            event_startTimeDataFolder = event_startTimeData
            event_endTimeDateFolder = event_endTimeDate
            directoryJp2TrackID = directoryJp2 + "\\" + str(track_idData) + "_" + event_startTimeDataFolder.replace(":",
                                                                                                                    "_") + "_" + event_endTimeDateFolder.replace(
                ":", "_") + "_Count" + str(counter)
            print(directoryJp2TrackID)
            if not os.path.isdir(directoryJp2TrackID):
                os.makedirs(directoryJp2TrackID)
            directoryJpgTrackID = directoryJpg + "\\" + str(track_idData) + "_" + event_startTimeDataFolder.replace(":",
                                                                                                                    "_") + "_" + event_endTimeDateFolder.replace(
                ":", "_") + "_Count" + str(counter)
            if not os.path.isdir(directoryJpgTrackID):
                os.makedirs(directoryJpgTrackID)
                
            #
            directoryJpgCropTrackID = directoryJpgCrop + "\\" + str(track_idData) + "_" + event_startTimeDataFolder.replace(":",
                                                                                                                    "_") + "_" + event_endTimeDateFolder.replace(
                ":", "_") + "_Count" + str(counter)
            if not os.path.isdir(directoryJpgCropTrackID):
                os.makedirs(directoryJpgCropTrackID)                
                            
			#Create directory for saving the image with polygon drawing-->start		
            '''
            directoryJpgPolyTrackID = directoryJpgPoly + "\\" + str(
                track_idData) + "_" + event_startTimeDataFolder.replace(":",
                                                                        "_") + "_" + event_endTimeDateFolder.replace(
                ":", "_") + "_Count" + str(counter)
            if not os.path.isdir(directoryJpgPolyTrackID):
                os.makedirs(directoryJpgPolyTrackID)
            '''
			#Create directory for saving the image with polygon drawing-->end	
				
            counter = counter + 1
            start = datetime.datetime.strptime(event_startTimeData, "%Y-%m-%dT%H:%M:%S")
            end = datetime.datetime.strptime(event_endTimeDate, "%Y-%m-%dT%H:%M:%S")
            while start <= end:
                dateDownload = str(start.date()) + "T" + str(start.time()) + "Z"
                print(dateDownload)
                uriHelioviewer = "https://api.helioviewer.org/v2/getJP2Image/?"
                valuesImageUri = {"date": dateDownload, "sourceId": "10",
                                  "jpip": "true"}
                dataImageUri = urllib.parse.urlencode(valuesImageUri)
                reqImageUri = urllib.request.Request(uriHelioviewer + dataImageUri)
                respImageUri = urllib.request.urlopen(reqImageUri)
                resImageUriData = respImageUri.read()
                resImageUriData = str(resImageUriData).split("171", 1)[1]
                resImageUriData = resImageUriData.replace(".jp2'", "")
                resImageUriData = resImageUriData.strip()
                print(resImageUriData)
                urlHelioviewer = "https://api.helioviewer.org/v2/getJP2Image/?"
                valuesImage = {"date": dateDownload, "sourceId": "10"}
                dataImage = urllib.parse.urlencode(valuesImage)
                urllib.request.urlretrieve(urlHelioviewer + dataImage, directoryJp2TrackID + resImageUriData + ".jp2")
                img = Image(directoryJp2TrackID + resImageUriData + ".jp2")
                img.write(directoryJpgTrackID + resImageUriData + ".png")
                
                #CROP IMAGE WITH BOUNDING BOX
                
                image_obj = PILImage.open(directoryJpgTrackID + resImageUriData + ".png")
                cropped_image = image_obj.crop((x4, y4, x2, y2))
                cropped_image.save(directoryJpgCropTrackID + resImageUriData + ".png")
                
                #draw polygon--->start
                '''
                im2 = cv2.imread(directoryJpgTrackID + resImageUriData + ".png")
                pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(im2, [pts], True, (0, 0, 255), 5)
                cv2.imwrite(directoryJpgPolyTrackID + resImageUriData + ".png", im2)
                '''
				#draw polygon--->end here
				
                # im2=cv2.imread(directoryJpgTrackID + resImageUriData + ".jpeg")
                # cv2.rectangle(im2, (714, 2474), (809, 2528), (0, 0, 0), 5)
                # cv2.imwrite(directoryJpgPoly+resImageUriData + ".jpeg",im2)
                start = start + datetime.timedelta(seconds=36)
                print(start)


    except Exception as e:
        print('Error:', str(e))


def main():
    functionality = input("Choose functionality(hek/helioviewer): ")

    if "hek" in functionality:
        hek()
    elif "helioviewer" in functionality:
        helioviewer()


if __name__ == "__main__":
    main()
