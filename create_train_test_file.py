import os
import glob

text_files = glob.glob('E:\\master_thesis\\project\\vehicle_dataset\\CUSTOM_DATA_yolo//*.txt')

car = 0
truck = 0
bicycle = 0
bus = 0
pedestrian = 0
motor_bike = 0
total = 0
with open('E:\\master_thesis\\project\\vehicle_dataset\\test.txt', 'w') as f:
    for filename in text_files[:-1]:
        files = open(filename)
        strng_list = files.readlines()
        d1 = 0
        d2 = 0
        d3 = 0
        d4 = 0
        d5 = 0
        d6 = 0

        for i in strng_list:
            if int(i[0]) == 0:
                car += 1
                d1 = 1
            elif int(i[0]) == 1:
                truck += 1
                d2 = 1
            elif int(i[0]) == 2:
                bicycle += 1
                d3 = 1
            elif int(i[0]) == 3:
                bus += 1
                d4 = 1
            elif int(i[0]) == 4:
                pedestrian += 1
                d5 = 1
            elif int(i[0]) == 5:
                motor_bike += 1
                d6 = 1
        dd = d1 + d2 + d3 + d4 + d5 + d6

        if car <= 115 and d1 == 1 and dd == 1:
            line = filename[41:]
            line1 = line[:-4]
            line1 = line1[17:]
            f.write(line[:16]+'/'+line1+'.jpg'+'\n')
            text_files.remove(filename)
            total += 1
        elif truck <= 75 and d2 == 1 and dd == 1:
            line = filename[41:]
            line1 = line[:-4]
            line1 = line1[17:]
            f.write(line[:16]+'/'+line1+'.jpg'+'\n')
            text_files.remove(filename)
            total += 1
        elif bicycle <= 90 and d3 == 1 and dd == 1:
            line = filename[41:]
            line1 = line[:-4]
            line1 = line1[17:]
            f.write(line[:16]+'/'+line1+'.jpg'+'\n')
            text_files.remove(filename)
            total += 1
        elif bus <= 90 and d4 == 1 and dd == 1:
            line = filename[41:]
            line1 = line[:-4]
            line1 = line1[17:]
            f.write(line[:16]+'/'+line1+'.jpg'+'\n')
            text_files.remove(filename)
            total += 1
        elif pedestrian <= 85 and d5 == 1 and dd == 1:
            line = filename[41:]
            line1 = line[:-4]
            line1 = line1[17:]
            f.write(line[:16]+'/'+line1+'.jpg'+'\n')
            text_files.remove(filename)
            total += 1
        elif motor_bike <= 75 and d6 == 1 and dd == 1:
            line = filename[41:]
            line1 = line[:-4]
            line1 = line1[17:]
            f.write(line[:16]+'/'+line1+'.jpg'+'\n')
            text_files.remove(filename)
            total += 1
f.close()
with open('E:\\master_thesis\\project\\vehicle_dataset\\train.txt', 'w') as ff:
    for i in text_files:
        if i[41:] != 'CUSTOM_DATA_yolo\classes.txt':
            line = i[41:]
            line1 = line[:-4]
            line1 = line1[17:]
            ff.write(line[:16]+'/'+line1+'.jpg'+'\n')
ff.close()

