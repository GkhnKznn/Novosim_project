"""
Copyright (C) 2022 by Crystal Instruments Corporation.
All rights reserved

MQTT Python Example - Running a Test and requesting a signal frame data
"""

import mqtt
import time
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import glob
import sys

# Connect to MQTT Broker
mqttClient = mqtt.pubsub(brokerIP = "192.168.1.34")
mqttClient.subclient.loop_start()

# initial sleep is needed to get system status messages
time.sleep(2)


dosya_konumu = r'C:\Users\GkhnKznn\Documents\EDM\Spider_DSA\FFT2\Default Folder 1-21-2024 3-13-27 PM\BlackBox\24663680'

# Belirtilen konumdaki tüm dosyaları bul
dosya_listesi = glob.glob(os.path.join(dosya_konumu, '*'))

# Dosyaları sil
for dosya in dosya_listesi:
    try:
        os.remove(dosya)
        print(f"{dosya} silindi.")
    except Exception as e:
        print(f"{dosya} silinirken bir hata oluştu: {e}")


r = input("\nPress enter once pre-test is done")
mqttClient.run()

# any other key followed by enter will exit, or ctrl-c will exit
if r != '':
    exit()

# Start test after pre-test
mqttClient.proceed()
time.sleep(2)

testStatus = mqttClient.LUT['EDM/App/Test/Status'].split("Status")[1].split(",")[0][3:-1]

signalFrame = []
count = 0
plt.ion()

try:
    while(testStatus == "Running"):
        r = input("Programı durdurmak için 's' tuşuna basın: ")
        if r.lower() == 's':
            break
        # Request signal data, such as Channel 1
        mqttClient.get_channel_data(1)
        # Wait for receiving and parsing message
        #  Change time to receive data in real time or occasional updates
        time.sleep(0.01) 
        # Refresh graph to view the current signal frame 
        # or comment it out to have a time history graph of the signal
        plt.clf()


        ## Parsing the received message ##
        signalName = mqttClient.LUT['EDM/App/Test/SignalData'].split("ValueX")[0].split("Name")[1].split(",")[0][3:-1]
        signalUnitX = mqttClient.LUT['EDM/App/Test/SignalData'].split("ValueX")[0].split("UnitX")[1].split(",")[0][3:-1]
        signalUnitY = mqttClient.LUT['EDM/App/Test/SignalData'].split("ValueX")[0].split("UnitY")[1].split(",")[0][3:-1]

        Xvalues = mqttClient.LUT['EDM/App/Test/SignalData'].split("ValueX")[1].split("ValueY")[0].split("ValueZ")[0][3:-3]
        X = Xvalues.split(",")
        X = np.fromiter(X, float)

        Yvalues = mqttClient.LUT['EDM/App/Test/SignalData'].split("ValueX")[1].split("ValueY")[1].split("ValueZ")[0][3:-3]
        Y = Yvalues.split(",")
        Y = np.fromiter(Y, float)

        #Zvalues = mqttClient.LUT['EDM/App/Test/SignalData'].split("ValueX")[1].split("ValueY")[1].split("ValueZ")[1][3:-3]
        #Z = Zvalues.split(",")
        #Z = np.fromiter(Z, float)

        signalFrame.append(X)
        signalFrame.append(Y)
        #signalFrame.append(Z)

        thd = np.array(signalFrame)


        ## Print Statements for displaying points of the signal frame ##
        # columnlabelstring = "X Values || Y Values || Z Value"
        # print("\n" + " " * (len(columnlabelstring)//4) + "Signal Frame Data\n" + "-" * len(columnlabelstring))
        # print(columnlabelstring + "\n")

        # nspace1 = 5
        # nspace2 = 11
        # nspace3 = 13

        # for x in range(len(signalFrame[0] / 4)):
        #     print(" "*nspace1, signalFrame[0][x], " "*nspace2, signalFrame[1][x]," "*nspace3, signalFrame[2][0])


        ## Save data to .npy files in the created save folder above ##
        #np.save(savedirectory + "/fullSignalFrame"+str(count)+".npy", thd)
        # # Save individual slices too
        # np.save(savedirectory + "/X"+str(count)+".npy", thd[0])
        # np.save(savedirectory + "/Y"+str(count)+".npy", thd[1])
        # np.save(savedirectory + "/Z"+str(count)+".npy", thd[2])

        # print("\nSaved .npy files to ", savedirectory)


        ## Generate plot ##
        # Adjust the X values for time domain signals
        # comment out for frequency domain signals
        thd[0] = thd[0] - thd[1][0]  #thd[0] = thd[0] - thd[2][0]

        plt.figure(1)
        plt.plot(thd[0], thd[1], 'r', label=signalName)
        plt.title("Signal Frame Data of " + signalName)
        plt.xlabel(signalUnitX)
        plt.ylabel(signalUnitY)
        plt.draw()
        # plt.savefig(savedirectory + "/"+signalName+"-plot"+str(count)+".png")
        plt.pause(0.1)

        count += 1
        signalFrame = []
        testStatus = mqttClient.LUT['EDM/App/Test/Status'].split("Status")[1].split(",")[0][3:-1]

except KeyboardInterrupt:
    print('Keyboard Interrupt received -- exiting main loop')

plt.ioff()  # Etkileşimli modu kapat

# For any interruption from the python script, stop the test
mqttClient.stop()
print("Test durduruldu")

try:
    mqttClient.subclient.loop_stop()
    print("MQTT loop stopped successfully")
except:
    print("Failed to stop subscriber loop")
