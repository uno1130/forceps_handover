import time
import winsound
import csv
import os
import glob
import socket
import json
import matplotlib.pyplot as plt
from ctypes import windll
import numpy as np
from scipy.spatial.transform import Rotation as R
from FileIO.FileIO import FileIO
from Participant.ParticipantMotion import ParticipantMotion
from Recorder.DataRecord import DataRecordManager
from Robot.CAMotion import CAMotion
from Robot.xArmTransform import xArmTransform
from xarm.wrapper import XArmAPI


# ---------- Settings: Input mode ---------- #
motionDataInputMode = "optitrack"
gripperDataInputMode = "bendingsensor"

class ProcessorClass:
    def __init__(self) -> None:

        # setting.csvの読み込みのためにFile.IOクラスを使用　与えたcsvを多次元の配列にして返す
        fileIO = FileIO()
        dat = fileIO.Read("config/settings.csv", ",") # datの中身はdat.txtを参照

        # 以下では作成したdatから特定の部分のみ取り出している
        # リストの１要素目が特定の文字となるところをif文で探索する
        xArmIP_left = [addr for addr in dat if "xArmIPAddress_left" in addr[0]][0][1]
        initialpos_left = [addr for addr in dat if "initialpos_left" in addr[0]]
        initialrot_left = [addr for addr in dat if "initialrot_left" in addr[0]]
        initAngleList_left = [addr for addr in dat if "initAngleList_left" in addr[0]]

        xArmIP_right = [addr for addr in dat if "xArmIPAddress_right" in addr[0]][0][1]
        initialpos_right = [addr for addr in dat if "initialpos_right" in addr[0]]
        initialrot_right = [addr for addr in dat if "initialrot_right" in addr[0]]
        initAngleList_right = [addr for addr in dat if "initAngleList_right" in addr[0]]

        wirelessIP = [addr for addr in dat if "wirelessIPAddress" in addr[0]][0][1]
        localIP = [addr for addr in dat if "localIPAddress" in addr[0]][0][1]
        motiveserverIP = [addr for addr in dat if "motiveServerIPAddress" in addr[0]][0][1]
        motivelocalIP = [addr for addr in dat if "motiveLocalIPAddress" in addr[0]][0][1]
        frameRate = [addr for addr in dat if "frameRate" in addr[0]][0][1]

        lstmClientAddress = [addr for addr in dat if "lstmClientAddress" in addr[0]][0][1]
        lstmClientPort = [addr for addr in dat if "lstmClientPort" in addr[0]][0][1]
        lstmServerAddress = [addr for addr in dat if "lstmServerAddress" in addr[0]][0][1]
        lstmServerPort = [addr for addr in dat if "lstmServerPort" in addr[0]][0][1]

        isExportData =  [addr for addr in dat if "isExportData" in addr[0]][0][1]
        if isExportData == "False":
            isExportData = 0
        elif isExportData == "True":
            isExportData = 1
        dirPath = [addr for addr in dat if "dirPath" in addr[0]][0][1]

        participantNum = [addr for addr in dat if "participantNum" in addr[0]][0][1]
        gripperNum = [addr for addr in dat if "gripperNum" in addr[0]][0][1]
        otherRigidBodyNum = [addr for addr in dat if "otherRigidBodyNum" in addr[0]][0][1]
        robotNum = [addr for addr in dat if "robotNum" in addr[0]][0][1]
        idList = [addr for addr in dat if "idList" in addr[0]]
        differenceLimit = [addr for addr in dat if "differenceLimit" in addr[0]][0][1]

        recordedDataPath = [addr for addr in dat if "recordedDataPath" in addr[0]][0][1]

        weightListPos = [addr for addr in dat if "weightListPos" in addr[0]]
        weightListRot = [addr for addr in dat if "weightListRot" in addr[0]]
        practicemode = [addr for addr in dat if "practicemode" in addr[0]][0][1]
        recordNum = [addr for addr in dat if "recordNum" in addr[0]][0][1]

        # 上で作成した変数をインスタンス変数としている
        self.xArmIpAddress_left = xArmIP_left
        self.initialpos_left = initialpos_left
        self.initislrot_left = initialrot_left
        self.initAngleList_left =  list(map(float, initAngleList_left[0][1:]))

        self.xArmIpAddress_right = xArmIP_right
        self.initialpos_right = initialpos_right
        self.initislrot_right = initialrot_right
        self.initAngleList_right =  list(map(float, initAngleList_right[0][1:]))

        self.wirelessIpAddress = wirelessIP
        self.localIpAddress = localIP
        self.motiveserverIpAddress = motiveserverIP
        self.motivelocalIpAddress = motivelocalIP
        self.frameRate = int(frameRate)

        self.lstmClientAddress = lstmClientAddress
        self.lstmClientPort = int(lstmClientPort)
        self.lstmServerAddress = lstmServerAddress
        self.lstmServerPort = int(lstmServerPort)

        self.isExportData = bool(isExportData)
        self.dirPath = dirPath

        self.participantNum = int(participantNum)
        self.gripperNum = int(gripperNum)
        self.otherRigidBodyNum = int(otherRigidBodyNum)
        self.robotNum = int(robotNum)
        self.idList = idList

        self.differenceLimit = float(differenceLimit)

        self.recordedDataPath = recordedDataPath

        self.weightListPos = weightListPos
        self.weightListRot = weightListRot

        self.practicemode = int(practicemode)
        self.recordNum = int(recordNum)

    def mainloop(self, isEnablexArm: bool = True):
        """
        Send the position and rotation to the xArm
        """

        # ----- Process info ----- #
        self.loopCount = 0
        self.taskTime = []
        self.errorCount = 0
        taskStartTime = 0
        ratiolist = []
        timelist = []

        # ----- Instantiating custom classes ----- #
        # 各クラスのインスタンスを作成
        caMotion = CAMotion(defaultParticipantNum=self.participantNum, otherRigidBodyNum=self.otherRigidBodyNum,differenceLimit=self.differenceLimit)
        transform_left = xArmTransform(initpos=self.initialpos_left, initrot=self.initislrot_left, initangle=self.initAngleList_left)
        transform_right = xArmTransform(initpos=self.initialpos_right, initrot=self.initislrot_right, initangle=self.initAngleList_right)
        dataRecordManager = DataRecordManager(participantNum=self.recordNum, otherRigidBodyNum=self.otherRigidBodyNum, bendingSensorNum=self.gripperNum, robotNum=self.robotNum)
        participantMotion = ParticipantMotion(defaultParticipantNum=self.participantNum, otherRigidBodyNum=self.otherRigidBodyNum, motionInputSystem=motionDataInputMode, mocapServer=self.motiveserverIpAddress, mocapLocal=self.motivelocalIpAddress, idList=self.idList)

        # ----- Load recorded data. ----- #
        # 藤原先生の左右のcsvデータを読み込み、それぞれ変数に格納（３が左手、４が右手）
        for i in [3, 4]:
            participant_path = os.path.join(self.recordedDataPath, f"*Transform_Participant_{i-2}*.csv")
            globals()[f"participant{i}_data"] = self.load_csv_data(glob.glob(participant_path)[0])

        # ----- weight list ----- #
        # 融合割合を固定する場合に利用する
        # weightListRotのうち、実際の割合部分のみ抽出（順番：学習者左、学習者右、熟練者左、熟練者右）
        weightListPosfloat = list(map(float, self.weightListPos[0][1:]))
        weightListRotfloat = list(map(float, self.weightListRot[0][1:]))
        weightList = [weightListPosfloat, weightListRotfloat]

        # ----- Initialize robot arm ----- #
        # XArmの初期化を行う
        if isEnablexArm:
            # XArnAPIクラスを各ロボットのIPアドレスでインスタンス化する
            # そこで返ってきた値を用いてメソッドを実行することで、ロボットとの接続や初期化が行われる（関数 IniitializeAll内）
            arm_1 = XArmAPI(self.xArmIpAddress_left)
            self.InitializeAll(arm_1, transform_left)

            arm_2 = XArmAPI(self.xArmIpAddress_right)
            self.InitializeAll(arm_2, transform_right)

        # ----- Control flags ----- #
        isMoving = False

        try:
            while True:
                if isMoving:
                    # ----- Get relative----- #
                    localPosition = participantMotion.LocalPosition(loopCount=self.loopCount)
                    localRotation = participantMotion.LocalRotation(loopCount=self.loopCount)
                    relativePosition = caMotion.GetRelativePosition(position=localPosition)
                    relativeRotation = caMotion.GetRelativeRotation(rotation=localRotation)
                    print(localPosition, localRotation)

                    # ----- record ----- #
                    for i in [3, 4]:
                        relativePosition[f"participant{i}"] = np.array(globals()[f"participant{i}_data"][min(self.loopCount, len(globals()[f"participant{i}_data"]) - 1)]["position"])
                        relativeRotation[f"participant{i}"] = np.array(globals()[f"participant{i}_data"][min(self.loopCount, len(globals()[f"participant{i}_data"]) - 1)]["rotation"])

                    # ----- before or practice or after ----- #
                    if self.practicemode == 1 or self.practicemode == 2:
                        # ----- Difference calculation and transmission to transparent ----- #
                        average_diff, left_diff, right_diff = caMotion.calculate_difference(relativePosition)
                        self.frameRate = 200 - (average_diff / self.differenceLimit) * (200 - 50)
                        data_to_send = {"frameRate": self.frameRate, "average_diff": average_diff, "left_diff": left_diff, "right_diff": right_diff}
                        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                            sock.sendto(json.dumps(data_to_send).encode(), ('133.68.108.26', 8000))

                        # ----- Control ratio varies depending on the deference. ----- #
                        ratio_left = left_diff/self.differenceLimit
                        ratio_right = right_diff/self.differenceLimit
                        ratio_average = average_diff/self.differenceLimit
                        ratiolist.append(ratio_average)
                        timelist.append(time.perf_counter() - taskStartTime)
                        weightList = [[1-ratio_left, 1-ratio_right, ratio_left, ratio_right], [1-ratio_left, 1-ratio_right, ratio_left, ratio_right]]
                            
                        if self.loopCount >  len(globals()[f"participant3_data"]):
                            raise KeyboardInterrupt

                    # ----- Calculate the integration ----- #
                    robotpos, robotrot = caMotion.participant2robot(relativePosition, relativeRotation, weightList)
                
                    # ----- Send to xArm ----- #
                    if isEnablexArm:
                        if self.practicemode == 1:
                            arm_1.set_servo_cartesian(transform_left.Transform(relativepos=robotpos["robot1"], relativerot=robotrot["robot1"], isLimit=False))
                            arm_2.set_servo_cartesian(transform_right.Transform(relativepos=robotpos["robot2"], relativerot=robotrot["robot2"], isLimit=False))

                    # ----- Data recording ----- #
                    if self.isExportData:
                        dataRecordManager.Record(position=relativePosition, rotation=relativeRotation, weight=weightList, robotpos=robotpos, robotrot=robotrot, duration=time.perf_counter() - taskStartTime)

                    # ---------- fix framerate ---------- #
                    self.fix_framerate((time.perf_counter() - loop_start_time), 1/self.frameRate)
                    self.loopCount += 1
                    loop_start_time = time.perf_counter()

                    # ---------- time limit ---------- #
                    elapsed_time = time.perf_counter() - taskStartTime
                    if elapsed_time >= 120:
                        raise KeyboardInterrupt

                else:
                    keycode = input('Input > "q": quit, "r": Clean error and init arm, "s": start control \n')
                    # ----- Quit program ----- #
                    if keycode == "q":
                        if isEnablexArm:
                            arm_1.disconnect()
                            arm_2.disconnect()
                        self.PrintProcessInfo()

                        windll.winmm.timeEndPeriod(1)
                        break

                    # ----- Reset xArm and gripper ----- #
                    elif keycode == "r":
                        if isEnablexArm:
                            self.InitializeAll(arm_1, transform_left)
                            self.InitializeAll(arm_2, transform_right)

                    # ----- Start streaming ----- #
                    elif keycode == "s":
                        # ----- A beep sounds after 5 seconds and send s-key to the Mac side ----- #
                        time.sleep(5)
                        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                            sock.sendto(b's', ('133.68.108.26', 8000))
                        winsound.Beep(1000,1000)

                        # ----- set initial pos and rot ----- #
                        caMotion.SetOriginPosition(participantMotion.LocalPosition())
                        caMotion.SetInversedMatrix(participantMotion.LocalRotation())

                        # ----- flag and tasktime ----- #
                        isMoving = True
                        taskStartTime = loop_start_time = time.perf_counter()

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt >> Stop: mainloop()")

            self.taskTime.append(time.perf_counter() - taskStartTime)
            self.PrintProcessInfo()

            # Mac側にsキーを送信
            if self.loopCount > 100:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.sendto(b's', ('133.68.108.26', 8000))
                    print("Send stop to transparent")

                winsound.Beep(1000,1000)

            if self.isExportData:
                dataRecordManager.ExportSelf(dirPath=self.dirPath)

            # ----- Disconnect ----- #
            if isEnablexArm:
                arm_1.disconnect()
                arm_2.disconnect()

            windll.winmm.timeEndPeriod(1)

            if self.loopCount > 100:
                plt.plot(timelist, ratiolist, linestyle='-')
                plt.xlabel('Time')
                plt.ylabel('Ratio')
                plt.ylim(0, 1)
                if self.isExportData:
                    plt.savefig(self.dirPath + "/" + "ratio.png")
                plt.show()

        except:
            print("----- Exception has occurred -----")
            windll.winmm.timeEndPeriod(1)
            import traceback

            traceback.print_exc()

    def ConvertToModbusData(self, value: int):
        """
        Converts the data to modbus type.

        Parameters
        ----------
        value: int
            The data to be converted.
            Range: 0 ~ 800
        """

        if int(value) <= 255 and int(value) >= 0:
            dataHexThirdOrder = 0x00
            dataHexAdjustedValue = int(value)

        elif int(value) > 255 and int(value) <= 511:
            dataHexThirdOrder = 0x01
            dataHexAdjustedValue = int(value) - 256

        elif int(value) > 511 and int(value) <= 767:
            dataHexThirdOrder = 0x02
            dataHexAdjustedValue = int(value) - 512

        elif int(value) > 767 and int(value) <= 1123:
            dataHexThirdOrder = 0x03
            dataHexAdjustedValue = int(value) - 768

        modbus_data = [0x08, 0x10, 0x07, 0x00, 0x00, 0x02, 0x04, 0x00, 0x00]
        modbus_data.append(dataHexThirdOrder)
        modbus_data.append(dataHexAdjustedValue)

        return modbus_data

    def PrintProcessInfo(self):
        """
        Print process information.
        """

        print("----- Process info -----")
        print("Total loop count > ", self.loopCount)
        for ttask in self.taskTime:
            print("Task time\t > ", "{:.2f}".format(ttask), "[s]")
            print("Frame Rate\t > ", "{:.2f}".format(self.loopCount/ttask), "[fps]")
        print("Error count\t > ", self.errorCount)
        print("------------------------")

    def InitializeAll(self, robotArm, transform, isSetInitPosition=True, isSetInitAngle=True):
        """
        Initialize the xArm

        Parameters
        ----------
        robotArm: XArmAPI
            XArmAPI object.
        transform: xArmTransform
            xArmTransform object.
        isSetInitPosition: (Optional) bool
            True -> Set to "INITIAL POSITION" of the xArm studio
            False -> Set to "ZERO POSITION" of the xArm studio
        """
        # 以下XArm提供のプログラム（SDK）
        robotArm.connect()
        if robotArm.warn_code != 0:
            robotArm.clean_warn()
        if robotArm.error_code != 0:
            robotArm.clean_error()
        robotArm.motion_enable(enable=True)
        robotArm.set_mode(0)  # set mode: position control mode
        robotArm.set_state(state=0)  # set state: sport state
        # if isSetInitAngle:
        #     init_angle_list = transform.GetInitialAngle()
        #     robotArm.set_servo_angle(angle=init_angle_list, is_radian=False, wait=True)
        if isSetInitPosition:
            initX, initY, initZ, initRoll, initPitch, initYaw = transform.GetInitialTransform()
            robotArm.set_position(x=initX, y=initY, z=initZ, roll=initRoll, pitch=initPitch, yaw=initYaw, wait=True)
        else:
            robotArm.reset(wait=True)
        print("Initialized > xArm")

        robotArm.set_mode(1)
        robotArm.set_state(state=0)

    def fix_framerate(self, process_duration, looptime):
        sleeptime = looptime - process_duration
        if sleeptime < 0:
            pass
        else:
            time.sleep(sleeptime)

    def load_csv_data(self, file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            data = []
            for row in reader:
                data.append({
                    "time": float(row["time"]),
                    "position": [float(row["x"]), float(row["y"]), float(row["z"])],
                    "rotation": [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]
                })
        return data