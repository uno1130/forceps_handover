from math import pi

import numpy as np
from FileIO.FileIO import FileIO
from Filter.Filter import MotionFilter


class xArmTransform:
    """
    xArmの座標と回転を保持するクラス
    """

    x, y, z = 0, 0, 0
    roll, pitch, yaw = 0, 0, 0

    # ----- Minimum limitation ----- #
    __minX, __minY, __minZ = -9999, -9999, -9999
    __minRoll, __minPitch, __minYaw = -9999, -9999, -9999

    # ----- Maximum limitation ----- #
    __maxX, __maxY, __maxZ = 9999, 9999, 9999
    __maxRoll, __maxPitch, __maxYaw = 9999, 9999, 9999

    def __init__(self, initpos: list, initrot: list, initangle: list):
        self.n = 4
        self.fp = 40
        self.fs = 180
        self.filter_robot = MotionFilter()
        self.filter_robot.InitLowPassFilterWithOrder(self.fs, self.fp, self.n)

        self.beforefilt = [[0, 0, 0, 0, 0, 0]] * self.n
        self.afterfilt = [[0, 0, 0, 0, 0, 0]] * self.n

        self.__initX, self.__initY, self.__initZ = float(initpos[0][1]), float(initpos[0][2]), float(initpos[0][3])
        self.__initRoll, self.__initPitch, self.__initYaw = float(initrot[0][1]), float(initrot[0][2]), float(initrot[0][3])
        self.__initAngleList = initangle

    def GetInitialTransform(self):
        """
        Get the initial position and rotation.
        """

        return self.__initX, self.__initY, self.__initZ, self.__initRoll, self.__initPitch, self.__initYaw
    
    def GetInitialAngle(self):
        return self.__initAngleList
    
    def Transform(self, relativepos: list, relativerot: list, isLimit=True, isOnlyPosition=False):
        relativepos_mm = relativepos * 1000
        x, y, z = relativepos_mm[0] + self.__initX, relativepos_mm[1] + self.__initY, relativepos_mm[2] + self.__initZ
        roll, pitch, yaw = relativerot[0] + self.__initRoll, relativerot[1] + self.__initPitch, relativerot[2] + self.__initYaw

        if isOnlyPosition:
            roll, pitch, yaw = self.__initRoll, self.__initPitch, self.__initYaw

        if isLimit:
            if x > self.__maxX:
                x = self.__maxX
            elif x < self.__minX:
                x = self.__minX

            if y > self.__maxY:
                y = self.__maxY
            elif y < self.__minY:
                y = self.__minY

            if z > self.__maxZ:
                z = self.__maxZ
            elif z < self.__minZ:
                z = self.__minZ

            if roll > self.__maxRoll:
                roll = self.__maxRoll
            elif roll < self.__minRoll:
                roll = self.__minRoll

            if pitch > self.__maxPitch:
                pitch = self.__maxPitch
            elif pitch < self.__minPitch:
                pitch = self.__minPitch

            if yaw > self.__maxYaw:
                yaw = self.__maxYaw
            elif yaw < self.__minYaw:
                yaw = self.__minYaw

        return np.array([x, y, z, roll, pitch, yaw])

    def Transform_LPF(self, relativepos: list, relativerot: list, isLimit=True, isOnlyPosition=False):
        """
        Converts from motive coordinate to xarm coordinate and limits values.
        """
        relativepos_mm = relativepos * 1000

        self.beforefilt.append([relativepos_mm[0], relativepos_mm[1], relativepos_mm[2], relativerot[0], relativerot[1], relativerot[2]])
        self.robotfilt = self.filter_robot.lowpass2(self.beforefilt, self.afterfilt)
        self.afterfilt.append(self.robotfilt)
        del self.beforefilt[0]
        del self.afterfilt[0]

        x, y, z = relativepos_mm[0] + self.__initX, relativepos_mm[1] + self.__initY, relativepos_mm[2] + self.__initZ
        roll, pitch, yaw = relativerot[0] + self.__initRoll, relativerot[1] + self.__initPitch, relativerot[2] + self.__initYaw

        if isOnlyPosition:
            roll, pitch, yaw = self.__initRoll, self.__initPitch, self.__initYaw

        if isLimit:
            if x > self.__maxX:
                x = self.__maxX
            elif x < self.__minX:
                x = self.__minX

            if y > self.__maxY:
                y = self.__maxY
            elif y < self.__minY:
                y = self.__minY

            if z > self.__maxZ:
                z = self.__maxZ
            elif z < self.__minZ:
                z = self.__minZ

            if roll > self.__maxRoll:
                roll = self.__maxRoll
            elif roll < self.__minRoll:
                roll = self.__minRoll

            if pitch > self.__maxPitch:
                pitch = self.__maxPitch
            elif pitch < self.__minPitch:
                pitch = self.__minPitch

            if yaw > self.__maxYaw:
                yaw = self.__maxYaw
            elif yaw < self.__minYaw:
                yaw = self.__minYaw

        return np.array([x, y, z, roll, pitch, yaw])
