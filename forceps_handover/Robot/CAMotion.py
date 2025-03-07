import math
import numpy as np
import scipy.spatial.transform as scitransform
from Filter.Filter import MotionFilter
from scipy.spatial.transform import Rotation as R, Slerp

"""
##### IMPORTANT #####
If you are using average rotations of three or more people, please cite the following paper
see also: https://github.com/UCL/scikit-surgerycore

Thompson S, Dowrick T, Ahmad M, et al.
SciKit-Surgery: compact libraries for surgical navigation.
International Journal of Computer Assisted Radiology and Surgery. May 2020.
DOI: 10.1007/s11548-020-02180-5
"""
import sksurgerycore.algorithms.averagequaternions as aveq


class CAMotion:
    originPositions = {}
    inversedMatrixforPosition = {}
    inversedMatrix = {}

    beforePositions = {}
    weightedPositions = {}

    beforeRotations = {}
    weightedRotations = {}

    beforePositionsRobot = {}
    beforeRotationsRobot = {}

    def __init__(self, defaultParticipantNum: int, otherRigidBodyNum: int, differenceLimit: float) -> None:
        for i in range(4):
            self.originPositions["participant" + str(i + 1)] = np.zeros(3)
            self.inversedMatrixforPosition["participant" + str(i + 1)] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.inversedMatrix["participant" + str(i + 1)] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            self.beforePositions["participant" + str(i + 1)] = np.zeros(3)
            self.weightedPositions["participant" + str(i + 1)] = np.zeros(3)

            self.beforeRotations["participant" + str(i + 1)] = np.array([0, 0, 0, 1])
            self.weightedRotations["participant" + str(i + 1)] = np.array([0, 0, 0, 1])
        
        for i in range(2):
            self.beforePositionsRobot["robot" + str(i + 1)] = np.zeros(3)
            self.beforeRotationsRobot["robot" + str(i + 1)] = np.array([0, 0, 0, 1])

        for i in range(otherRigidBodyNum):
            self.originPositions["otherRigidBody" + str(i + 1)] = np.zeros(3)
            self.inversedMatrix["otherRigidBody" + str(i + 1)] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        self.participantNum = defaultParticipantNum
        self.otherRigidBodyNum = otherRigidBodyNum
        self.before_position = [[0, 0, 0], [0, 0, 0]]
        self.customweightPosition = [0, 0, 0]
        self.before_sharedPosition = [0, 0, 0]
        self.differenceLimit = differenceLimit

        n = 2
        fp = 10
        fs = 700
        self.filter_FB = MotionFilter()
        self.filter_FB.InitLowPassFilterWithOrder(fs, fp, n)
        self.get_pos_1_box = [[0]] * n
        self.get_pos_1_filt_box = [[0]] * n
        self.get_pos_2_box = [[0]] * n
        self.get_pos_2_filt_box = [[0]] * n
        self.get_rot_1_box = [[0]] * n
        self.get_rot_1_filt_box = [[0]] * n
        self.get_rot_2_box = [[0]] * n
        self.get_rot_2_filt_box = [[0]] * n

    def participant2robot(self, motive_position: dict, motive_rotation: dict, weight: list):
        # ----- change cordinate from motive to xArm ----- #
        order_pos = [2, 1, 0]
        order_rot = [2, 1, 0, 3]
        xarm_position = self.motive2xarm(motive_position, order_pos, 4, [1, 2])
        xarm_rotation = self.motive2xarm(motive_rotation, order_rot, 4, [1, 2])

        for i in range(4):
            # ----- position ----- #
            weightedDiffPos = (np.array(xarm_position["participant" + str(i + 1)]) - np.array(self.beforePositions["participant" + str(i + 1)])) * weight[0][i]

            if i % 2 == 0:
                sharedPosition_left = self.beforePositionsRobot["robot1"] + weightedDiffPos
                self.beforePositionsRobot["robot1"] = sharedPosition_left
            elif i % 2 == 1:
                sharedPosition_right = self.beforePositionsRobot["robot2"] + weightedDiffPos
                self.beforePositionsRobot["robot2"] = sharedPosition_right

            self.beforePositions["participant" + str(i + 1)] = xarm_position["participant" + str(i + 1)]

            # ----- rotation ----- #
            slerpRot = self.weighted_slerp(self.beforeRotations["participant" + str(i + 1)], xarm_rotation["participant" + str(i + 1)], weight[1][i])
            qw, qx, qy, qz = self.beforeRotations["participant" + str(i + 1)][3], self.beforeRotations["participant" + str(i + 1)][0], self.beforeRotations["participant" + str(i + 1)][1], self.beforeRotations["participant" + str(i + 1)][2]
            mat4x4 = np.array([
                    [qw, qz, -qy, qx],
                    [-qz, qw, qx, qy],
                    [qy, -qx, qw, qz],
                    [-qx, -qy, -qz, qw]])
            weightedDiffRot = np.dot(np.linalg.inv(mat4x4), slerpRot)

            if i % 2 == 0:
                sharedRotation_quaternion_left = (R.from_quat(weightedDiffRot) * R.from_quat(self.beforeRotationsRobot["robot1"])).as_quat()
                self.beforeRotationsRobot["robot1"] = sharedRotation_quaternion_left
            elif i % 2 == 1:
                sharedRotation_quaternion_right = (R.from_quat(weightedDiffRot) * R.from_quat(self.beforeRotationsRobot["robot2"])).as_quat()
                self.beforeRotationsRobot["robot2"] = sharedRotation_quaternion_right

            self.beforeRotations["participant" + str(i + 1)] = xarm_rotation["participant" + str(i + 1)]
            
        sharedRotation_euler_left = self.Quaternion2Euler(sharedRotation_quaternion_left)
        sharedRotation_euler_right = self.Quaternion2Euler(sharedRotation_quaternion_right)

        self.posarm = dict(robot1=sharedPosition_left, robot2=sharedPosition_right)
        self.rotarm = dict(robot1=np.concatenate((sharedRotation_euler_left, sharedRotation_quaternion_left)), robot2=np.concatenate((sharedRotation_euler_right, sharedRotation_quaternion_right)))

        return self.posarm, self.rotarm

    # ----------------------------------------------------------------------------------------------------------------------------------------------

    def SetOriginPosition(self, position) -> None:
        """
        Set the origin position

        Parameters
        ----------
        position: dict, numpy array
            Origin position
        """
        # ----- numpy array to dict: position ----- #
        if type(position) is np.ndarray:
            position = self.NumpyArray2Dict(position)

        for i in range(self.participantNum):
            self.originPositions["participant" + str(i + 1)] = position["participant" + str(i + 1)]

        for i in range(self.otherRigidBodyNum):
            self.originPositions["otherRigidBody" + str(i + 1)] = position["otherRigidBody" + str(i + 1)]

    def GetRelativePosition(self, position) -> None:
        """
        Get the relative position

        Parameters
        ----------
        position: dict, numpy array
            Position to compare with the origin position.
            [x, y, z]

        Returns
        ----------
        relativePos: dict
            Position relative to the origin position.
            [x, y, z]
        """

        # ----- numpy array to dict: position ----- #
        if type(position) is np.ndarray:
            position = self.NumpyArray2Dict(position)

        relativePos = {}
        for i in range(self.participantNum):
            relativePos["participant" + str(i + 1)] = position["participant" + str(i + 1)] - self.originPositions["participant" + str(i + 1)]

        for i in range(self.otherRigidBodyNum):
            relativePos["otherRigidBody" + str(i + 1)] = position["otherRigidBody" + str(i + 1)] - self.originPositions["otherRigidBody" + str(i + 1)]

        return relativePos

    def SetInversedMatrix(self, rotation) -> None:
        """
        Set the inversed matrix

        Parameters
        ----------
        rotation: dict, numpy array
            Quaternion.
            Rotation for inverse matrix calculation
        """

        # ----- numpy array to dict: rotation ----- #
        if type(rotation) is np.ndarray:
            rotation = self.NumpyArray2Dict(rotation)

        for i in range(self.participantNum):
            q = rotation["participant" + str(i + 1)]
            qw, qx, qy, qz = q[3], q[1], q[2], q[0]
            mat4x4 = np.array([ [qw, -qy, qx, qz],
                                [qy, qw, -qz, qx],
                                [-qx, qz, qw, qy],
                                [-qz, -qx, -qy, qw]])
            self.inversedMatrix["participant" + str(i + 1)] = np.linalg.inv(mat4x4)

        for i in range(self.otherRigidBodyNum):
            q = rotation["otherRigidBody" + str(i + 1)]
            qw, qx, qy, qz = q[3], q[1], q[2], q[0]
            mat4x4 = np.array([ [qw, -qy, qx, qz],
                                [qy, qw, -qz, qx],
                                [-qx, qz, qw, qy],
                                [-qz, -qx, -qy, qw]])
            self.inversedMatrix["otherRigidBody" + str(i + 1)] = np.linalg.inv(mat4x4)


    def GetRelativePosition_r(self, position):
        """
        Get the relative position
        Parameters
        ----------
        position: dict, numpy array
            Position to compare with the origin position.
            [x, y, z]

        Returns
        ----------
        relativePos: dict
            Position relative to the origin position.
            [x, y, z]
        """

        # ----- numpy array to dict: position ----- #
        if type(position) is np.ndarray:
            position = self.NumpyArray2Dict(position)

        relativePos = {}
        for i in range(self.participantNum):
            relativePos["participant" + str(i + 1)] = np.dot(self.inversedMatrixforPosition["participant" + str(i + 1)], position["participant" + str(i + 1)] - self.originPositions["participant" + str(i + 1)])
        relativePos["endEffector"] = np.dot(self.inversedMatrixforPosition["endEffector"], position["endEffector"] - self.originPositions["endEffector"])

        return relativePos

    def GetRelativeRotation(self, rotation):
        """
        Get the relative rotation

        Parameters
        ----------
        rotation: dict, numpy array
            Rotation to compare with the origin rotation.
            [x, y, z, w]

        Returns
        ----------
        relativeRot: dict
            Rotation relative to the origin rotation.
            [x, y, z, w]
        """

        # ----- numpy array to dict: rotation ----- #
        if type(rotation) is np.ndarray:
            rotation = self.NumpyArray2Dict(rotation)

        relativeRot = {}
        for i in range(self.participantNum):
            relativeRot["participant" + str(i + 1)] = np.dot(self.inversedMatrix["participant" + str(i + 1)], rotation["participant" + str(i + 1)])

        for i in range(self.otherRigidBodyNum):
            relativeRot["otherRigidBody" + str(i + 1)] = np.dot(self.inversedMatrix["otherRigidBody" + str(i + 1)], rotation["otherRigidBody" + str(i + 1)])

        return relativeRot

    def GetRelativeRotation_r(self, rotation):
        """
        Get the relative rotation

        Parameters
        ----------
        rotation: dict, numpy array
            Rotation to compare with the origin rotation.
            [x, y, z, w]

        Returns
        ----------
        relativeRot: dict
            Rotation relative to the origin rotation.
            [x, y, z, w]
        """

        # ----- numpy array to dict: rotation ----- #
        if type(rotation) is np.ndarray:
            rotation = self.NumpyArray2Dict(rotation)

        relativeRot = {}
        for i in range(self.participantNum):
            relativeRot["participant" + str(i + 1)] = np.dot(
                self.inversedMatrix["participant" + str(i + 1)],
                rotation["participant" + str(i + 1)],
            )
        relativeRot["endEffector"] = np.dot(
            self.inversedMatrix["endEffector"], rotation["endEffector"]
        )

        return relativeRot

    # def Quaternion2Euler(self, q, isDeg: bool = True):
    #     """
    #     クォータニオンからオイラー角を計算します。

    #     回転行列
    #     |m00 m01 m02 0|
    #     |m10 m11 m12 0|
    #     |m20 m21 m22 0|
    #     | 0   0   0  1|

    #     Parameters
    #     ----------
    #     q: np.ndarray
    #         クォータニオン。
    #         [x, y, z, w]
    #     isDeg: (Optional) bool
    #         このフラグがTrueの場合、返される角度は度単位になります。デフォルトはTrueです。

    #     Returns
    #     ----------
    #     rotEuler: np.ndarray
    #         オイラー角。
    #         [x, y, z]
    #     """
    #     qx = q[0]
    #     qy = q[1]
    #     qz = q[2]
    #     qw = q[3]

    #     # 回転行列の要素を計算
    #     m00 = 1 - 2 * (qy**2 + qz**2)
    #     m01 = 2 * (qx * qy + qw * qz)
    #     m02 = 2 * (qx * qz - qw * qy)
    #     m10 = 2 * (qx * qy - qw * qz)
    #     m11 = 1 - 2 * (qx**2 + qz**2)
    #     m12 = 2 * (qy * qz + qw * qx)
    #     m20 = 2 * (qx * qz + qw * qy)
    #     m21 = 2 * (qy * qz - qw * qx)
    #     m22 = 1 - 2 * (qx**2 + qy**2)

    #     # ジンバルロック条件を確認
    #     if m02 == 1:
    #         tx = math.atan2(m10, m11)
    #         ty = math.pi / 2
    #         tz = 0
    #     elif m02 == -1:
    #         tx = math.atan2(m21, m20)
    #         ty = -math.pi / 2
    #         tz = 0
    #     else:
    #         tx = math.atan2(m12, m22)
    #         ty = math.asin(m02)
    #         tz = math.atan2(m01, m00)

    #     if isDeg:
    #         tx = np.rad2deg(tx)
    #         ty = np.rad2deg(ty)
    #         tz = np.rad2deg(tz)

    #     rotEuler = np.array([tx, ty, tz])
    #     return rotEuler


    def Quaternion2Euler(self, q, isDeg: bool = True):
        """
        Calculate the Euler angle from the Quaternion.


        Rotation matrix
        |m00 m01 m02 0|
        |m10 m11 m12 0|
        |m20 m21 m22 0|
        | 0   0   0  1|

        Parameters
        ----------
        q: np.ndarray
            Quaternion.
            [x, y, z, w]
        isDeg: (Optional) bool
            Returned angles are in degrees if this flag is True, else they are in radians.
            The default is True.

        Returns
        ----------
        rotEuler: np.ndarray
            Euler angle.
            [x, y, z]
        """

        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]

        # 1 - 2y^2 - 2z^2
        m00 = 1 - (2 * qy**2) - (2 * qz**2)
        # 2xy + 2wz
        m01 = (2 * qx * qy) + (2 * qw * qz)
        # 2xz - 2wy
        m02 = (2 * qx * qz) - (2 * qw * qy)
        # 2xy - 2wz
        m10 = (2 * qx * qy) - (2 * qw * qz)
        # 1 - 2x^2 - 2z^2
        m11 = 1 - (2 * qx**2) - (2 * qz**2)
        # 2yz + 2wx
        m12 = (2 * qy * qz) + (2 * qw * qx)
        # 2xz + 2wy
        m20 = (2 * qx * qz) + (2 * qw * qy)
        # 2yz - 2wx
        m21 = (2 * qy * qz) - (2 * qw * qx)
        # 1 - 2x^2 - 2y^2
        m22 = 1 - (2 * qx**2) - (2 * qy**2)

        # 回転軸の順番がX->Y->Zの固定角(Rz*Ry*Rx)
        # if m01 == -1:
        # 	tx = 0
        # 	ty = math.pi/2
        # 	tz = math.atan2(m20, m10)
        # elif m20 == 1:
        # 	tx = 0
        # 	ty = -math.pi/2
        # 	tz = math.atan2(m20, m10)
        # else:
        # 	tx = -math.atan2(m02, m00)
        # 	ty = -math.asin(-m01)
        # 	tz = -math.atan2(m21, m11)

        # 回転軸の順番がX->Y->Zのオイラー角(Rx*Ry*Rz)
        if m02 == 1:
            tx = math.atan2(m10, m11)
            ty = math.pi / 2
            tz = 0
        elif m02 == -1:
            tx = math.atan2(m21, m20)
            ty = -math.pi / 2
            tz = 0
        else:
            tx = -math.atan2(-m12, m22)
            ty = -math.asin(m02)
            tz = -math.atan2(-m01, m00)

        if isDeg:
            tx = np.rad2deg(tx)
            ty = np.rad2deg(ty)
            tz = np.rad2deg(tz)

        rotEuler = np.array([tx, ty, tz])
        return rotEuler

    def ScipyQuaternion2Euler(self, q, sequence: str = "xyz", isDeg: bool = True):
        """
        Calculate the Euler angle from the Quaternion.
        Using scipy.spatial.transform.Rotation.as_euler

        Parameters
        ----------
        q: np.ndarray
            Quaternion.
            [x, y, z, w]
        sequence: (Optional) str
            Rotation sequence of Euler representation, specified as a string.
            The rotation sequence defines the order of rotations about the axes.
            The default is xyz.
        isDeg: (Optional) bool
            Returned angles are in degrees if this flag is True, else they are in radians.
            The default is True.

        Returns
        ----------
        rotEuler: np.ndarray
            Euler angle.
            [x, y, z]
        """

        quat = scitransform.Rotation.from_quat(q)
        rotEuler = quat.as_euler(sequence, degrees=isDeg)
        return rotEuler

    def Euler2Quaternion(self, e):
        """
        Calculate the Quaternion from the Euler angle.

        Parameters
        ----------
        e: np.ndarray
            Euler.
            [x, y, z]

        Returns
        ----------
        rotQuat: np.ndarray
            Quaternion
            [x, y, z, w]
        """

        roll = np.deg2rad(e[0])
        pitch = np.deg2rad(e[1])
        yaw = np.deg2rad(e[2])

        cosRoll = np.cos(roll / 2.0)
        sinRoll = np.sin(roll / 2.0)
        cosPitch = np.cos(pitch / 2.0)
        sinPitch = np.sin(pitch / 2.0)
        cosYaw = np.cos(yaw / 2.0)
        sinYaw = np.sin(yaw / 2.0)

        q0 = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw
        q1 = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw
        q2 = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw
        q3 = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw

        rotQuat = [q1, q2, q3, q0]
        return rotQuat

    def ScipyEuler2Quaternion(self, e, sequence: str = "xyz", isDeg: bool = True):
        """
        Calculate the Quaternion from the Euler angle.
        Using scipy.spatial.transform.Rotation.as_quat

        Parameters
        ----------
        e: np.ndarray
            Euler.
            [x, y, z]
        sequence: (Optional) str
            Rotation sequence of Euler representation, specified as a string.
            The rotation sequence defines the order of rotations about the axes.
            The default is xyz.
        isDeg: (Optional) bool
            If True, then the given angles are assumed to be in degrees. Default is True.

        Returns
        ----------
        rotQuat: np.ndarray
            Quaternion
            [x, y, z, w]
        """

        quat = scitransform.Rotation.from_euler(sequence, e, isDeg)
        rotQuat = quat.as_quat()
        return rotQuat

    def InversedRotation(self, rot, axes: list = []):
        """
        Calculate the inversed rotation.

        ----- CAUTION -----
        If "axes" is set, it will be converted to Euler angles during the calculation process, which may result in inaccurate rotation.
        In addition, the behavior near the singularity is unstable.

        Parameters
        ----------
        rot: np.ndarray
            Quaternion.
            [x, y, z, w]
        axes: (Optional) list[str]
            Axes to be inversed.
            If length of axes is zero, return inversed quaternion

        Returns
        ----------
        inversedRot: np.ndarray
            Inversed rotation
            [x, y, z, w]
        """

        if len(axes) == 0:
            quat = scitransform.Rotation.from_quat(rot)
            inversedRot = quat.inv().as_quat()
            return inversedRot

        rot = self.ScipyQuaternion2Euler(rot)

        for axis in axes:
            if axis == "x":
                rot[0] = -rot[0]
            elif axis == "y":
                rot[1] = -rot[1]
            elif axis == "z":
                rot[2] = -rot[2]

        inversedRot = self.ScipyEuler2Quaternion(rot)

        return inversedRot

    def NumpyArray2Dict(self, numpyArray, dictKey: str = "participant"):
        """
        Convert numpy array to dict.

        Parameters
        ----------
        numpyArray: numpy array
            Numpy array.
        dictKey: (Optional) str
            The key name of the dict.
        """

        if type(numpyArray) is np.ndarray:
            dictionary = {}
            if len(numpyArray.shape) == 1:
                dictionary[dictKey + str(1)] = numpyArray
            else:
                for i in range(len(numpyArray)):
                    dictionary[dictKey + str(i + 1)] = numpyArray[i]
        else:
            print("Type Error: argument is NOT Numpy array")
            return

        return dictionary

    def quaternion_to_rotation_matrix(self, quaternion):
        """
        クォータニオンを回転行列に変換する関数
        """
        x, y, z, w = quaternion
        rotation_matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        return rotation_matrix

    def weighted_slerp(self, beforeRot, currentRot, weight):
        """
        クォータニオンでのSlerpを使用して重み付けした補間を行う。

        Parameters:
        beforeRot : numpy array, 以前のクォータニオン [x, y, z, w]
        currentRot : numpy array, 現在のクォータニオン [x, y, z, w]
        weight : float, 補間の重み（0.0から1.0の範囲）

        Returns:
        weightedRot : numpy array, 補間されたクォータニオン [x, y, z, w]
        """
        # Slerpの準備
        times = [0, 1]
        rotations = R.from_quat([beforeRot, currentRot])
        slerp = Slerp(times, rotations)

        # 重みに基づいて補間
        weightedRot = slerp([weight]).as_quat()[0]
        
        return weightedRot

    def motive2xarm(self, data, order, n_keys, negate_idx):
        data_copy = data.copy()
        keys_list = list(data_copy.keys())
        for i in range(n_keys):
            key = keys_list[i]
            data_copy[key] = [data_copy[key][j] for j in order]
            
            # Apply negation based on i (even/odd)
            data_copy[key][negate_idx[i % 2]] = -1 * data_copy[key][negate_idx[i % 2]]
        return data_copy

    def calculate_difference(self, participant_positions):
        learner_left = np.array(participant_positions["participant1"])
        learner_right = np.array(participant_positions["participant2"])
        expert_left = np.array(participant_positions["participant3"])
        expert_right = np.array(participant_positions["participant4"])

        left_diff = np.linalg.norm(learner_left - expert_left)
        right_diff = np.linalg.norm(learner_right - expert_right)

        capped_left_diff = min(left_diff, self.differenceLimit)
        capped_right_diff = min(right_diff, self.differenceLimit)
        capped_average_diff = (capped_left_diff + capped_right_diff) / 2

        return capped_average_diff, capped_left_diff, capped_right_diff