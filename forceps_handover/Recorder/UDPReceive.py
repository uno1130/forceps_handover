from socket import *


## UDP受信クラス
class udprecv:
    def __init__(self):
        SrcIP = "133.68.108.109"  # 受信元IP
        SrcPort = 7000  # 受信元ポート番号
        self.SrcAddr = (SrcIP, SrcPort)  # アドレスをtupleに格納

        self.robot_head = [0, 0, 0, 0, 0, 0]
        self.BUFSIZE = 1024  # バッファサイズ指定
        self.udpServSock = socket(AF_INET, SOCK_DGRAM)  # ソケット作成
        self.udpServSock.bind(self.SrcAddr)  # 受信元アドレスでバインド

    def recv(self):
        try:
            while True:  # 常に受信待ち

                data, addr = self.udpServSock.recvfrom(self.BUFSIZE)
                # 受信
                data_decode = data.decode()
                self.robot_head = [float(x) for x in data_decode.split(",")]

        except KeyboardInterrupt:
            print("fin")
