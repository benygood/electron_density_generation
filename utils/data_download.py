import os.path
import paramiko  # 用于调用scp命令
from scp import SCPClient
import pickle
import fire

class ScpClient():
    def __init__(self, host, user, passwd):
        self.__host = host  # 服务器ip地址
        self.__port = 22  # 端口号
        self.__username = user # ssh 用户名
        self.__password = passwd  # 密码

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self.ssh_client.connect(self.__host, self.__port, self.__username, self.__password)
        self.scpclient = SCPClient(self.ssh_client.get_transport(), socket_timeout=60.0)

    def __del__(self):
        self.ssh_client.close()
        self.scpclient.close()

    def set_scp_server_information(self,host_ip="192.168.0.232",port = 22,username = "root",password = "root"):
        self.__host = host_ip
        self.__port = port
        self.__username = username
        self.__password = password

    def get(self, remote_path="/A8/",  local_path="D:\python_eg"):
        file_path_lo = local_path
        file_path_re = remote_path
        try:
            self.scpclient.get(file_path_re, file_path_lo)  # 从服务器中获取文件
        except FileNotFoundError as e:
            print(e)
            print("system could not find the specified file" + local_path)

    def put(self, remote_path="/A8/",  local_path="D:\python_eg"):
        file_path_lo = local_path
        file_path_re = remote_path
        try:
           self.scpclient.put(file_path_lo, file_path_re)  # 上传到服务器指定文件
        except FileNotFoundError as e:
            print(e)
            print("system could not find the specified file" + local_path)


def scp(host, user, passwd, source_dir, filenames_config, to_dir):
    scp_cc = ScpClient(host, user, passwd)
    with open(filenames_config, 'rb') as f:
        fn_list = pickle.load(f)
    for i, fn in enumerate(fn_list):
        scp_cc.get(os.path.join(source_dir, "{}.pdb".format(fn[:-4])), to_dir)
        if i % 5000 == 0:
            print("scp {} files".format(i + 1))
    print("scp {} files".format(i + 1))


if __name__ == "__main__":
    fire.Fire(scp)
