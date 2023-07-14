import subprocess
import os
from ftplib import FTP
import zipfile
import time

FAIL_COUNT_MAX = 10

def run_docker_compose_up():
    try:
        subprocess.call('wsl --update', shell=True)
        check = input("open docker yet?")
        if check == "":
            subprocess.run(['docker', 'load', '-i', 'tia_image.tar'], check=True)
        else:
            raise("Fail")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    except FileNotFoundError:
        print("docker-compose command not found. Please make sure Docker Compose is installed.")
    check_device = "CPU"
    try:
        subprocess.check_call("nvidia-smi")
        print("GPU")
        check_device = "GPU"
        subprocess.run(['docker', 'compose', '-f', 'docker-compose-gpu.yml', 'up'], check=True)
    except:
        if check_device == "CPU":
            print("CPU")
            subprocess.run(['docker', 'compose', '-f', 'docker-compose-cpu.yml', 'up'], check=True)

def ftp_download():
    path = os.getcwd()
    try:
        with FTP("pca571.nseb.co.th", timeout=20) as ftp:
            ftp.encoding = "utf-8"
            ftp.login()
            ftp.cwd("AISetUp")
            for file_name in ftp.nlst():
                path_write_file = path + "/" + file_name
                if file_name not in os.listdir(path):
                    download_file_size = 0
                    for _ in range(FAIL_COUNT_MAX):
                        with open(path_write_file, "wb") as file:
                            ftp.retrbinary(f"RETR {file_name}", file.write)
                        download_file_size = os.stat(path_write_file).st_size
                        if download_file_size != ftp.size(f"{file_name}"):
                            os.remove(path_write_file)
                        else:
                            print(f"Download {file_name} Success", flush=True)
                            break
                elif os.stat(path_write_file).st_size != ftp.size(f"{file_name}"):
                    os.remove(path_write_file)
                    download_file_size = 0
                    for _ in range(FAIL_COUNT_MAX):
                        with open(path_write_file, "wb") as file:
                            ftp.retrbinary(f"RETR {file_name}", file.write)
                        download_file_size = os.stat(path_write_file).st_size
                        if download_file_size != ftp.size(f"{file_name}"):
                            os.remove(path_write_file)
                        else:
                            print(f"Download {file_name} Success", flush=True)
                            break
                    
    except Exception as e:
        print(e, flush=True)
        raise("error")

def extract_file_zip():
    path_to_zip_file = "tia_set_up.zip"
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())

def install_nvidia():
    try:
        subprocess.call('536.40-desktop-win10-win11-64bit-international-dch-whql.exe', shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    except FileNotFoundError:
        print("docker-compose command not found. Please make sure Docker Compose is installed.")

def install_docker():
    try:
        subprocess.call('"Docker Desktop Installer.exe"', shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    except FileNotFoundError:
        print("docker-compose command not found. Please make sure Docker Compose is installed.")


while True:
    if not os.path.isfile("stage.txt"):
        with open("stage.txt", "w") as f:
            f.write(str(0))
    else:
        with open("stage.txt", "r") as r:
            value = r.read()
            print(value)
        if os.stat("stage.txt").st_size == 0:
            with open("stage.txt", "w") as f:
                f.write(str(0))

        if "0" == value:
            print("Download file from ftp", flush=True)
            with open("stage.txt", "w") as f:
                try:
                    ftp_download()
                    f.write(str(1))
                except:
                    f.write(str(0))
        elif "1" == value:
            print("Extract file from ftp", flush=True)
            with open("stage.txt", "w") as f:
                try:
                    extract_file_zip()
                    f.write(str(2))
                except:
                    f.write(str(1))
        elif "2" == value:
            print("Install Nvidia Driver", flush=True)
            with open("stage.txt", "w") as f:
                try:
                    install_nvidia()
                    f.write(str(3))
                except:
                    f.write(str(2))
        elif "3" == value:
            print("Install Docker", flush=True)
            try:
                install_docker()
                with open("stage.txt", "w") as f:
                    f.write(str(4))
                for i in range(100000):
                    print("")
            except:
                with open("stage.txt", "w") as f:
                    f.write(str(3))
            os.system("shutdown /r /t 1")
            for i in range(100000):
                print("")
                    
        elif "4" == value:
            print("Docker compose", flush=True)
            try:
                run_docker_compose_up()
                with open("stage.txt", "w") as f:
                    f.write(str(5))
            except:
                with open("stage.txt", "w") as f:
                    f.write(str(5))
        elif "5" == value:
            print("Remove used file", flush=True)
            os.remove("stage.txt")
            os.remove("536.40-desktop-win10-win11-64bit-international-dch-whql.exe")
            os.remove("Docker Desktop Installer.exe")
            #os.remove("tia_set_up.zip")
            break

        time.sleep(0.5)
