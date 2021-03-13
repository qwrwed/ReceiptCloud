import cv2
import socket

def printv(v=True, *args, end="\n"):
    if v:
        print(*args, end=end)

def show_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    print(f"RUNNING ON {s.getsockname()[0]}:5000")
    s.close()

# https://stackoverflow.com/a/62639343
def list_ports(v=False):
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port, cv2.CAP_DSHOW)
        if not camera.isOpened():
            is_working = False
            printv(v, f"Port {dev_port} is not working.")
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                printv(v, f"Port {dev_port} is working and reads images ({h} x {w})")
                working_ports.append(dev_port)
            else:
                printv(v, f"Port {dev_port} for camera ( {h} x {w}) is present but does not reads.")
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports, working_ports