from threading import Thread

def runA():
    while True:
        print ('A\n')

def runB():
    while True:
        print ('B\n')

if __name__ == "__main__":
    t1 = Thread(target = runA)
    t2 = Thread(target = runB)
    t1.setDaemon(True)
    t2.setDaemon(True)
    t1.start()
    t2.start()
    while True:
        pass