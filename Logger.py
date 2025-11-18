import sys

class PrintLog:
    def __init__(self, filepath, mode="w"):
        self.file = open(filepath, mode)
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self.file.write(data)

    def flush(self):
        self._stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self._stdout
        self.file.close()
