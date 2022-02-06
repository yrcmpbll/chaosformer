LOGFOLDER = './.logs/'


class Logger:

    def __init__(self, filename) -> None:
        self.filename = filename
        self.logs = list()
    
    def log(self, *s):
        s = ' '.join([str(x) for x in s])
        self.logs.append(s+'\n')
    
    def write(self):
        with open(LOGFOLDER+'array_size.log', mode='w') as f:
            f.writelines(self.logs)