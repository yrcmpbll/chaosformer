import inspect
import os

LOGFOLDER = './.logs/'


class Logger:

    def __init__(self, filename) -> None:
        self.filename = filename
        self.logs = list()
    
    def log(self, *s):
        s = ' '.join([str(x) for x in s])
        self.logs.append(s+'\n')
    
    def write(self):
        CHECK_FOLDER = os.path.isdir(LOGFOLDER)
        if not CHECK_FOLDER:
            os.makedirs(LOGFOLDER)

        with open(LOGFOLDER+self.filename, mode='w') as f:
            f.writelines(self.logs)


class TestLogger(Logger):

    def __init__(self, filename=None) -> None:
        if filename is not None:
            self.filename = filename
        else:
            caller_name = inspect.stack()[1][3]
            functionality_name = '_'.join(caller_name.split('_')[1:])
            self.filename = 'log_'+functionality_name+'.log'

            # print(inspect.stack()[0][3])
            # print(inspect.stack()[1][3])  # will give the caller of foos name, if something called foo

        super().__init__(filename=self.filename)