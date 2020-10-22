from sys import stdout


def refresh_print(data):
    stdout.write("\r\033[1;37m>>\x1b[K" + data.__str__())
    stdout.flush()
