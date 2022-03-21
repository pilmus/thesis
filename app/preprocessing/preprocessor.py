PREPROCESSOR = None


def get_preprocessor():
    global PREPROCESSOR
    if not PREPROCESSOR:
        PREPROCESSOR = Preprocessor()
    return PREPROCESSOR

class Preprocessor():
    def __init__(self):
        print("ape")
