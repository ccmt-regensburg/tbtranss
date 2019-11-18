class CoordinateError(Exception):
    '''Exception raised for coordinate errors. E.g. not inside boundary or vector not fitting between sites.'''

    def __init__(self, message, vector):
            self.msg = message
            self.vecmsg = "Your specified vector was " + str(vector) + "."
