# -*- coding: utf-8 -*-
"""
Created on Fri May 22 08:52:50 2015

@author: sblanco
         Modified by jcid to log messages to standard output
"""

import logging
import sys


class Log:

    __logger__ = None
    __error__ = False

    def __init__(self, path, crear=False):

        try:
            self.__logger__ = logging.getLogger(__name__)
            self.__logger__ .setLevel(logging.DEBUG)

            # create a file handler
            # mode w create new file:
            if crear is True:
                handler = logging.FileHandler(path, mode='w')
            else:
                handler = logging.FileHandler(path)
            handler.setLevel(logging.DEBUG)

            # create a logging format
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)

            # add the handlers to the logger
            self.__logger__.addHandler(handler)

            # Add
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.__logger__.addHandler(ch)

        except:
            self.__error__ = True

    def debug(self, msg):
        if (self.__error__):
            print('DEBUG: {}'.format(msg))
        else:
            self.__logger__.debug(msg)

    def info(self, msg):
        if (self.__error__):
            print('INFO: {}'.format(msg))
        else:
            self.__logger__.info(msg)

    def warn(self, msg):
        if (self.__error__):
            print('WARN: {}'.format(msg))
        else:
            self.__logger__.warn(msg)

    def error(self, msg):
        if (self.__error__):
            print('ERROR: {}'.format(msg))
        else:
            self.__logger__.error(msg)

    def critical(self, msg):
        if (self.__error__):
            print('CRITICAL: {}'.format(msg))
        else:
            self.__logger__.critical(msg)
