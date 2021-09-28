# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:38:34 2015

@author: breakthoven
"""
import sys
if sys.version_info.major == 3:
    import configparser as cf
else:
    import ConfigParser as cf


class ConfigCfg:

    __config = dict()

    def __init__(self, ruta='config.cf'):
        try:
            c = cf.ConfigParser()
            c.read(ruta)
            self.__config = c._sections
        except Exception as e:
            print('Error loading config file... {}'.format(e))
            exit(1)

    def get(self, section, param):

        try:
            return (self.__config[section][param.lower()])
        except:
            print('{0}:{1} not found in config file'.format(section, param))
            return None
