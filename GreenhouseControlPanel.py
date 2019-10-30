# coding: utf-8
"""
Saturnalia @ WUR-Tecent Hackathon of AutoGreenhouse
Fangrui Liu 
Aug 2019
"""

import json
import time
import requests
import datetime
from greenhousecontrolpanel import paramParser


class Parameters:
    """
    Input Parameters to the simulation environment
    """


class MyENv(object):
    def __init__(self):
        self.para = None
        self.key = '0b466aad-aaec-41e3-85c7-a81e3c854db4'
        self.url = 'https://www.digigreenhouse.wur.nl/AGC/'
        self.headers = {'Timeout': '300', 'ContentType': 'json', 'RequestMethod': 'post'}
        self.data = None
        self.startdate = datetime.date(2017, 12, 15)
        self.enddate = datetime.date(2017, 12, 15)
        self.deltatime = 0
        # self.parser = paramParser.ParamParser()

    def reset(self):
        self.deltatime += 1
        self.enddate += datetime.timedelta(days=self.deltatime)
        with open('parameters.json', 'r') as p:
            self.para = json.load(p)
        self.para["simset"]["@endDate"] = self.enddate.strftime("%d-%m-%Y")
        print(self.para["simset"]["@endDate"])
        self.data = {'key': self.key, 'parameters': json.dumps(self.para), 'intermediate': []}
        r = requests.get(self.url, data=self.data, headers=self.headers)
        data = json.loads(r.text)
        # print(data)
        s = [None for i in range(32)]
        for i, _key in enumerate(data["data"]):
            s[i] = data["data"][_key]["data"][0]
        s.remove(s[0])
        print(s)

        return s

    def step(self, action=None):
        self.deltatime += 1
        self.enddate += datetime.timedelta(days=self.deltatime)
        # 修改parameters_.json
        with open('parameters.json', 'r') as p:
            self.para = json.load(p)
        self.para["simset"]["@endDate"] = self.enddate.strftime("%d-%m-%Y")
        self.para["comp1"]["setpoints"]["temp"]["@heatingTemp"]["01-06"] = action
        self.data = {'key': self.key, 'parameters': json.dumps(self.para), 'intermediate': []}
        r = requests.get(self.url, data=self.data, headers=self.headers)
        data = json.loads(r.text)
        s_ = [None for i in range(32)]
        for i, _key in enumerate(data["data"]):
            s_[i] = data["data"][_key]["data"][0]
        s_.remove(s_[0])
        reward = data["stats"]["economics"]["balance"]

        if reward <= -100 or reward >= 100:
            done = True
        else:
            done = False
        return s_, reward, done


class GreenhouseControlPanel:
    """
    Greenhouse Control Panel will provide you a accessible API to control all the sensors and devices
    It accepts python types and output a json-styled string to each control params.
    """

    def __init__(self):
        self.content = {}
        self.keys = {
            'DateTime': 'TIME',
            #   Data from Measuring Box
            'comp1_Air_T': 'FLOAT', 'comp1_Air_RH': 'FLOAT', 'comp1_Air_ppm': 'FLOAT',
            #   Data from Weather Station
            'common_Iglob_Value': 'FLOAT', 'common_TOut_Value': 'FLOAT', 'common_RHout_Value': 'FLOAT',
            'common_Windsp_Value': 'FLOAT', 'common_Tsky_Value': 'FLOAT',
            #   Data from PAR Sensor (Illumination)
            'comp1_PARsensor_Above': 'FLOAT',
            #   Data for heating power control
            'comp1_ConPipes_TSupPipe1': 'FLOAT', 'comp1_PConPipe1_Value': 'FLOAT',
            #   Data from opening vent
            'comp1_ConWin_WinLee': 'FLOAT', 'comp1_ConWin_WinWind': 'FLOAT',
            #   Setpoints
            'comp1_Setpoints_SpHeat': 'FLOAT', 'comp1_Setpoints_SpVent': 'FLOAT',
            #   Position of Screens
            'comp1_Scr1_Pos': 'FLOAT', 'comp1_Scr2': 'FLOAT',
            #
            'comp1_ConLmp1_PowerFac': 'FLOAT', 'comp1_McConAir_Value': 'FLOAT',
            #   Irrigations
            'comp1_ConWater_slabEC': 'FLOAT', 'comp1_ConWater_Irrigation': 'FLOAT', 'comp1_ConWater_Drain': 'FLOAT',
            'comp1_ConWater_slabSaturation': 'FLOAT', 'comp1_ConWater_lightSum': 'FLOAT',
            #   Crop growth monitoring
            'comp1_Crop_LAI': 'FLOAT', 'comp1_Plant_Stemden': 'FLOAT', 'comp1_Plant_PlantLoad': 'FLOAT',
            #   harvest
            'comp1_Harvest_CumFruitsDW': 'FLOAT', 'comp1_Harvest_CumFruitsFW': 'FLOAT'
        }
        #   Establish keys and structures
        for key in self.keys.keys():
            if self.keys[key] == 'TIME':
                """
                time.gmtime([secs])
                Convert a time expressed in seconds since the epoch to a struct_time in UTC in which 
                the dst flag is always zero. If secs is not provided or None, the current time as
                returned by time() is used. Fractions of a second are ignored. See above for a description 
                of the struct_time object. See calendar.timegm() for the inverse of this function.
                """
                self.content[key] = time.time()
            elif self.keys[key] == 'FLOAT':
                self.content[key] = 0.0
            elif self.keys[key] == 'INT':
                self.content[key] = 0
            else:
                raise TypeError("Type `" + key + "` is defined in the package")

    def err_type_mismatch(self, key, type, value):
        raise TypeError("Key `" + key + "` does not match the desired type `" + type + "` for value `" + value + "`")

    def check(self):
        """
        Check if the parameter is valid for its type
        """
        for key in self.keys.keys():
            if self.keys[key] == 'TIME':
                if self.content[key] is not time.struct_time:
                    self.err_type_mismatch(key, self.keys[key], self.content[key])
            elif self.keys[key] == 'FLOAT':
                if self.content[key] is not float:
                    self.err_type_mismatch(key, self.keys[key], self.content[key])
            elif self.keys[key] == 'INT':
                if self.content[key] is not int:
                    self.err_type_mismatch(key, self.keys[key], self.content[key])
            else:
                raise TypeError("Type `" + key + "` is defined in the package")
        print("Keys check passed!")


if __name__ == '__main__':
    myenv = MyENv()
    myenv.reset()
    # with open('parameters.json', 'r') as p:
    #     para = json.load(p)
    # key = '0b466aad-aaec-41e3-85c7-a81e3c854db4'
    # url = 'https://www.digigreenhouse.wur.nl/AGC/'
    # headers = {'Timeout': '300', 'ContentType': 'json', 'RequestMethod': 'post'}
    # data = {'key': key, 'parameters': json.dumps(para), 'intermediate': []}
    # r = requests.get(url, data=data, headers=headers)
    # with open("Output_01-06.json", 'w') as out:
    #     out.write(r.text)
