#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
import collections
import Properties as prop


def createConfig(basePath):
    configDict = collections.OrderedDict()
    sparkEnv = {}
    executor_files = []
    args_list = []

    configMap = prop.parse(basePath + "/conf/ruleEngine.properties")
    jar_name = str(configMap.get("jar.path"))
    main_class = str(configMap.get("main.class"))
    configDict["file"] = "file://" + basePath + "/" + jar_name
    configDict["className"] = main_class

    configDict["numExecutors"] = int(configMap.get("executor.num"))
    configDict["executorCores"] = int(configMap.get("executor.cores"))
    configDict["executorMemory"] = str(configMap.get("executor.memory"))
    configDict["driverCores"] = int(configMap.get("driver.cores"))
    configDict["driverMemory"] = str(configMap.get("driver.memory"))
    configDict["queue"] = str(configMap.get("yarn.queue"))

    sparkEnv["spark.executor.extraJavaOptions"] = "-Dlog4j.debug -Dlog4j.configuration=log4j-executor.properties"
    sparkEnv[
        "spark.driver.extraJavaOptions"] = "-Dlog4j.debug -Dlog4j.configuration=file://" + basePath + "/conf/log4j-driver.properties"
    sparkEnv["spark.dynamicAllocation.executorIdleTimeout"] = "3600"
    sparkEnv["spark.dynamicAllocation.enabled"] = "false"
    sparkEnv["spark.shuffle.service.enabled"] = "false"
    configDict["conf"] = sparkEnv

    executor_files.append("file://" + basePath + "/conf/log4j-executor.properties")
    configDict["files"] = executor_files

    configDict["jars"] = loadLibPath(basePath + "/lib")

    app_name = configMap.get("application.name")

    args_list = [app_name, basePath + "/conf/template.properties"]
    configDict["args"] = args_list

    return configDict


# 加载依赖，并组成list
def loadLibPath(libPath):
    libList = []
    for libJar in os.listdir(libPath):
        if os.path.splitext(libJar)[1] == ".jar":
            libList.append("file://" + libPath + libJar)
    return libList


if __name__ == '__main__':
    basePath = os.path.dirname(os.getcwd())
    config_dict = createConfig(basePath)
    fp = file(basePath + '/conf/start-engine.json', 'w')
    json.dump(config_dict, fp, indent=4)