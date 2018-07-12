# -*- coding:utf-8 -*-

import socket
import sys

serversocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

host = '127.0.0.1'
port = 8888

serversocket.bind((host,port))

serversocket.listen(1)

def gp(uid):
    click_action = {}
    file = open('./logfile.txt')
    for line in file.readlines():
        line = line.strip()
        ls = line.split("&")
        if ls[7] != "1":
            continue
        if ls[1] not in click_action.keys():
            click_action[ls[1]] = []
        click_action[ls[1]].append(ls[4])
    if uid in click_action.keys():
        return "&&".join(click_action[uid])


comment_log = {} # key uid value video_ids

def log_process(request):
    print "xiao ming's log"
    print request

    request = request.strip()
    ls = request.split("&")
    if ls[1] not in comment_log.keys():
        comment_log[ls[1]] = []

    print(comment_log)
    comment_log[ls[1]].append(ls[4])

    for k,v in comment_log.items():
        print k + "\t"  + "&&".join(v)
    return "yes I got it"

while True:
    client_connection,client_address = serversocket.accept()
    request = client_connection.recv(1024)
    #http_response = gp(request)
    http_response = log_process(request)
    client_connection.sendall(http_response)
    client_connection.close()