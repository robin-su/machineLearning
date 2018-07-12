
import random
'''
   生成日志代码
'''
albet_Num = ["a","b","c","d","e","f","g","h","1","2","3","4","5","6","7","A","B","O","P","Q"]
user_list = ["one","two","three","four","five"]
num = ["1","2","3","4","5","6","7","8","9","0"]
log_type_array = ["1","2","3","4","5","6","7"]
topic_array = ["空气净化器","净水器","加湿器","空气净化滤芯"]

file_object = open('./logfile.txt','w')

for n in range(0,2000):
    cookie= "".join(random.sample(albet_Num,6))
    uid = "".join(random.sample(user_list,1))
    user_agent = "Macintosh Chrome Safari"
    ip = "192.168.89.177"
    video_id = "".join(random.sample(num,7))
    topic = "".join(random.sample(topic_array,1))
    order_id = "0"
    log_type = "".join(random.sample(log_type_array,1))
    final = cookie + "&" + uid + "&" + user_agent + "&" + ip + "&" + video_id + "&" + topic + "&" + order_id + "&" + log_type + "\r\n"
    file_object.write(final)


file_object.close()
