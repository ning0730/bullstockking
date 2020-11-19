from flask import Flask,render_template,request,url_for,redirect,session
from flask_bootstrap import Bootstrap
import jieba
import codecs
import urllib
import collections
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from threading import Thread
from apscheduler.schedulers.blocking import BlockingScheduler
import tushare as ts
import http.client
import hashlib
import urllib
import random
import pymysql
from datetime import datetime, timedelta
import requests  # 导入需要的包
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# coding=utf-8
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


app = Flask(__name__)
#配置数据库地址
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://stardust@bullstockking:ZYY99c0e222@bullstockking.mysql.database.chinacloudapi.cn:3306/bullstockking?charset=utf8'
#app.config['SQLALCHEMY_DATABASE_URI']='mysql+pymysql://root:200507@localhost:3306/stock?charset=utf8'
# 该配置为True,则每次请求结束都会自动commit数据库的变动
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SQLALCHEMY_COMMIT_TEARDOWN'] = True
app.config['SECRET_KEY'] = '21312412412dfafaivhoseiriuiqe2175887w'

app.config['MAIL_SERVER'] = 'smtp.qq.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '527546458@qq.com'
app.config['MAIL_PASSWORD'] = 'ddugmxhmxexdbijh'
mail = Mail(app)
db = SQLAlchemy(app)

sid = SentimentIntensityAnalyzer()

#将中文文本翻译成英文文本，使用百度API，便于后续使用NLTK进行情感分析
def translate(q):
    appid = '20190709000316123'  # 你的appid
    secretKey = 'MbkpV5nmW9GCxEfYRw2H'  # 你的密钥

    httpClient = None
    myurl = '/api/trans/vip/translate'
    fromLang = 'zh'
    toLang = 'en'
    salt = random.randint(32768, 65536)

    sign = appid + q + str(salt) + secretKey

    sign = hashlib.md5(sign.encode(encoding='UTF-8')).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result = json.loads(response.read().decode())
        return (result['trans_result'][0]['dst'])
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()

#给定一段中文文本，返回文本的情感
def getSentiment(chText):
    chText = chText.replace("\n", "")
    chText = chText[0:900]
    #调用函数，翻译
    enText = translate(chText)
    if(enText!=None):
        #使用NLTK对翻译过后的英文文本进行情感分析
        ss = sid.polarity_scores(enText)
        #返回有四个值：neg，pos，new，compound，表示情感
        return ss
    else:
        return None
#新闻的类
class newsInfo:
    title=""
    description = ""
    article=""
    url=""

    def __init__(self,title,url,description,article):
        self.title = title
        self.url = url
        self.description = description
        self.article = article
#股票具体信息的类
class stockDetails:
    stockCode=""
    stockName =""
    stockNews = []
    stockHisData = []

    def __init__(self,stockCode):
        self.stockCode = stockCode

#股票的类
class newsInfo:
    title = ""
    description = ""
    article = ""
    url = ""

    def __init__(self, title, url, description, article):
        self.title = title
        self.url = url
        self.description = description
        self.article = article


class stockDetails:
    stockCode = ""
    stockName = ""
    stockNews = []
    stockHisData = []

    def __init__(self, stockCode):
        self.stockCode = stockCode

#读取原始数据，并生成训练样本
def getData(df, column, train_end=-300, days_before=30, return_all=True, generate_index=False):
    '''
    读取原始数据，并生成训练样本
    df             : 原始数据
    column         : 要处理的列
    train_end      : 训练集的终点
    days_before    : 多少天来预测下一天
    return_all     : 是否返回所有数据，默认 True
    generate_index : 是否生成 index
    '''
    series = df[column].copy()

    # 划分数据
    # 0 ~ train_end 的为训练数据，但实际上，最后的 n 天只是作为 label
    # 而 train 中的 label，可用于 test
    train_series, test_series = series[:train_end], series[train_end - days_before:]

    # 创建训练集
    train_data = pd.DataFrame()

    # 通过移位，创建历史 days_before 天的数据
    for i in range(days_before):
        # 当前数据的 7 天前的数据，应该取 开始到 7 天前的数据； 昨天的数据，应该为开始到昨天的数据，如：
        # [..., 1,2,3,4,5,6,7] 昨天的为 [..., 1,2,3,4,5,6]
        # 比如从 [2:-7+2]，其长度为 len - 7
        train_data['c%d' % i] = train_series.tolist()[i: -days_before + i]

    # 获取对应的 label
    train_data['y'] = train_series.tolist()[days_before:]

    # 是否生成 index
    if generate_index:
        train_data.index = train_series.index[days_before:]

    if return_all:
        return train_data, series, df.index.tolist()

    return train_data

#创建 LSTM 层
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,  # 输入尺寸为 1，表示一天的数据
            hidden_size=64,
            num_layers=1,
            batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(64, 1))

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全 0 的 state
        out = self.out(r_out[:, -1, :])  # 取最后一天作为输出

        return out

class TrainSet(Dataset):
    def __init__(self, data):
        # 定义好 image 的路径
        # data 取前多少天的数据， label 取最后一天的数据
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
# 超参数
LR = 0.0001
EPOCH = 100
TRAIN_END=-300
DAYS_BEFORE=7

#使用 LSTM 得到预测股价，返回预测图
def predict(stockCode):
    df = ts.get_hist_data(stockCode, start='2016-01-05', end='2019-07-01')
    # 注意历史数据靠前
    df = df.sort_index(ascending=True)
    df.to_csv('sh.csv')
    df = pd.read_csv('sh.csv', index_col=0)
    df.index = list(map(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'), df.index))
    df.head()

    # 数据集建立
    train_data, all_series, df_index = getData(df, 'high', days_before=DAYS_BEFORE, train_end=TRAIN_END)

    # 获取所有原始数据
    all_series = np.array(all_series.tolist())
    # 绘制原始数据的图
    plt.figure(figsize=(12, 8))
    plt.plot(df_index, all_series, label='real-data')

    # 归一化，便与训练
    train_data_numpy = np.array(train_data)
    train_mean = np.mean(train_data_numpy)
    train_std = np.std(train_data_numpy)
    train_data_numpy = (train_data_numpy - train_mean) / train_std
    train_data_tensor = torch.Tensor(train_data_numpy)

    # 创建 dataloader
    train_set = TrainSet(train_data_tensor)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

    # rnn = LSTM()
    #
    # if torch.cuda.is_available():
    #     rnn = rnn.cuda()
    #
    # optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    # loss_func = nn.MSELoss()
    #
    # for step in range(EPOCH):
    #     for tx, ty in train_loader:
    #
    #         if torch.cuda.is_available():
    #             tx = tx.cuda()
    #             ty = ty.cuda()
    #
    #         output = rnn(torch.unsqueeze(tx, dim=2))
    #         loss = loss_func(torch.squeeze(output), ty)
    #         optimizer.zero_grad()  # clear gradients for this training step
    #         loss.backward()  # back propagation, compute gradients
    #         optimizer.step()
    #     print(step, loss.cpu())
    #     if step % 10:
    #         torch.save(rnn, 'rnn.pkl')
    # torch.save(rnn, 'rnn.pkl')
    rnn = torch.load('rnn.pkl')

    generate_data_train = []
    generate_data_test = []

    # 测试数据开始的索引
    test_start = len(all_series) + TRAIN_END

    # 对所有的数据进行相同的归一化
    all_series = (all_series - train_mean) / train_std
    all_series = torch.Tensor(all_series)

    for i in range(DAYS_BEFORE, len(all_series)):
        x = all_series[i - DAYS_BEFORE:i]
        # 将 x 填充到 (bs, ts, is) 中的 timesteps
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=2)

        if torch.cuda.is_available():
            x = x.cuda()

        y = rnn(x)

        if i < test_start:
            generate_data_train.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)
        else:
            generate_data_test.append(torch.squeeze(y.cpu()).detach().numpy() * train_std + train_mean)

    plt.figure(figsize=(12, 8))
    plt.plot(df_index[DAYS_BEFORE: TRAIN_END], generate_data_train, 'b', label='train-data', )
    plt.plot(df_index[TRAIN_END:], generate_data_test, 'k', label='predict-data')
    plt.plot(df_index, all_series.clone().numpy() * train_std + train_mean, 'r', label='real-data')
    plt.legend()
    #保存图片
    plt.savefig('/Users/yidan/PycharmProjects/BullStockKing4.2/static/predict'+stockCode+'.png')

#获取输入框提示的函数
@app.route('/home_page/selectFill')
@app.route('/details/selectFill')
@app.route('/my_collection/selectFill')
@app.route('/selectFill',methods=['GET'])
def selectFill():
    #链接数据库
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)
    cur = conn.cursor()
    #获取所有股票
    cur.execute("select * from stocks;")
    data = cur.fetchall()
    conn.commit()
    conn.close()
    #将数据传给前端
    result = json.dumps((data, request.cookies.get('user')))
    return result


@app.route('/details/?<string:stockCode>')
def details(stockCode):
    if request.cookies.get('user') == None:
        resp = redirect(url_for('login'))
        resp.set_cookie('status','302')
        return resp
    else:
        return render_template('stock_details.html')

#点击搜索按钮到这个路由
@app.route('/getHis',methods=['GET','POST'])
def getHis():
    stockName = request.form.get("code")
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)

    cur = conn.cursor()
    #获取相对应的股票名称
    cur.execute("select stock_code from stocks where cns_name= \'" + stockName + "\';")
    stockCode = cur.fetchone()[0]
    conn.commit()
    conn.close()
    #跳转页面
    return redirect(url_for('details',stockCode=stockCode))

#详情页面会通过ajax加载的路由
@app.route('/details/hisData',methods=["POST"])
def hisData():
    #获取股票代码
    stockCode = request.form.get("code")
    details = stockDetails(stockCode)
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)

    cur = conn.cursor()
    #向数据库获取股票的所有历史数据
    cur.execute("select date,open,close,low,high from historydata where stock_code = \'"+ str(stockCode)+"\';")
    details.stockHisData = cur.fetchall()
    #获取股票的名称
    cur.execute("select cns_name from stocks where stock_code = \'"+ str(stockCode)+"\';")
    details.stockName = cur.fetchone()[0]
    #获取股票的新闻
    details.stockNews = getStockNews(details.stockName)
    #获取股票的词频
    result0 = cipinAndSentiment(details.stockNews)
    cipin_list = result0[0]
    #获取股票的情感
    sentiment = result0[1]
    #获取用户是否关注股票
    follow = judgecollect(request.cookies.get('user'),details.stockCode)
    #将所有数据打包，发给前端
    result = json.dumps((details.stockName,details.stockNews,details.stockHisData,cipin_list,follow,sentiment))
    #生成股价预测图像
    predict(str(stockCode))
    conn.commit()
    conn.close()
    return result

#判断用户是否关注这一支股票
def judgecollect(user_id,code):
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)
    cur = conn.cursor()
    cur.execute("select * from collect where user_id= %s and stock_code= %s",[user_id,code])
    result=cur.fetchall()
    if (result.__len__()!=0):
        return True
    else:
        return False
#获取股票的新闻
def getStockNews(stockName):
    stockNews = []
    result = []
    headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36' }

    url = "https://www.baidu.com/s?ie=utf-8&cl=2&medium=2&rtt=4&bsst=1&rsv_dl=news_t_sk&tn=news&word=股+"+stockName
    #解析html，获取链接
    r = requests.get(url,headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    #解析html
    div = soup.select('#content_left .result .c-title a')

    divSummary = soup.select('#content_left .result div')
    for each,each1 in zip(div,divSummary):
        #新闻文本
        article=""
        #获取新闻的href
        linkurl=each.get('href')
        description = each1.text
        title = each.text
        r = requests.get(linkurl, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        div=soup.select('.article-content p span')
        #找到股票页面的新闻文本
        for eachDiv in div:
            article= article + eachDiv.text +'\n\n'
        stockNews.append((title,linkurl,description,article))
    return stockNews

#获取新闻的词频和情感
def cipinAndSentiment(stockNews):
    articles = ""
    for each in stockNews:
        articles += each[3]
    # 分词
    articles=articles.replace("\n","")
    articles=articles.replace(" ","")
    seg_list = jieba.cut(articles)
    seg_result = []
    for w in seg_list:
        seg_result.append(w)
    # 读取停用词
    stopwords = set()  # 集合
    fr = codecs.open('./static/stop.txt', 'r', 'utf-8')
    for word in fr:
        stopwords.add(word.strip())
    fr.close()

    result = list(filter(lambda x: x not in stopwords, seg_result))

    mycount = collections.Counter(result)
    cipin_list = []
    for key, val in mycount.most_common(1000):
        cipin = (key, str(val))
        cipin_list.append(cipin)
    #获取新闻的情感
    ss = getSentiment(articles)
    result = [cipin_list, ss]
    return result
# 发送邮件
'''
def send_async_email(app,msg):
    with app.app_context():
        mail.send(msg)


def SendMail():
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)
    cur = conn.cursor()
    cur.execute("select * from collect ")
    result = cur.fetchall()
    msg = Message('股价下跌预警', sender='527546458@qq.com', recipients=["916333607@qq.com"])
    msg.body = 'From QQ'
    msg.html = '<b>尊敬的用户:</b><b>你所收藏的股票出现了大幅下跌情况，请您尽快登录网站查看</b>'
    thr = Thread(target=send_async_email,args=[app,msg])
    thr.start()
    return 'ok'
'''

def SendMail():

    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)
    cur = conn.cursor()
    cur.execute("select email,stock_detail_info.code,stock_detail_info.name from collect,user_info,stock_detail_info where user_info.id = collect.user_id and stock_detail_info.code = collect.stock_code and stock_detail_info.changepercent<0 ")
    results = cur.fetchall()
    conn.close()
    for result in results:
        if result[0] is not None:
            msg = Message('股价下跌预警', sender='527546458@qq.com', recipients=[result[0]])
            msg.body = 'From QQ'
            msg.html = '<b>尊敬的用户:</b><b>你所收藏的股票代码为%s的%s股票价格跌幅已经超过2%%，请您尽快登录网站查看，并做好应对措施，及时止损\n牛股王团队竭诚为您服务</b>'%(result[1],result[2])
            thr = Thread(target=send_async_email,args=[app,msg])
            thr.start()


# 异步函数
def send_async_email(app,msg):
    with app.app_context():
        mail.send(msg)

# 每隔一个小时提醒一次
def alert():
    scheduler = BlockingScheduler()
    scheduler.add_job(func=SendMail, trigger='cron', second='*/3')
    scheduler.start()


@app.route('/my_collection/',methods=['POST', 'GET'])
def my_collection():
    if request.cookies.get('user') == None:
        resp = redirect(url_for('login'))
        resp.set_cookie('status','302')
        return resp
    else:
        from models import User
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 1, type=int)
        #user = User.query.filter(User.id == '1').first()
        pagination = User.query.filter(User.id == request.cookies.get('user')).paginate(page, per_page)
        return render_template("my_collection.html", pagination =pagination,per_page=per_page)


@app.route('/uncollect/<code>',methods=['POST', 'GET'])
def uncollect(code):
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)
    cur = conn.cursor()
    cur.execute("delete from collect where user_id=%s and stock_code= %s;",[request.cookies.get('user'),code])
    conn.commit()
    conn.close()
    return redirect(url_for('my_collection'))


@app.route('/details/follow',methods=['POST'])
def follow():
    stockCode = request.form.get("stockCode")
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)
    cur = conn.cursor()
    cur.execute("insert into collect(user_id, stock_code) values (%s,%s);",[request.cookies.get('user'),stockCode])
    conn.commit()
    conn.close()

    result = json.dumps(True)
    return result

@app.route('/details/unfollow',methods=['POST'])
def unfollow():
    stockCode = request.form.get("stockCode")
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)
    cur = conn.cursor()
    cur.execute("delete from collect where user_id = %s and stock_code = %s;" ,[request.cookies.get('user'),stockCode])
    conn.commit()
    conn.close()

    result = json.dumps(True)
    return result

#用户使用get方式访问登陆页面绑定的函数
@app.route('/login/',methods=['GET'])
def login():
    return render_template('login.html')

#用户使用post方式访问登陆页面绑定的函数，并进行用户名密码判断，若用户名或密码不正确则会弹窗提示
@app.route('/login/',methods=['POST'])
def servlet():
    id = request.form.get("username")
    password = request.form.get("password")
    resp1 = redirect(url_for('homePage'))
    resp2 = redirect(url_for('login'))
    resp1.set_cookie('user',id)
    resp2.set_cookie('user',id)
    resp1.delete_cookie('status')
    resp2.delete_cookie('status')
    resp2.set_cookie('status', '404')
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)
    cur = conn.cursor()
    cur.execute("select id,password from user_info where id = %s and password = %s", [id, password])
    userinfo = cur.fetchall()
    conn.commit()
    conn.close()
    if len(userinfo) > 0:
        return resp1
    else:
        return resp2

#主页面跳转函数，加载显示网站主页
@app.route('/home_page/')
def homePage():
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)
    cur = conn.cursor()
    cur.execute("select name,code from stock_detail_info where changepercent > 0")
    namelist = cur.fetchall()
    list = namelist[0:30]
    conn.commit()
    conn.close()
    resp = urllib.request.urlopen('https://www.quantinfo.com/API/Argus/predict')
    showlist = json.loads(resp.read())
    list1 = showlist[0:18]
    list2 = showlist[18:36]
    return render_template("home_page.html", list=list, namelist=namelist, list1 = list1,list2 = list2)

#根路由跳转函数，作用等同于主页面
@app.route('/')
def hello_world():
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222", host="bullstockking.mysql.database.chinacloudapi.cn",port=3306)
    cur = conn.cursor()
    cur.execute("select name,code from stock_detail_info where changepercent > 0")
    namelist = cur.fetchall()
    list = namelist[0:30]
    conn.commit()
    conn.close()
    resp = urllib.request.urlopen('https://www.quantinfo.com/API/Argus/predict')
    showlist = json.loads(resp.read())
    list1 = showlist[0:18]
    list2 = showlist[18:36]
    return render_template("home_page.html",list = list,namelist = namelist,list1 = list1,list2 = list2)

#用户使用get方式访问注册页面绑定的函数
@app.route('/register/',methods=['GET'])
def register():
    return render_template('register.html')

#用户使用post方式访问注册页面绑定的函数
@app.route('/register/',methods=['POST'])
def success():
    username = request.form.get("username")
    password = request.form.get("password")
    email = request.form.get("email")
    name = request.form.get("name")
    phone = request.form.get("phone")
    country = request.form.get("country")
    resp = redirect(url_for('homePage'))
    resp.set_cookie('user', username)
    conn = pymysql.connect(db="bullstockking", user="stardust@bullstockking", password="ZYY99c0e222",
                           host="bullstockking.mysql.database.chinacloudapi.cn", port=3306)
    cur = conn.cursor()
    cur.execute("select id from user_info where id = %s", [username])
    userlist = cur.fetchall()
    if len(userlist) > 0:
        resp = redirect(url_for('register'))
        resp.set_cookie('state', '404')
        return resp
    else:
        resp.delete_cookie('state')
        cur.execute("insert into user_info values (%s, %s, %s, %s, %s ,%s)",
                    [username, password, email, name, country, phone])
        conn.commit()
        conn.close()
        return resp

#用户点击导航栏中的退出按钮执行的注销操作
@app.route('/logout/',methods=['GET','POST'])
def logout():
    resp = redirect(url_for('homePage'))
    resp.delete_cookie('user')
    return resp

#用户点击忘记密码执行相关操作的函数
@app.route('/forget/',methods=['GET'])
def forget():
    return render_template('forget.html')

#用户提交用户名或邮箱后跳转到主页的函数
@app.route('/forget/',methods=['POST'])
def okay():
    return redirect(url_for('homePage'))

if __name__ == '__main__':
    app.debug = True
    bootstrap = Bootstrap(app)
    db.metadata.clear()
    thr = Thread(target=alert)
    thr.start()
    app.run()
