#@Author : 王文龙
from sqlalchemy import create_engine, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, UniqueConstraint, Index,Float
from sqlalchemy.orm import relationship
from app import db


# 辅助表 记录收藏关系
Collect = db.Table('collect',  # 表名
              db.Column('user_id', db.String(45),db.ForeignKey('user_info.id'),primary_key=True),
              db.Column('stock_code', db.String(45),db.ForeignKey('stock_detail_info.code'),primary_key=True)
              )


'''
class Collect(db.Model):
    __tablename__ = 'collect'

    user_id = db.Column(db.String(45), primary_key=True)
    stock_code = db.Column(db.String(45), primary_key=True)
'''

# 创建一张数据表//user信息表
class User(db.Model):
    __tablename__ = 'user_info'

    id = db.Column(db.String(45), primary_key=True)
    password = db.Column(db.String(45))
    email = db.Column(db.String(45))
    stock = db.relationship("Stock",secondary="collect",backref=db.backref("user_infos"))
    #stock = db.Column(db.String(45),db.ForeignKey('stock_detail_info.code'))


# 创建另一张数据表 stock 信息表

class Stock(db.Model):
    __tablename__ = "stock_detail_info"

    code = db.Column(db.String(45),primary_key=True)
    name = db.Column(db.Text)
    changepercent = db.Column(db.Float)
    trade = db.Column(db.Float)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    settlement = db.Column(db.Float)


# 创建所有表
db.create_all()


