#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 22:20
# @Author  : dabaoi
# @Site    : 
# @File    : pltloss.py
# @Software: PyCharm
if (i % 50 == 0):
	print ("Epoch:%d,Step:%d,Loss:%.3f" % (epoch,i,loss.item ( )))
	torch.save (net.state_dict ( ),'\parameter.pkl')
	losslist.append (loss.item ( ))
losslist.append (loss.item ( ))

plt.plot (losslist [ 1: ])
plt.xlabel ('epoch')
plt.ylabel ('loss')
plt.title ('')
plt.show ( )