#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 14:15
# @Author  : dabaoi
# @Site    : 
# @File    : predict.py
# @Software: PyCharm
# 数据集预测
def computeTestSetAccuracy ( model,loss_function ):
	device = torch.device ("cuda:0" if torch.cuda.is_available ( ) else "cpu")
	test_acc = 0.0
	test_loss = 0.0
	with torch.no_grad ( ):
		model.eval ( )
		for j,(inputs,labels) in enumerate (test_data):
			inputs = inputs.to (device)
			labels = labels.to (device)
			outputs = model (inputs)
			loss = loss_function (outputs,labels)
			test_loss += loss.item ( ) * inputs.size (0)
			ret,predictions = torch.max (outputs.data,1)
			correct_counts = predictions.eq (labels.data.view_as (predictions))
			acc = torch.mean (correct_counts.type (torch.FloatTensor))
			test_acc += acc.item ( ) * inputs.size (0)
			print ("Test Batch Number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format (
				j,loss.item ( ),acc.item ( )
			))
	avg_test_loss = test_loss / test_data_size
	avg_test_acc = test_acc / test_data_size
	print ("Test accuracy : " + str (avg_test_acc))
# 单张数据预测
def predict ( model,test_image_name ):
	transform = image_transforms [ 'test' ]
	test_image = Image.open (test_image_name)
	draw = ImageDraw.Draw (test_image)
	test_image_tensor = transform (test_image)
	if torch.cuda.is_available ( ):
		test_image_tensor = test_image_tensor.view (1,3,224,224).cuda ( )
	else:
		test_image_tensor = test_image_tensor.view (1,3,224,224)
	with torch.no_grad ( ):
		model.eval ( )
		out = model (test_image_tensor)
		ps = torch.exp (out)
		topk,topclass = ps.topk (1,dim = 1)
		print ("Prediction : ",idx_to_class [ topclass.cpu ( ).numpy ( ) [ 0 ] [ 0 ] ],", Score: ",
			   topk.cpu ( ).numpy ( ) [ 0 ] [ 0 ])
		text = idx_to_class [ topclass.cpu ( ).numpy ( ) [ 0 ] [ 0 ] ] + " " + str (topk.cpu ( ).numpy ( ) [ 0 ] [ 0 ])
		font = ImageFont.truetype ('arial.ttf',36)
		draw.text ((0,0),text,(255,0,0),font = font)
		test_image.show ( )
# 模型调用
#选用前面训练的最好的模型
model = torch.load('models/animals-10_model_4.pt')
loss_func = nn.NLLLoss()
computeTestSetAccuracy(model, loss_func)
predict(model, 'animals-10/test/bear/009_0071.jpg')