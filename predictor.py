# -*- coding: utf-8 -*

from flyai.dataset import Dataset
from model import Model

print('调用了predict')

data = Dataset()
model = Model(data)

# img_path = 'CNY/0HI6RGPO.jpg'
p = model.predict_all(data.get_all_data()[0])
# print(p)
