# -*- coding: utf-8 -*
from flyai.source.source import Source
from flyai.utils.yaml_helper import Yaml
from flyai.dataset import Dataset
from model import Model

print('调用了predict')




data = Dataset()
model = Model(data)

p = model.predict_all(data.get_all_data()[0])
print(p)
