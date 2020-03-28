# CNNnumpy
문의사항  : won19600@kau.kr
Questions : won19600@kau.kr

using only numpy CNN

It is Mnist & Cifar10 CNN model. (default = Cifar10) 

this version has 2 conv layer

First layer : 3*3, 30layers

Second layer : 3*3, 30layers
↓

Max pooling(2*2) & dropout(0.25)
↓

3rd layer : 3*3, 60layers
↓

Max pooling(2*2) & dropout(0.25)
↓

4th layer : 3*3, 120layers
↓

500 Perceptrons Fully connected layer

dropout(0.5)

500 Perceptrons Fully connected layer

dropout(0.5)

↓

output
