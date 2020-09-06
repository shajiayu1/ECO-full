import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D,Conv3D, BatchNorm, Linear

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                name_scope,
                num_channels,
                num_filters,
                filter_size,
                stride=1,
                padding=0,
                groups=1,
                act=None):
        super(ConvBNLayer,self).__init__(name_scope)

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm = BatchNorm(num_filters,act=act)
    def forward(self,inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class InceptionV2ModuleA(fluid.dygraph.Layer):
    def __init__(self, name_scope,num_channels,out_channels1,out_channels2reduce,out_channels2,out_channels3reduce,out_channels3,out_channels4):
        super(InceptionV2ModuleA, self).__init__(name_scope)

      #  self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)
        self.branch1 = ConvBNLayer(self.full_name(),num_channels=num_channels,num_filters=out_channels1,filter_size=1,act='relu')

        self.branch2 = fluid.dygraph.Sequential(
           # ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
           # ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, padding=1),
            ConvBNLayer(self.full_name(),num_channels=num_channels,num_filters=out_channels2reduce,filter_size=1,act='relu'),
            ConvBNLayer(self.full_name(),num_channels=out_channels2reduce,num_filters=out_channels2,filter_size=3,padding=1,act='relu'),
        )

        self.branch3 = fluid.dygraph.Sequential(
            #ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            #ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3),
            #ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3),
            ConvBNLayer(self.full_name(),num_channels=num_channels,num_filters=out_channels3reduce,filter_size=1,act='relu'),
            ConvBNLayer(self.full_name(),num_channels=out_channels3reduce,num_filters=out_channels3,filter_size=3,padding=1,act='relu'),
            ConvBNLayer(self.full_name(),num_channels=out_channels3,num_filters=out_channels3,filter_size=3,padding=1,act='relu'),

        )

        self.branch4 = fluid.dygraph.Sequential(
            #nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            #ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
            Pool2D(pool_size=3,pool_stride=1,pool_padding=1,pool_type='avg'),
            ConvBNLayer(self.full_name(),num_channels=num_channels,num_filters=out_channels4,filter_size=1,act='relu'),

        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out =  fluid.layers.concat([out1, out2, out3, out4], axis=1)
        #torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV2ModulC(fluid.dygraph.Layer):
    def __init__(self, name_scope,num_channels,out_channels1, out_channels2):
        super(InceptionV2ModulC, self).__init__(name_scope)     
        self.branch1 = ConvBNLayer(self.full_name(),num_channels=num_channels,num_filters=out_channels1,filter_size=1,act='relu')
        self.branch2 = ConvBNLayer(self.full_name(),num_channels=out_channels1,num_filters=out_channels2,filter_size=3,stride=2,padding=1,act='relu')


    def forward(self, inputs):
        y = self.branch1(inputs)
        y = self.branch2(y)
        return y


class InceptionV2ModulDoubleC(fluid.dygraph.Layer):
    def __init__(self, name_scope,num_channels,out_channels1, out_channels2):
        super(InceptionV2ModulDoubleC, self).__init__(name_scope)     
        self.branch1 = ConvBNLayer(self.full_name(),num_channels=num_channels,num_filters=out_channels1,filter_size=1,act='relu')
        self.branch2 = ConvBNLayer(self.full_name(),num_channels=out_channels1,num_filters=out_channels2,filter_size=3,padding=1,act='relu')


    def forward(self, inputs):
        y = self.branch1(inputs)
        y = self.branch2(y)
        return y


class InceptionV2ModuleD(fluid.dygraph.Layer):
    def __init__(self, name_scope,num_channels,out_channels1,out_channels2,out_channels3):
        super(InceptionV2ModuleD, self).__init__(name_scope)

      #  self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)
     #   self.branch1 = ConvBNLayer(self.full_name(),num_channels=num_channels,num_filters=out_channels1,filter_size=1,act='relu')

        self.branch1 = fluid.dygraph.Sequential(
           # ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
           # ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, padding=1),
            ConvBNLayer(self.full_name(),num_channels=num_channels,num_filters=out_channels1,filter_size=1,act='relu'),
            ConvBNLayer(self.full_name(),num_channels=out_channels1,num_filters=out_channels2,filter_size=3,stride=2,padding=1,act='relu'),
        )

        self.branch2 = fluid.dygraph.Sequential(
            #ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            #ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3),
            #ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3),
            ConvBNLayer(self.full_name(),num_channels=num_channels,num_filters=out_channels2,filter_size=1,act='relu'),
            ConvBNLayer(self.full_name(),num_channels=out_channels2,num_filters=out_channels3,filter_size=3,padding=1,act='relu'),
            ConvBNLayer(self.full_name(),num_channels=out_channels3,num_filters=out_channels3,filter_size=3,stride=2,padding=1,act='relu'),

        )

        self.branch3 = fluid.dygraph.Sequential(
            #nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            #ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
            Pool2D(pool_size=3,pool_stride=2,pool_padding=1,pool_type='max'),

        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
 

       
        out =  fluid.layers.concat([out1, out2, out3], axis=1)
        #torch.cat([out1, out2, out3, out4], dim=1)
        return out





class Res3(fluid.dygraph.Layer):
    def __init__(self,
                name_scope,
                groups=1
                ):        
        super(Res3, self).__init__(name_scope)

        self._conv1 = Conv3D(
            num_channels=96,
            num_filters=128,
            filter_size=3,
            stride=1,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm1 = BatchNorm(128,act='relu')

        self._conv2 = Conv3D(
            num_channels=128,
            num_filters=128,
            filter_size=3,
            stride=1,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm2 = BatchNorm(128,act='relu')

        self._conv3 = Conv3D(
            num_channels=128,
            num_filters=128,
            filter_size=3,
            stride=1,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm3 = BatchNorm(128,act='relu')

    def forward(self,inputs):
        out1 = self._conv1(inputs)
        y = self._batch_norm1(out1)
        y = self._conv2(y)
        y = self._batch_norm2(y)
        y = self._conv3(y)
        y = fluid.layers.elementwise_add(x=y, y=out1, act=None)
        y = self._batch_norm3(y)

        return y


class Res4(fluid.dygraph.Layer):
    def __init__(self,
                name_scope,               
                groups=1
                ):        
        super(Res4, self).__init__(name_scope)

        self._conv1 = Conv3D(
            num_channels=128,
            num_filters=256,
            filter_size=3,
            stride=2,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm1 = BatchNorm(256,act='relu')

        self._conv2 = Conv3D(
            num_channels=256,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)

        self._conv3 = Conv3D(
            num_channels=128,
            num_filters=256,
            filter_size=3,
            stride=2,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm2 = BatchNorm(256,act='relu')
#########################################b
        self._conv4 = Conv3D(
            num_channels=256,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm3 = BatchNorm(256,act='relu')

        self._conv5 = Conv3D(
            num_channels=256,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm4 = BatchNorm(256,act='relu')

    def forward(self,inputs):
        y = self._conv1(inputs)
        y = self._batch_norm1(y)
        y = self._conv2(y)
        out1 = self._conv3(inputs)
        out2 = fluid.layers.elementwise_add(x=y, y=out1, act=None)
       ###b
        y = self._batch_norm2(out2)
        y = self._conv4(y)
        y = self._batch_norm3(y)
        y = self._conv5(y)        
        y = fluid.layers.elementwise_add(x=y, y=out2, act=None)
        y = self._batch_norm4(y)

        return y



class Res5(fluid.dygraph.Layer):
    def __init__(self,
                name_scope,
                groups=1):        
        super(Res5, self).__init__(name_scope)

        self._conv1 = Conv3D(
            num_channels=256,
            num_filters=512,
            filter_size=3,
            stride=2,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm1 = BatchNorm(512,act='relu')

        self._conv2 = Conv3D(
            num_channels=512,
            num_filters=512,
            filter_size=3,
            stride=1,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)

        self._conv3 = Conv3D(
            num_channels=256,
            num_filters=512,
            filter_size=3,
            stride=2,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm2 = BatchNorm(512,act='relu')
#########################################b
        self._conv4 = Conv3D(
            num_channels=512,
            num_filters=512,
            filter_size=3,
            stride=1,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm3 = BatchNorm(512,act='relu')

        self._conv5 = Conv3D(
            num_channels=512,
            num_filters=512,
            filter_size=3,
            stride=1,
            padding=1,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm4 = BatchNorm(512,act='relu')

    def forward(self,inputs):
        y = self._conv1(inputs)
        y = self._batch_norm1(y)
        y = self._conv2(y)
        out1 = self._conv3(inputs)
        out2 = fluid.layers.elementwise_add(x=y, y=out1, act=None)
       ######################b
        y = self._batch_norm2(out2)
        y = self._conv4(y)
        y = self._batch_norm3(y)
        y = self._conv5(y)
        y = fluid.layers.elementwise_add(x=y, y=out2, act=None)
        y = self._batch_norm4(y)

        return y



class TSNResNet(fluid.dygraph.Layer):
    def __init__(self,name_scope,layers=50,class_dim=10, seg_num =10,weight_devay=None):
        super(TSNResNet,self).__init__(name_scope)

        self.layers = layers
        self.seg_num = seg_num
        self.block1 = fluid.dygraph.Sequential(
            ConvBNLayer(self.full_name(),num_channels=3,num_filters=64,filter_size=7,stride=2,padding=3,act='relu'),#*64
            Pool2D(pool_size=3,pool_stride=2,pool_padding=1,pool_type='max'),#pool_padding=1
            ConvBNLayer(self.full_name(),num_channels=64,num_filters=64,filter_size=1,act='relu'),
            ConvBNLayer(self.full_name(),num_channels=64,num_filters=192,filter_size=3,padding=1,act='relu'),
            Pool2D(pool_size=3,pool_stride=2,pool_padding=1,pool_type='max'),#pool_padding=1
        
        )
        
        self._inseptionA = InceptionV2ModuleA(self.full_name(),num_channels=192,out_channels1=64,out_channels2reduce=64,out_channels2=64,out_channels3reduce=64,out_channels3=96,out_channels4=32)
        self._inseptionB = InceptionV2ModuleA(self.full_name(),num_channels=256,out_channels1=64,out_channels2reduce=64,out_channels2=96,out_channels3reduce=64,out_channels3=96,out_channels4=64)
        self._inseptionDoubleC = InceptionV2ModulDoubleC(self.full_name(),num_channels=320,out_channels1=64, out_channels2=96)

        self._inseptionC = InceptionV2ModulC(self.full_name(),num_channels=320,out_channels1=128, out_channels2=160)

        self._conv = ConvBNLayer(self.full_name(),num_channels=96,num_filters=96,filter_size=3,stride=2,padding=1,act='relu')##
        self._pool2d = Pool2D(pool_size=3,pool_stride=2,pool_padding=1,pool_type='max')#pool_padding=1

        self.D2nets = fluid.dygraph.Sequential(
            #inseption4a  
            InceptionV2ModuleA(self.full_name(),num_channels=576,out_channels1=224,out_channels2reduce=64,out_channels2=96,out_channels3reduce=96,out_channels3=128,out_channels4=128),
            #inseption4b 
            InceptionV2ModuleA(self.full_name(),num_channels=576,out_channels1=192,out_channels2reduce=96,out_channels2=128,out_channels3reduce=96,out_channels3=128,out_channels4=128),
            #inseption4c 
            InceptionV2ModuleA(self.full_name(),num_channels=576,out_channels1=160,out_channels2reduce=128,out_channels2=160,out_channels3reduce=128,out_channels3=160,out_channels4=128),  
            #inseption4d  
            InceptionV2ModuleA(self.full_name(),num_channels=608,out_channels1=96,out_channels2reduce=128,out_channels2=192,out_channels3reduce=160,out_channels3=192,out_channels4=128),
            #inseption4e 
            InceptionV2ModuleD(self.full_name(),num_channels=608,out_channels1=128,out_channels2=192,out_channels3=256),
            #inseption5a 
            InceptionV2ModuleA(self.full_name(),num_channels=1056,out_channels1=352,out_channels2reduce=192,out_channels2=320,out_channels3reduce=160,out_channels3=224,out_channels4=128),
            #inseption5b 
            InceptionV2ModuleA(self.full_name(),num_channels=1024,out_channels1=352,out_channels2reduce=192,out_channels2=320,out_channels3reduce=192,out_channels3=224,out_channels4=128),           			

        )


        self._res3 = Res3(self.full_name())
        self._res4 = Res4(self.full_name())
        self._res5 = Res5(self.full_name())
        
       # self._pool3d = fluid.layers.pool3d(pool_size=[1,7,7],pool_stride=1,pool_padding=1,pool_type='avg')
   

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self.out = Linear(input_dim=1536,
                          output_dim=class_dim,
                          act='softmax',
                          param_attr=fluid.param_attr.ParamAttr(
                             initializer=fluid.initializer.Uniform(-stdv,stdv)))

    def forward(self,inputs,label=None):
        out=fluid.layers.reshape(inputs,[-1,inputs.shape[2],inputs.shape[3],inputs.shape[4]])
        y = self.block1(out)
        y = self._inseptionA(y)
        y = self._inseptionB(y)

        out1 = self._inseptionC(y)     #   
        feature = self._inseptionDoubleC(y)
        out3 = self._pool2d(y)
        out2 = self._conv(feature)
        out3d = fluid.layers.reshape(x=feature,shape=[-1,feature.shape[1],self.seg_num,feature.shape[2],feature.shape[3]])
        y = self._res3(out3d)
        y = self._res4(y)
        y = self._res5(y)
    #    print(out1.shape)
    #    print(out2.shape)
    #    print(out3.shape)
        out2dnets = fluid.layers.concat([out1, out2, out3], axis=1)
        out = self.D2nets(out2dnets)

        out = fluid.layers.pool2d(input=out,pool_size=7,pool_stride=1,pool_type='avg')
        out = fluid.layers.dropout(out, dropout_prob=0.5)

        out = fluid.layers.reshape(x=out,shape=[-1,out.shape[1],self.seg_num,out.shape[2],out.shape[3]])	
        out = fluid.layers.pool3d(input=out,pool_size=[self.seg_num,1,1],pool_stride=1,pool_type='avg')
        last_duration = int(self.seg_num / 4)
        y = fluid.layers.pool3d(input=y,pool_size=[last_duration,7,7],pool_stride=1,pool_type='avg')
        y = fluid.layers.dropout(y, dropout_prob=0.3)
        out = fluid.layers.concat([out, y], axis=1)#
    #    print(out.shape)
        out = fluid.layers.reshape(x=out,shape=[-1,out.shape[1]])#
    #    print(out.shape)
        y = self.out(out)
    #    print(y.shape)
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y
 



if __name__ == '__main__':
     with fluid.dygraph.guard():
        network = TSNResNet('resnet', 101)
        img = np.zeros([1, 10, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = network(img).numpy()
        print(outs)

