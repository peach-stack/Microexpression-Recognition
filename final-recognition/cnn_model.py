from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

def model1(shape, num_classes, l2_regularization=0.01):
    #输入图像的形状、分类个数、正则化参数
    regularization = l2(l2_regularization) #正则化参数

    # 基础设置
    img = Input(shape)
    # 2D卷积层（8个过滤器，过滤器大小为3*3）
    x = Conv2D(8, (3, 3), strides=(1, 1), use_bias=False,kernel_regularizer=regularization)(img)
    # kernel_regularization对该层中的权值进行正则化，亦即对权值进行限制，使其不至于过大。use_bias不使用偏置
    x = BatchNormalization()(x)
    #对输入激活函数的数据进行归一化，解决输入数据发生偏移和增大的影响。
    x = Activation('relu')(x)
    #激活函数层，采用Relu函数
    x = Conv2D(8, (3, 3), strides=(1, 1),use_bias=False,kernel_regularizer=regularization)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module1
    residual = Conv2D(16, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    #深度可分离2D卷积，边缘补充像素，保证输出像素与输入相同
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    #最大池化
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    #返回x+residual

    # module2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module3
    residual = Conv2D(64, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module4
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),padding='same')(x)
    #增加一个全局平均池化层
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img, output)
    return model

if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    model = model1(input_shape, num_classes)
    model.summary()
