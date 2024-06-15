import pandas as pd
import plotly.express as px
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB4
import os
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 导入train.csv
train_df = pd.read_csv("D:/MyWord/DeepLearning/PaddyDiseaseClassification/train.csv")
# 显示train.csv的前五行数据，确认数据正确加载和数据格式
print(train_df.head())

# 统计label列每个类别对应的样本数，了解类别分布，检查数据不平衡、数据质量
print(train_df['label'].value_counts())

# 统计label列不同类别数，以确认是否是10个类别
print(train_df['label'].nunique())

rescale = tf.keras.layers.Rescaling(1./255)

# 数据增强和归一化
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 训练集
train_ds = train_datagen.flow_from_directory(
    'D:/MyWord/DeepLearning/PaddyDiseaseClassification/train_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    seed=123
)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
# 验证集
validation_ds = validation_datagen.flow_from_directory(
    'D:/MyWord/DeepLearning/PaddyDiseaseClassification/train_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=123
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# 测试集
test_ds = test_datagen.flow_from_directory(
    'D:/MyWord/DeepLearning/PaddyDiseaseClassification/',
    classes=['test_images'],  # Specify the subdirectory containing the images
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

# 散点图，展示不同水稻品种和年龄之间的关系
fig = px.scatter(train_df, x="age", y="variety", color="label")
fig.show()
# 条形图，展示每种病害类型的平均年龄
fig = px.bar(train_df, x='label', y='age', color='label')
fig.show()
# 旭日图，展示病害类型和水稻品种之间的层次结构关系
fig = px.sunburst(train_df, path=['label', 'variety'], values='age', color='label')
fig.show()

# 使用迁移学习的预训练模型
efficientnet_base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# 解除冻结预训练的基础模型层
efficientnet_base.trainable = True

# 模型构建
model = Sequential()
# 将预训练的 EfficientNetB4 基础模型添加到新模型中
model.add(efficientnet_base)
# 全局平均池化层
model.add(AveragePooling2D(pool_size=(2, 2)))
# Flatten层
model.add(Flatten())
# 全连接层
model.add(Dense(220, activation='relu'))
# Dropout层
model.add(Dropout(0.25))
# 输出层
model.add(Dense(10, activation='softmax'))

# model.summary()

# 学习率
base_learning_rate = 0.0001
# 模型编译
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# 早停法
early_stopping = EarlyStopping(patience=10)

# 模型训练
history = model.fit(train_ds,
                    validation_data=validation_ds,
                    epochs=100,
                    callbacks=[early_stopping])

# 评估模型
# 评估模型在验证集上的性能。该方法将返回模型的损失值和指定的评估指标值（在这里是准确率）。
loss = model.evaluate(validation_ds)

# 绘制训练集和验证集损失值变化曲线
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# 绘制训练集和验证集准确率变化曲线
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

labels = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast',
          'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

# 预测测试集的标签
predictions = model.predict(test_ds)
# 对预测结果进行解码
predicted_labels = [labels[prediction.argmax()] for prediction in predictions]
set(predicted_labels)

# 创建提交文件
submission_df = pd.DataFrame({'image_id': test_ds.filepaths, 'label': predicted_labels})

# 去掉路径中的前缀
submission_df['image_id'] = submission_df['image_id'].apply(lambda x: x.split('\\')[-1])

submission_df.to_csv('D:/MyWord/DeepLearning/PaddyDiseaseClassification/sample_submission.csv', index=False)

print(submission_df.head())
