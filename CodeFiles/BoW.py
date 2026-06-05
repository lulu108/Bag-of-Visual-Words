import argparse
import cv2
import numpy as np 
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
'''
python CodeFiles/BoW.py --train_path dataset/train --test_path dataset/test --no_clusters 100 --kernel linear 
'''
# 获取文件路径列表
def getFiles(train, path):
    images = []
    count = 0
    for folder in os.listdir(path): #遍历 path 目录下的所有类别文件夹
        for file in  os.listdir(os.path.join(path, folder)):#进入每个类别文件夹，遍历里面的图片文件
            images.append(os.path.join(path, os.path.join(folder, file)))

    # 训练集打乱有助于均衡采样
    if(train is True):
        np.random.shuffle(images)
    
    return images

def getDescriptors(sift, img):
    # 提取 SIFT 关键点与描述子
    kp, des = sift.detectAndCompute(img, None)
    return des #返回描述子

#读取并预处理图像
def readImage(img_path):
    # 灰度读取并统一尺寸，减少尺度差异
    img = cv2.imread(img_path, 0)
    return cv2.resize(img,(150,150))

#堆叠所有描述子
def vstackDescriptors(descriptor_list):
    # 将每张图的描述子堆叠为一个大矩阵
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 

    return descriptors

def clusterDescriptors(descriptors, no_clusters):
    # KMeans 形成视觉词典
    kmeans = KMeans(n_clusters = no_clusters).fit(descriptors)
    return kmeans

def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    # 统计每张图在视觉词典上的直方图表示
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features

def normalizeFeatures(scale, features):
    # 标准化使不同维度的计数可比
    return scale.transform(features)

def plotHistogram(im_features, no_clusters, fig_dir=None, kernel=None):
    # 绘制整体视觉词频分布
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.tight_layout()
    if fig_dir is not None and kernel is not None:
        save_path = os.path.join(fig_dir, f"vocabulary_frequency_k{no_clusters}_{kernel}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def svcParamSelection(X, y, kernel, nfolds):
    # 网格搜索选择 SVM 超参数
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
    gammas = [0.1, 0.11, 0.095, 0.105]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def findSVM(im_features, train_labels, kernel):
    features = im_features
    if(kernel == "precomputed"):
    # 预计算核需要 Gram 矩阵
        features = np.dot(im_features, im_features.T)
    
    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(C_param, gamma_param)
    # 通过类别权重缓解类别不平衡
    class_weight = {
        0: (807 / (7 * 140)),
        1: (807 / (7 * 140)),
        2: (807 / (7 * 133)),
        3: (807 / (7 * 70)),
        4: (807 / (7 * 42)),
        5: (807 / (7 * 140)),
        6: (807 / (7 * 142)) 
    }
  
    svm = SVC(kernel = kernel, C =  C_param, gamma = gamma_param, class_weight = class_weight)
    svm.fit(features, train_labels)
    return svm

def plotConfusionMatrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          save_path=None):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    return ax

def plotConfusions(true, predictions, fig_dir=None, no_clusters=None, kernel=None):
    np.set_printoptions(precision=2)

    class_names = ["city", "face", "green", "house_building", "house_indoor", "office", "sea"]
    save_path = None
    if fig_dir is not None and no_clusters is not None and kernel is not None:
        save_path = os.path.join(fig_dir, f"confusion_matrix_k{no_clusters}_{kernel}.png")

    plotConfusionMatrix(true, predictions, classes=class_names,
                      title='Confusion matrix, without normalization',
                      save_path=save_path)

    save_path = None
    if fig_dir is not None and no_clusters is not None and kernel is not None:
        save_path = os.path.join(fig_dir, f"confusion_matrix_normalized_k{no_clusters}_{kernel}.png")

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',
                      save_path=save_path)

    if fig_dir is None:
        plt.show()

def findAccuracy(true, predictions):
    # 输出分类准确率
    accuracy = accuracy_score(true, predictions)
    print ('accuracy score: %0.3f' % accuracy)
    return accuracy

def trainModel(path, no_clusters, kernel, fig_dir=None):
    images = getFiles(True, path)
    print("Train images path detected.")
    # 需要 opencv-contrib 的 SIFT
    sift = cv2.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])
    label_count = 7
    image_count = len(images)

    for img_path in images:
        # 从路径包含的类别名映射到标签
        if("city" in img_path):
            class_index = 0
        elif("face" in img_path):
            class_index = 1
        elif("green" in img_path):
            class_index = 2
        elif("house_building" in img_path):
            class_index = 3
        elif("house_indoor" in img_path):
            class_index = 4
        elif("office" in img_path):
          class_index = 5
        else:
          class_index = 6

        train_labels = np.append(train_labels, class_index)
        img = readImage(img_path)
        des = getDescriptors(sift, img)
        descriptor_list.append(des)

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptors vstacked.")

    kmeans = clusterDescriptors(descriptors, no_clusters)
    print("Descriptors clustered.")

    im_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)
    print("Images features extracted.")

    # 使用训练集统计量进行标准化
    scale = StandardScaler().fit(im_features)        
    im_features = scale.transform(im_features)
    print("Train images normalized.")

    plotHistogram(im_features, no_clusters, fig_dir=fig_dir, kernel=kernel)
    print("Features histogram plotted.")

    svm = findSVM(im_features, train_labels, kernel)
    print("SVM fitted.")
    print("Training completed.")

    return kmeans, scale, svm, im_features

def testModel(path, kmeans, scale, svm, im_features, no_clusters, kernel, fig_dir=None, result_dir=None):
    test_images = getFiles(False, path)
    print("Test images path detected.")

    count = 0
    true = []
    descriptor_list = []

    name_dict =	{
        "0": "city",
        "1": "face",
        "2": "green",
        "3": "house_building",
        "4": "house_indoor",
        "5": "office",
        "6": "sea"
    }

    sift = cv2.SIFT_create()

    for img_path in test_images:
        img = readImage(img_path)
        des = getDescriptors(sift, img)

        # 过滤无有效描述子的图片
        if(des is not None):
            count += 1
            descriptor_list.append(des)

            if("city" in img_path):
                true.append("city")
            elif("face" in img_path):
                true.append("face")
            elif("green" in img_path):
                true.append("green")
            elif("house_building" in img_path):
                true.append("house_building")
            elif("house_indoor" in img_path):
                true.append("house_indoor")
            elif("office" in img_path):
                true.append("office")
            else:
                true.append("sea")

    descriptors = vstackDescriptors(descriptor_list)

    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)

    test_features = scale.transform(test_features)
    
    kernel_test = test_features
    if(kernel == "precomputed"):
        # 测试集 Gram 矩阵需与训练集对齐
        kernel_test = np.dot(test_features, im_features.T)
    
    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print("Test images classified.")

    plotConfusions(true, predictions, fig_dir=fig_dir, no_clusters=no_clusters, kernel=kernel)
    print("Confusion matrixes plotted.")

    accuracy = findAccuracy(true, predictions)
    print("Accuracy calculated.")
    if result_dir is not None:
        cm = confusion_matrix(true, predictions)
        result_path = os.path.join(result_dir, f"result_k{no_clusters}_{kernel}.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(f"no_clusters: {no_clusters}\n")
            f.write(f"kernel: {kernel}\n")
            f.write(f"accuracy: {accuracy:.6f}\n")
            f.write("confusion_matrix:\n")
            f.write(str(cm))
            f.write("\n")
    print("Execution done.")

def execute(train_path, test_path, no_clusters, kernel, fig_dir=None, result_dir=None):
    # 训练 + 测试完整流程
    kmeans, scale, svm, im_features = trainModel(train_path, no_clusters, kernel, fig_dir=fig_dir)
    testModel(test_path, kmeans, scale, svm, im_features, no_clusters, kernel, fig_dir=fig_dir, result_dir=result_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', action="store", dest="train_path", required=True)
    parser.add_argument('--test_path', action="store", dest="test_path", required=True)
    parser.add_argument('--no_clusters', action="store", dest="no_clusters", default=50)
    parser.add_argument('--kernel_type', action="store", dest="kernel_type", default="linear")

    args =  vars(parser.parse_args())
    if(not(args['kernel_type'] == "linear" or args['kernel_type'] == "precomputed")):
        print("Kernel type must be either linear or precomputed")
        exit(0)

    output_dir = os.path.join("outputs", f"k{args['no_clusters']}_{args['kernel_type']}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_dir = os.path.join(output_dir, "figures")
    model_dir = os.path.join(output_dir, "models")
    result_dir = os.path.join(output_dir, "results")

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    execute(args['train_path'], args['test_path'], int(args['no_clusters']), args['kernel_type'], fig_dir=fig_dir, result_dir=result_dir)
