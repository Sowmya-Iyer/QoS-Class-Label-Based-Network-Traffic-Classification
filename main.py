import pandas as pd
import os
import sys
import numpy as np
import pickle
import glob
import argparse
import collections
from IPython.display import Image
import matplotlib.pyplot as plt
import itertools
from sklearn.utils import shuffle
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from dataPreprocess import preprocess
from imblearn.over_sampling import SMOTE


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='QoS Classification')
    parser.add_argument('data', metavar='DATA_FILE/FOLDER',
                        help='the folder or file that contains all the input data')
    parser.add_argument('-f', '--file_type', choices=['pkl','csv'], default='pkl',required=True,
                        help='Type of file entered')
    parser.add_argument('-t', '--train', action='store_true', default=False,
                        help='Train model')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='Only test model')
    parser.add_argument('-l','--label_col', action='store', type=str, default='Label',
                        help='Custom label column name')
    parser.add_argument('-m', '--model', choices=["logistic", "naive_bayes", "decision_trees", "random_forest"], default='decision_trees',action='store', type=str, 
                        help='ML Model (DEFAULT: decision_trees)')
    parser.add_argument('-p', '--predict', action='store_true', default=False,
                        help='Predict on unlabelled data')
    
    parser.add_argument('-o', '--oversample', action='store_true', default=False,
                        help='Oversample training data')

    # parser.add_argument('-n', '--n_training_examples', type=int, default=-1,
    #                     help='number of training examples used. -1 means all. (DEFAULT: -1)')

    args = parser.parse_args(argv)
    return args


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.rcParams.update({'font.size': 10})
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else ' '
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# classes=['Best Effort','Critical voice RTP' ,'Flash Override','Flash voice', 'Immediate','Internetwork control','Network Control','Not Known','Priority]
def set_labels(labels):
    global classes
    classes = labels
def train(data_transformed, model,label_col,oversampling=True):
    y_name =[label_col]
    Y = data_transformed[y_name][:]
    le = preprocessing.LabelEncoder().fit(Y)
    set_labels(le.inverse_transform([0,1,2,3,4,5,6,7,8]))
    Y_sm= Y = le.transform(Y)
    X= data_transformed.drop(y_name, axis=1)
    X_sm=X = np.asarray(X)[:]
    # classes = le.inverse_transform([0,1,2,3,4,5,6,7])
    if oversampling:
      oversample= SMOTE('auto')
      X_sm,Y_sm = oversample.fit_resample(X, Y)
    print("Training set:")
    print("X: ")
    print(np.shape(X_sm))
    print("y: ")
    print(np.shape(Y_sm))

    X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, test_size=0.33, shuffle=True)

    Ml_Algo = ["logistic", "naive_bayes", "decision_trees", "random_forest"]


    if model.lower() in Ml_Algo:
      method = model.lower()

    print("Model used for classification problem: " + method)
    print()

    if method == "logistic" :
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
    
    elif method == "naive_bayes":
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
    
    elif method == "decision_trees":
        # Fitting Decision Tree Classification to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
    
    elif method == "random_forest":
        # Fitting Random Forest Classification to the      
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    print(f"Classifer description: {classifier}")
    print("Saving the model")
    try:
      os.mkdir(method)
    except OSError:  
        print ("Creation of the directory %s failed" % method)
    else:
        print ("Created the directory %s" % method)
    if oversampling:
      filename = method+"/"+method+"_model_oversampled.sav"
    else:
      filename = method+"/"+method+"_model.sav"
    pickle.dump(classifier, open(filename,'wb'))

    print("Model saved as %s"% filename)
    test(X_test,y_test,method,oversampling)

def test(X_test,y_test,method,oversampling):
    print("Testing:")
    if oversampling:
      filename = method+"/"+method+"_model_oversampled.sav"
    else:
      filename = method+"/"+method+"_model.sav"
    classifier= pickle.load(open(filename, 'rb'))
    y_pred = classifier.predict(X_test)
    print('accuracy %s' % accuracy_score(y_test, y_pred))
    print("Classes: ", classes)
    cm=classification_report(y_test, y_pred, target_names= classes, labels=range(9))
    print(cm)
    # print("Classes: ", le.inverse_transform([0,1,2,3,4,5,6,7]))
    # print(classification_report(y_test, y_pred, target_names= le.inverse_transform([0,1,2,3,4,5,6,7]), labels=range(8)))
    
    fig = plt.figure()
    fig.set_size_inches(15, 15, forward=True)
    fig.align_labels()

    #fig.subplots_adjust(left=-3.0, right=-2.0, bottom=0.0, top=1.0)
    print(confusion_matrix(y_test, y_pred))
    if oversampling:
      method_name = method+"_model_oversampled"
    else:
      method_name = method+"_model"
    np.save(method + "/ConfusionMatrix_using_" + method_name, cm)

    print("Saved Confusion Matrix")
    # plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=True,
    #                       title='Confusion matrix', classes= le.inverse_transform([0,1,2,3,4,5,6,7]))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=True,
                           title='Confusion matrix', classes= classes)
    
    fig.savefig(method+"/"+method_name+"_cm_norm.png")
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=False,
                          title='Confusion matrix', classes= classes)
    # plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=False,
    #                       title='Confusion matrix', classes= le.inverse_transform([0,1,2,3,4,5,6,7]))
    fig.savefig(method+"/"+ method_name+"_cm_abs.png")
global classes
classes=['Best Effort','Critical voice RTP' ,'Flash Override','Flash voice', 'Immediate','Internetwork control','Network Control','Not Known','Priority']
def test_only(data_transformed,method,label_col,oversampling):
    y_name =[label_col]
    Y = data_transformed[y_name][:]
    le = preprocessing.LabelEncoder().fit(classes)
    # print(le.inverse_transform([0,1,2,3,4,5,6,7,8]))
    # set_labels(le.inverse_transform([0,1,2,3,4,5,6,7,8]))
    y_test=  le.transform(Y)
    X= data_transformed.drop(y_name, axis=1)
    X_test= np.asarray(X)[:]
    print("Testing Data:")
    print(method)
    if oversampling:
      filename = method+"/"+method+"_model_oversampled.sav"
    else:
      filename = method+"/"+method+"_model.sav"
    classifier= pickle.load(open(filename, 'rb'))
    y_pred = classifier.predict(X_test)
    print('accuracy %s' % accuracy_score(y_test, y_pred))
    # classes=le.inverse_transform([0,1,2,3,4,5,6,7,8])
    print("Classes: ", classes)
    cm=classification_report(y_test, y_pred, target_names= classes, labels=range(9))
    print(cm)
    # print("Classes: ", le.inverse_transform([0,1,2,3,4,5,6,7]))
    # print(classification_report(y_test, y_pred, target_names= le.inverse_transform([0,1,2,3,4,5,6,7]), labels=range(8)))
    
    fig = plt.figure()
    fig.set_size_inches(15, 15, forward=True)
    fig.align_labels()

    #fig.subplots_adjust(left=-3.0, right=-2.0, bottom=0.0, top=1.0)
    print(confusion_matrix(y_test, y_pred))
    test_dir ="Test_results"
    try:
        os.mkdir(test_dir)
    except OSError:  
        print (test_dir+" directory already exists")
    else:
        print ("Created the directory %s" % test_dir)
    if oversampling:
      method_name = method+"_model_oversampled"
    else:
      method_name = method+"_model"
    np.save(test_dir+"/ConfusionMatrix_using_" + method_name, cm)

    print("Saved Confusion Matrix")
    # plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=True,
    #                       title='Confusion matrix', classes= le.inverse_transform([0,1,2,3,4,5,6,7]))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=True,
                           title='Confusion matrix', classes= classes)
    
    fig.savefig(test_dir+"/"+method_name+"_cm_norm.png")
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=False,
                          title='Confusion matrix', classes= classes)
    # plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=False,
    #                       title='Confusion matrix', classes= le.inverse_transform([0,1,2,3,4,5,6,7]))
    fig.savefig(test_dir+"/"+ method_name+"_cm_abs.png")

def predict(data, method,oversampling=True):
  # data_transformed=preprocess(data,False)
  X_pred = np.asarray(data)[:]
  if oversampling:
    filename = method+"/"+method+"_model_oversampled.sav"
  else:
    filename = method+"/"+method+"_model.sav"
  # filename = method+"/"+method+"_model.sav"
  classifier= pickle.load(open(filename, 'rb'))
  y_pred = classifier.predict(X_pred)
  print(set(list(y_pred)))
  # data_transformed['QoS_predicted'] = y
  result = [classes[i] for i in y_pred]
  data['Label']=result
  return data

def main():
    path= args.data
    if os.path.isdir(path):  
        if args.file_type == 'pkl':
            print("Reading .pkl files from %s" % args.data)
            files = glob.glob(path+'/*pkl')
            df = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)
        else:
            print("Reading .csv files from %s" % args.data)
            files = glob.glob(path+'/*csv')
            df = pd.concat([pd.read_csv(fp) for fp in files], ignore_index=True)
    elif os.path.isfile(path):  
        if args.file_type == 'pkl':
            print("Reading .pkl file: %s" % args.data)
            # files = glob.glob(args.data+'/*pkl')
            df = pd.read_pickle(path)
        else:
            print("Reading .csv file: %s" % args.data)
            df = pd.read_csv(path) 
    else:  
        print("File type not in [.csv,.pkl]" )
        sys.exit()
    # files = glob.glob('dump/*pkl')
# for file in files:
#   df_temp=pd.read_pickle(file)
#   print(df_temp.shape)
    # df = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)
# df['Protocol'].value_counts()
    model = args.model
    oversample = args.oversample

    if args.train:
        label= args.label_col
        data_transformed= preprocess(df,label,True)
        train(data_transformed,model,label,oversample)
    
    elif args.test_only:
        label= args.label_col
        data_transformed= preprocess(df,label,True)
        test_only(data_transformed,model,label,oversample)
        # file='/content/drive/MyDrive/cs536/data/20200325_10.csv'
        
    elif args.predict:
        label= args.label_col
        data_transformed= preprocess(df,label,False)
# data_transformed.columns
# X_pred = np.asarray(data_transformed)[:]
        result=predict(data_transformed,model,oversample)
        pred_dir='Predicted_results'
        try:
            os.mkdir(pred_dir)
        except OSError:  
            print (pred_dir+" directory already exists")
        else:
            print ("Created the directory %s" % pred_dir)
        name=path.split('.')[0].split('/')[-1]
        filename=pred_dir+'/result_'+name+'.csv'
        print("Result stored in %s" % filename)
        result.to_csv(filename, index=True)  


if __name__ == '__main__':
    args = get_arguments(sys.argv[1:])
    # args = utils.bin_config(get_arguments)
    main()
