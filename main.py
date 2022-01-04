import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from utils import write_list_to_file
import stopwords

class classifier:
    # train Data
    trainData = pd.read_csv("dataset.csv")
    # test Data
    testData = pd.read_csv("test.csv")


    # Cria os vetores de caracteristicas, para executar sem remover stopwords basta remover o parametro
    stopwords = stopwords.stopword_l

    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 use_idf=True,
                                 strip_accents='ascii',
                                 stop_words=stopwords)
    train_vectors = vectorizer.fit_transform(trainData['text'])
    test_vectors = vectorizer.transform(testData['text'])

    nexec = 1
    kernels = []
    resultPisc = []
    resultComport = []
    resultFisio = []
    conta = 0
    while (conta <= nexec):
        conta += 1
        print('\n\nkernel linear')

        # kernel linear
        classifier_linear = svm.SVC(kernel='linear')
        classifier_linear.fit(train_vectors, trainData['classificacao'])
        prediction_linear = classifier_linear.predict(test_vectors)

        report_linear = classification_report(testData['label'], prediction_linear, output_dict=True)
        print('baixa: ', report_linear['baixa'])
        print('média: ', report_linear['média'])
        print('alta: ', report_linear['alta'])

        # add results in a list
        kernels.append('Linear')
        resultPisc.append(report_linear['baixa'])
        resultComport.append(report_linear['média'])
        resultFisio.append(report_linear['alta'])

        # kernel rbf SVM
        print('\n\nkernel rbf')
        classifier_rbf = svm.SVC(kernel='rbf', gamma='scale')
        classifier_rbf.fit(train_vectors, trainData['classificacao'])
        prediction_rbf = classifier_rbf.predict(test_vectors)

        report_rbf = classification_report(testData['label'], prediction_rbf, output_dict=True)
        print('média: ', report_rbf['média'])
        print('baixa: ', report_rbf['baixa'])
        print('alta: ', report_rbf['alta'])

        # add results in a list
        kernels.append('RBF')
        resultPisc.append(report_rbf['baixa'])
        resultComport.append(report_rbf['média'])
        resultFisio.append(report_rbf['alta'])

    else:
        write_list_to_file(kernels, resultComport, resultPisc, resultFisio)


if __name__ == '__main__':
    classifier
