import utt_parser

from dl_fasttext import FastText
from dl_textrcnn import TextRCNN
from ml_methods import MachineLearningClassification

if __name__ == "__main__":
    # 1. Parse Dataset
    files = [
         'JurafskyClustered_swbd-damsl_PlusProcessed',
        # 'Clustered_swbd-damsl',
         #'swbd_202k_42tags',
         #'verbmobil',
        #'swbd'
    ]
    parent_path = './data/'
    accuracies = []
    classifier = ['SVM','KNN','Logistic Regression','Naive Bayes','Decision','Random Tree']
    for file in files:
        print("====== DataSet: {}=======".format(file))
        filename = parent_path + file
        utt_parser.parser_xml_to_csv(filename)
        csv_file = file
        #ml_cls = MachineLearningClassification(filename + '.csv')
        #for algo in classifier:
            #accuracies.append(ml_cls.run(algo))

        dl_textrcnn = TextRCNN(filename + '.csv')
        dl_textrcnn.run()
        # dl_fasttext = FastText(filename + '.csv')
        # dl_fasttext.run()
    