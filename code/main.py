import numpy
import matplotlib.pyplot as plt
import importlib
import sys
import os
import utils
import GaussianClassifier as GC
import LogisticRegression as LR
import SVM as SVM
import GMM
# ,  LogisticRegression as LR, SVM, GMM


# # # # # # # FUNCTIONS # # # # # # #


# # Load data
def load_data(defPath=""):
    print("Loading data")
    # # class 1 -> Positive pulsar signal
    # # class 0 -> Negative pulsar signal
    (data_train, labels_train), (data_test, lables_test) = utils.load_dataset_shuffle(
        defPath + "data/Train.txt", defPath + "data/Test.txt", 12
    )
    data_traing, data_testg = utils.features_gaussianization(data_train, data_test)
    print("Done.\n\n")
    return (data_train, labels_train), (data_test, lables_test)


# # Plot of the features

def plot_features(data_train, labels_train):
    print("Plotting features ...")
    utils.plot_features(data_train, labels_train, "plot_raw_features")
    utils.plot_correlations(data_train, labels_train)
    print("Done.\n\n")


def gaussian_classifier_report(data_train, labels_train):
    print("Gaussian Classifiers report:")
    iter = 6
    model = GC.GaussianClassifier()
    data_trainpca = data_train
    print("Apply K-Folds where K=7")
    for i in range(iter):
        print(f"# PCA m = {data_train.shape[0] - i}" if i > 0 else " >> RAW")
        if i > 0:
            PCA_ = utils.PCA(data_train, data_train.shape[0] - i)
            data_trainpca = PCA_[0]
        print("Full-Covariance")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca, labels_train, prior, model, ([prior, 1 - prior])
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("Diagonal-Covariance")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca, labels_train, prior, model, ([prior, 1 - prior], "NBG")
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("Tied Full-Covariance")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca,
                labels_train,
                prior,
                model,
                ([prior, 1 - prior], "MVG", True),
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("Tied Diagonal-Covariance")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca,
                labels_train,
                prior,
                model,
                ([prior, 1 - prior], "NBG", True),
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
    print("\n")
    print("Single-Split")
    for i in range(iter):
        print(f"# PCA m = {data_train.shape[0] - i}" if i > 0 else " >> RAW")
        if i > 0:
            PCA_ = utils.PCA(data_train, data_train.shape[0] - i)
            data_trainpca = PCA_[0]
        print("Full-Covariance")
        for prior in priors:
            minDCF = utils.single_split(
                data_trainpca, labels_train, prior, model, ([prior, 1 - prior])
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("Diagonal-Covariance")
        for prior in priors:
            minDCF = utils.single_split(
                data_trainpca, labels_train, prior, model, ([prior, 1 - prior], "NBG")
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("Tied Full-Covariance")
        for prior in priors:
            minDCF = utils.single_split(
                data_trainpca,
                labels_train,
                prior,
                model,
                ([prior, 1 - prior], "MVG", True),
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("Tied Diagonal-Covariance")
        for prior in priors:
            minDCF = utils.single_split(
                data_trainpca,
                labels_train,
                prior,
                model,
                ([prior, 1 - prior], "NBG", True),
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
    print("\n\n")

def logistic_regression_report(data_train, labels_train):
    print("Logistic Regression report:")
    model = LR.LogisticRegression()
    data_trainpca = data_train
    print("Plotting minDCF graphs")
    l = numpy.logspace(-5, 1, 10)
    for i in range(3):  # raw, pca7, pca6
        y5, y1, y9 = [], [], []
        title = "raw"
        if i > 0:
            PCA_ = utils.PCA(data_train, data_train.shape[0] - i)
            data_trainpca = PCA_[0]
            title = f"pca{data_train.shape[0] - i}"
        for il in l:
            y5.append(
                utils.kfolds(
                    data_trainpca, labels_train, priors[0], model, (il, priors[0])
                )[0]
            )
            y1.append(
                utils.kfolds(
                    data_trainpca, labels_train, priors[1], model, (il, priors[0])
                )[0]
            )
            y9.append(
                utils.kfolds(
                    data_trainpca, labels_train, priors[2], model, (il, priors[0])
                )[0]
            )
        utils.plot_minDCF_lr(
            l, y5, y1, y9, f"{title}_5-folds", f"5-folds / {title} / πT = 0.5"
        )
    print("Done.")
    print("# # 5-folds")
    for i in range(3):  # raw, pca7, pca6
        print(f"# PCA m = {data_train.shape[0] - i}" if i > 0 else "# RAW")
        if i > 0:
            PCA_ = utils.PCA(data_train, data_train.shape[0] - i)
            data_trainpca = PCA_[0]
        print("LogReg(λ = 1e-5, πT = 0.5)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca, labels_train, prior, model, (1e-5, priors[0])
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("LogReg(λ = 1e-5, πT = 0.1)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca, labels_train, prior, model, (1e-5, priors[1])
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("LogReg(λ = 1e-5, πT = 0.9)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca, labels_train, prior, model, (1e-5, priors[2])
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
    print("\n\n")



# # Linear SVM
def linear_svm_report(data_train, labels_train):
    print("Support Vector Machine report:")
    model = SVM.SVM()
    data_trainpca = data_train
    print("Plotting minDCF graphs ...")
    C = numpy.logspace(-4, 1, 10)
    for mode in ["unbalanced", "balanced"]:
        for i in priors:
            y5, y1, y9 = [], [], []
            PCA_ = utils.PCA(data_train, 11)
            data_trainpca = PCA_[0]
            title = f"pca7"
            for iC in C:
                y5.append(
                    utils.kfolds(
                        data_trainpca,
                        labels_train,
                        priors[0],
                        model,
                        ("linear", i, mode == "balanced", 1, iC),
                    )[0]
                )
                y1.append(
                    utils.kfolds(
                        data_trainpca,
                        labels_train,
                        priors[1],
                        model,
                        ("linear", i, mode == "balanced", 1, iC),
                    )[0]
                )
                y9.append(
                    utils.kfolds(
                        data_trainpca,
                        labels_train,
                        priors[2],
                        model,
                        ("linear", i, mode == "balanced", 1, iC),
                    )[0]
                )
            utils.plot_minDCF_svm(
                C,
                y5,
                y1,
                y9,
                f"linear_{title}_{mode}{i}_5-folds",
                f'5-folds / {title} / {f"πT = {i}" if mode == "balanced" else "unbalanced"}',
            )
            if mode == "unbalanced":
                break
    print("Done.")
    print("5-folds")
    for i in range(2):  # raw, pca7
        print(f"# PCA m = {data_train.shape[0] - i}" if i > 0 else "# RAW")
        if i > 0:
            PCA_ = utils.PCA(data_train, data_train.shape[0] - i)
            data_trainpca = PCA_[0]
        print("Linear SVM(C = 1e-1)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca,
                labels_train,
                prior,
                model,
                ("linear", priors[0], False, 1, 1e-2),
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("Linear SVM(C = 1e-1, πT = 0.5)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca,
                labels_train,
                prior,
                model,
                ("linear", priors[0], True, 1, 1e-2),
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("Linear SVM(C = 1e-1, πT = 0.1)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca,
                labels_train,
                prior,
                model,
                ("linear", priors[1], True, 1, 1e-2),
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("Linear SVM(C = 1e-1, πT = 0.9)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca,
                labels_train,
                prior,
                model,
                ("linear", priors[2], True, 1, 1e-2),
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
    print("\n\n")

def quadratic_svm_report(data_train, labels_train):
    print("RBF SVM, Poly SVM report:")
    model = SVM.SVM()
    data_trainpca = data_train
    print("Plotting minDCF graphs ...")
    C = numpy.logspace(-4, 1, 10)
    for i in range(2):  # raw, pca7
        y5, y1, y9 = [], [], []
        title = "raw"
        if i > 0:
            PCA_ = utils.PCA(data_train, data_train.shape[0] - i)
            data_trainpca = PCA_[0]
            title = f"pca{data_train.shape[0] - i}"
        for iC in C:
            y5.append(
                utils.kfolds(
                    data_trainpca,
                    labels_train,
                    priors[0],
                    model,
                    ("poly", priors[0], False, 1, iC, 1, 2),
                )[0]
            )
            y1.append(
                utils.kfolds(
                    data_trainpca,
                    labels_train,
                    priors[1],
                    model,
                    ("poly", priors[0], False, 1, iC, 10, 2),
                )[0]
            )
            y9.append(
                utils.kfolds(
                    data_trainpca,
                    labels_train,
                    priors[2],
                    model,
                    ("poly", priors[0], False, 1, iC, 100, 2),
                )[0]
            )
        utils.plot_minDCF_svm(
            C,
            y5,
            y1,
            y9,
            f"poly_{title}_unbalanced_5-folds",
            f"5-folds / {title} / unbalanced",
            type="poly",
        )
    for i in range(2):  # raw, pca7
        y5, y1, y9 = [], [], []
        title = "raw"
        if i > 0:
            PCA_ = utils.PCA(data_train, data_train.shape[0] - i)
            data_trainpca = PCA_[0]
            title = f"pca{data_train.shape[0] - i}"
        for iC in C:
            y5.append(
                utils.kfolds(
                    data_trainpca,
                    labels_train,
                    priors[0],
                    model,
                    ("RBF", priors[0], False, 1, iC, 0, 0, 1e-3),
                )[0]
            )
            y1.append(
                utils.kfolds(
                    data_trainpca,
                    labels_train,
                    priors[1],
                    model,
                    ("RBF", priors[0], False, 1, iC, 0, 0, 1e-2),
                )[0]
            )
            y9.append(
                utils.kfolds(
                    data_trainpca,
                    labels_train,
                    priors[2],
                    model,
                    ("RBF", priors[0], False, 1, iC, 0, 0, 1e-1),
                )[0]
            )
            print("RBF GRAPH")
        utils.plot_minDCF_svm(
            C,
            y5,
            y1,
            y9,
            f"rbf_{title}_unbalanced_5-folds",
            f"5-folds / {title} / unbalanced",
            type="RBF",
        )
    print("Done.")
    print("# # 5-folds")
    for i in range(2):  # raw, pca7
        print(f"# PCA m = {data_train.shape[0] - i}" if i > 0 else "# RAW")
        if i > 0:
            PCA_ = utils.PCA(data_train, data_train.shape[0] - i)
            data_trainpca = PCA_[0]
        print("RBF SVM(C = 1e-1, γ = 1e-3)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca,
                labels_train,
                prior,
                model,
                ("RBF", priors[0], False, 1, 1e-1, 0, 0, 1e-3),
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("Poly SVM(C = 1e-3, c = 1, d = 2)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca,
                labels_train,
                prior,
                model,
                ("poly", priors[0], False, 1, 1e-3, 1, 2, 0),
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
    print("\n\n")




def gmm_report(data_train, labels_train):
    print("GMM report:")
    model = GMM.GMM()
    data_trainpca = data_train
    print("Plotting minDCF graphs ...")
    components = [2, 4, 8, 16, 32]
    for type in ["full", "tied", "diag"]:
        for i in range(2):  # raw, pca7
            y5, y1, y9 = [], [], []
            title = "raw"
            if i > 0:
                PCA_ = utils.PCA(data_train, data_train.shape[0] - i)
                data_trainpca = PCA_[0]
                title = f"pca{data_train.shape[0] - i}"
            for c in components:
                y5.append(
                    utils.kfolds(
                        data_trainpca, labels_train, priors[0], model, (c, type)
                    )[0]
                )
                y1.append(
                    utils.kfolds(
                        data_trainpca, labels_train, priors[1], model, (c, type)
                    )[0]
                )
                y9.append(
                    utils.kfolds(
                        data_trainpca, labels_train, priors[2], model, (c, type)
                    )[0]
                )
            utils.plot_minDCF_gmm(
                components, y5, y1, y9, f"{type}_{title}", f"gmm {type}-cov / {title}"
            )
    print("Done.")
    print("# # 5-folds")
    for i in range(2):  # raw, pca7
        print(f"# PCA m = {data_train.shape[0] - i}" if i > 0 else "# RAW")
        if i > 0:
            PCA_ = utils.PCA(data_train, data_train.shape[0] - i)
            data_trainpca = PCA_[0]
        print("GMM Full (8 components)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca, labels_train, prior, model, (8, "full")
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("GMM Diag (16 components)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca, labels_train, prior, model, (16, "diag")
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
        print("GMM Tied (32 components)")
        for prior in priors:
            minDCF = utils.kfolds(
                data_trainpca, labels_train, prior, model, (32, "tied")
            )[0]
            print(f"- with prior = {prior} -> minDCF = %.3f" % minDCF)
    print("\n\n")




# # Gaussian classifiers

# # # # # # # FUNCTIONS # # # # # # #


if __name__ == "__main__":
    importlib.reload(utils)
    importlib.reload(GC)
    importlib.reload(LR)
    importlib.reload(SVM)
    importlib.reload(GMM)

    priors = [0.5, 0.1, 0.9]

    (data_train, labels_train), (data_test, lables_test) = load_data()

    #plot_features(data_train, labels_train)
    #gaussian_classifier_report(data_train, labels_train)
    #logistic_regression_report(data_train, labels_train)
    #linear_svm_report(data_train, labels_train)
    #quadratic_svm_report(data_train, labels_train)
    #gmm_report(data_train, labels_train)
    #utils.plot_LDA(data_train, labels_train)
    utils.plot_scatter(data_train, labels_train)
    # score_calibration_report(data_train, labels_train)
    # evaluation_report(data_train, labels_train, data_test, lables_test)

    print("\n\n ------ END ------")
