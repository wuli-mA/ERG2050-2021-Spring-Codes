import numpy as np
import scipy.sparse  # Plaease install scipy: pip install scipy

class NaiveBayes(object):
    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None
        # Fill here (define your parameters, etc.)

    def train(self, train_X, train_y):
        classified=[]
        for y in range(20):
            y1 = []
            for i in range(len(train_y)):
                if train_y[i] == y:
                    y1.append(i)
            x1=[train_X[i] for i in y1]
            classified.append(x1)
        self.prior = np.array([(len(c))/(len(train_X)) for c in classified])
        self.avgs = np.array([np.mean(c,axis=0) for c in classified])
        self.vars = np.array([np.var(c,axis=0) for c in classified])
        self.n_class = 20
        # Fill here (estimate your parameters, etc.)

    def test(self, test_X, test_y):
        # Fill here (predict, evaluation, etc.)
        print('Please wait for about 10 seconds...\n')
        likelihoods = np.array([np.sum(-np.log(np.sqrt(2 * np.pi * (self.vars[c]+0.0001)))+
            (-(test_X[0]-self.avgs[c])**2/(2 * (self.vars[c] + 0.0001)))) for c in range(self.n_class)])
        for x in range(1,len(test_X)):
            likelihood = np.array([np.sum(-np.log(np.sqrt(2 * np.pi * (self.vars[c]+0.0001)))+
                (-(test_X[x]-self.avgs[c])**2/(2 * (self.vars[c] + 0.0001)))) for c in range(self.n_class)])
            likelihood *= self.prior
            likelihoods = np.vstack((likelihoods,likelihood))
        posteriors = likelihoods
        # posteriors = likelihoods*self.prior
        pred_y = np.argmax(posteriors,axis = 1) # your predicted results, it should be a vector with shape (7532,)
        accuracy = (pred_y == test_y).sum() / test_y.shape[0] # calculate score (do NOT modify this)
        print('\nThe accuracy in test set: {:.2f}%.'.format(accuracy*100))


def main():
    train_X, test_X = scipy.sparse.load_npz('training_feats.npz').toarray(), scipy.sparse.load_npz('test_feats.npz').toarray() # DO NOT modify the PATH
    train_y, test_y = np.load('training_labels.npy', allow_pickle=True), np.load('test_labels.npy', allow_pickle=True) # DO NOT modify the PATH
    nb = NaiveBayes()
    nb.train(train_X, train_y)
    nb.test(test_X, test_y)


if __name__ == '__main__':
    main()
