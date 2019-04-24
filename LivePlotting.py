import matplotlib.pyplot as plt
import numpy as np
class PlotSignal:
    def __init__(self, window=10000,bins=10):
        self.window = window
        self.values = {}
        self.bins=bins

    def update(self, **argv):
        import numpy as np
        for k in argv:
            value=argv[k]
            # flatting array of a batch of observations
            if k not in self.values:
                self.values[k] = [value]
                self.values[k]=np.reshape(self.values[k],[-1])
            else:
                self.values[k]=np.append(self.values[k],value)
                self.values[k]=np.reshape(self.values[k],[-1])

            self.values[k] = self.values[k][-self.window:]

    def update_dict(self, mydict):
        import numpy as np
        for k,v in mydict.items():
            # flatting array of a batch of observations
            if k not in self.values:
                self.values[k] = [v]
                self.values[k] = np.reshape(self.values[k], [-1])
            else:
                self.values[k] = np.append(self.values[k], v)
                self.values[k] = np.reshape(self.values[k], [-1])

            self.values[k] = self.values[k][-self.window:]

    def plot_signal(self, name=None):
        N = len(self.values)
        plt.clf()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)
            plt.subplots_adjust(hspace=0.5)
            plt.plot(self.values[k])
        if name is not None:
            plt.savefig(name)
        plt.pause(0.0000001)
        


    def plot_hist(self):
        N = len(self.values)
        plt.clf()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)
            print("k=",k)
            print("values.shape=",np.shape(self.values[k]))
            print("values[k]=",self.values[k])
            plt.hist(self.values[k],bins=self.bins)

        plt.pause(0.0000001)


    def last_plot(self):
        N = len(self.values)
        plt.clf()
        plt.ioff()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.show()


if __name__=="__main__":
    from config import config
    from utils import load_np_array_from_bin_file
    import os
    sub_dir="with_attention_glove_embeddings"
    config.log_dir_path=os.path.join(config.log_dir_path,sub_dir)
    accuracy=load_np_array_from_bin_file(config,"accuracy.npy")
    loss=load_np_array_from_bin_file(config,"loss.npy")

    myarrays={}
    myarrays.update({'accuracy': accuracy })
    myarrays.update({'loss': loss })

    print("myarray=",myarrays)
    plotting=PlotSignal()

    plotting.update_dict(myarrays)

    #plotting.update(myarrays)
    plotting.plot_signal("loss and accuracy.jpg")
    #plotting.plot_hist()