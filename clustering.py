import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

class clustering:
    
    def __init__(self):
        self.image = Image.open(sys.argv[1]).convert('RGB')
        self.k = int(sys.argv[2])
    
    def KMeans(self):
        image = np.asarray(self.image)
        pixelVal = image.reshape((-1, 3))

        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(pixelVal)
        
        centers = np.uint8(kmeans.cluster_centers_)
        labels = kmeans.labels_.flatten()

        segImg = centers[labels.flatten()]
        segImg = segImg.reshape(image.shape)
        im = Image.fromarray(segImg)
        im.show()



def main():
    cluster = clustering()
    cluster.KMeans()


if __name__ == "__main__":
    main()


