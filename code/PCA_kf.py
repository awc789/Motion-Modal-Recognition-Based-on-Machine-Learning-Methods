import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.decomposition import PCA
import glob

if __name__ == '__main__':
    # file location
    file = './walk_data/heavy_light/test/test_light_03.txt'

    # path = './walk_data/heavy_light/heavy/'
    # files = glob.glob(path + '*.txt')

    feature = []
    # get data
    # for file in files:
    measurements = np.array(KalmanFilter.get_data(file))
    x_k, y_k, z_k = KalmanFilter.kalman_filter_1(measurements)  # data after kalman filter
    waves = KalmanFilter.walk_wave(x_k, y_k, z_k)

    for i in range(len(waves)):
        p1 = waves[i][0]
        p2 = waves[i][1]
        p3 = waves[i][2]

        x = x_k[p1:p3]
        y = y_k[p1:p3]
        z = z_k[p1:p3]

        p_new = np.linspace(p1, p3, 100)

        tck_x = interpolate.splrep(range(p1, p3), x, s=0)
        tck_y = interpolate.splrep(range(p1, p3), y, s=0)
        tck_z = interpolate.splrep(range(p1, p3), z, s=0)

        x_new = np.array(interpolate.splev(p_new, tck_x, der=0))
        y_new = np.array(interpolate.splev(p_new, tck_y, der=0))
        z_new = np.array(interpolate.splev(p_new, tck_z, der=0))

        # combine data
        data = np.array([x_new, y_new, z_new]).T
        # PCA
        modle = PCA(n_components=1, copy=True, whiten=False)
        data_new = np.array(modle.fit_transform(data)).flatten()

        feature.append(data_new)

    np.savetxt("./walk_data/heavy_light/test.txt", feature, fmt="%f", delimiter=',')

    # plt.figure(1)
    # times = range(measurements.shape[0])
    # # plt.plot(times, measurements[:, 0], 'bo', times, x_k, 'b--', )
    # # plt.plot(times, measurements[:, 1], 'ro', times, y_k, 'r--', )
    # # plt.plot(times, measurements[:, 2], 'go', times, z_k, 'g--', )
    # plt.plot(times, x_k, 'b--', times, y_k, 'r--', times, z_k, 'g--')
    # plt.show()

