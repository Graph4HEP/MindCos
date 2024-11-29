import os, sys, time
import glob
import uproot
import numpy as np
import scipy
import uproot
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.sparse import save_npz, load_npz

class Detector():
    """
    Simulate a 2D rectangular straw setup.
    Made of successive modules, each one consisting of U/V double layers of straws.
    """
    n_modules=8 ## number of modules, each with 4 layers of straws
    n_layersPerModule=4
    n_layers = n_modules*n_layersPerModule
    n_straws=32
    
    straw_radius=0.25 # [cm]
    straw_distance=1.2 # distance between neighboring straw centers in same layer [cm]
    UV_distance=1.7   # distance between successive U/V submodules [cm]
    module_distance=7   # distance between successive modules [cm]
    yOffset_PerModule = 5 # y-offset per module, to allow for inward bend of tracks

    module_length = (n_layersPerModule/2)*straw_distance + UV_distance + 2*straw_radius #2*1.2+1.7+2*0.25=4.6cm
    length=n_modules*module_length + (n_modules-1)*module_distance #8*4.6+7*7=85.8cm
    width=n_straws*straw_distance+2*straw_radius #32*1.2+2*0.25=38.9
    
    ## Occupancy characteristics
    avg_tracks=2 ## avg number of tracks per event
    max_tracks=3 ## max number of tracks allowed per event

    
    layers=np.zeros(n_layers)
    straws=np.zeros(n_straws)
    x=np.zeros(n_layers)
    y=np.zeros((n_layers, n_straws))
    
    def make_straws(self):
        ## make straws, based on their locations as found in:
        ## https://cdcvs.fnal.gov/redmine/projects/gm2dqm/repository/revisions/develop/entry/node/tracker/public/tracker/scripts/trackerTracks.js
        ## returns the x,y (length,width) vectors of straw locations
        UView = {
            'x': [-12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -12.696, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, 121.6674, 121.6677, 121.6679, 121.6671, 121.6673, 121.6677, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 121.667, 126.8635, 126.8638, 126.863, 126.8632, 126.8634, 126.8637, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 126.863, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 256.021, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 261.217, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 390.375, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 395.571, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 524.725, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 529.921, 659.065,  659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 659.065, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 664.261, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 793.407, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 798.603, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 927.737, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933, 932.933],
            'y': [-64.3105, -58.2588, -52.207, -46.1552, -40.1034, -34.0517, -27.9999, -21.9481, -15.8963, -9.84457, -3.7928, 2.25898, 8.31075, 14.3625, 20.4143, 26.4661, 32.5178, 38.5696, 44.6214, 50.6732, 56.7249, 62.7767, 68.8285, 74.8803, 80.932, 86.9838, 93.0356, 99.0874, 105.139, 111.191, 117.243, 123.294, -61.2846, -55.2329, -49.1811, -43.1293, -37.0775, -31.0258, -24.974, -18.9222, -12.8704, -6.81867, -0.766897, 5.28488, 11.3367, 17.3884, 23.4402, 29.492, 35.5437, 41.5955, 47.6473, 53.6991, 59.7508, 65.8026, 71.8544, 77.9062, 83.9579, 90.0097, 96.0615, 102.113, 108.165, 114.217, 120.269, 126.32, -34.5014, -28.4497, -22.3979, -16.3461, -10.2943, -4.24257, 1.80921, 7.86098, 13.9128, 19.9645, 26.0163, 32.0681, 38.1198, 44.1716, 50.2234, 56.2752, 62.3269, 68.3787, 74.4305, 80.4823, 86.534, 92.5858, 98.6376, 104.689, 110.741, 116.793, 122.845, 128.896, 134.948, 141, 147.052, 153.104, -31.4755, -25.4238, -19.372, -13.3202, -7.26844, -1.21667, 4.83511, 10.8869, 16.9387, 22.9904, 29.0422, 35.094, 41.1457, 47.1975, 53.2493, 59.3011, 65.3528, 71.4046, 77.4564, 83.5082, 89.5599, 95.6117, 101.663, 107.715, 113.767, 119.819, 125.871, 131.922, 137.974, 144.026, 150.078, 156.129, -7.35165, -1.29987, 4.7519, 10.8037, 16.8554, 22.9072, 28.959, 35.0108, 41.0625, 47.1143, 53.1661, 59.2179, 65.2696, 71.3214, 77.3732, 83.425, 89.4767, 95.5285, 101.58, 107.632, 113.684, 119.736, 125.787, 131.839, 137.891, 143.943, 149.994, 156.046, 162.098, 168.15, 174.202, 180.253, -4.32575, 1.72603, 7.7778, 13.8296, 19.8813, 25.9331, 31.9849, 38.0367, 44.0884, 50.1402, 56.192, 62.2438, 68.2955, 74.3473, 80.3991, 86.4509, 92.5026, 98.5544, 104.606, 110.658, 116.71, 122.762, 128.813, 134.865, 140.917, 146.969, 153.02, 159.072, 165.124, 171.176, 177.227, 183.279, 17.1099, 23.1617, 29.2135, 35.2652, 41.317, 47.3688, 53.4206, 59.4723, 65.5241, 71.5759, 77.6277, 83.6794, 89.7312, 95.783, 101.835, 107.887, 113.938, 119.99, 126.042, 132.094, 138.145, 144.197, 150.249, 156.301, 162.352, 168.404, 174.456, 180.508, 186.56, 192.611, 198.663, 204.715, 20.1358, 26.1876, 32.2394, 38.2911, 44.3429, 50.3947, 56.4465, 62.4982, 68.55, 74.6018, 80.6536, 86.7053, 92.7571, 98.8089, 104.861, 110.912, 116.964, 123.016, 129.068, 135.12, 141.171, 147.223, 153.275, 159.327, 165.378, 171.43, 177.482, 183.534, 189.585, 195.637, 201.689, 207.741, 38.8201, 44.8719, 50.9237, 56.9754, 63.0272, 69.079, 75.1308, 81.1825, 87.2343, 93.2861, 99.3379, 105.39, 111.441, 117.493, 123.545, 129.597, 135.649, 141.7, 147.752, 153.804, 159.856, 165.907, 171.959, 178.011, 184.063, 190.114, 196.166, 202.218, 208.27, 214.322, 220.373, 226.425, 41.846, 47.8978, 53.9496, 60.0013, 66.0531, 72.1049, 78.1567, 84.2084, 90.2602, 96.312, 102.364, 108.416, 114.467, 120.519, 126.571, 132.623, 138.674, 144.726, 150.778, 156.83, 162.881, 168.933, 174.985, 181.037, 187.089, 193.14, 199.192, 205.244, 211.296, 217.347, 223.399, 229.451, 57.9122, 63.964, 70.0158, 76.0675, 82.1193, 88.1711, 94.2229, 100.275, 106.326, 112.378, 118.43, 124.482, 130.534, 136.585, 142.637, 148.689, 154.741, 160.792, 166.844, 172.896, 178.948, 184.999, 191.051, 197.103, 203.155, 209.207, 215.258, 221.31, 227.362, 233.414, 239.465, 245.517, 60.9381, 66.9899, 73.0417, 79.0934, 85.1452, 91.197, 97.2488, 103.301, 109.352, 115.404, 121.456, 127.508, 133.559, 139.611, 145.663, 151.715, 157.767, 163.818, 169.87, 175.922, 181.974, 188.025, 194.077, 200.129, 206.181, 212.232, 218.284, 224.336, 230.388, 236.44, 242.491, 248.543, 74.3793, 80.4311, 86.4828, 92.5346, 98.5864, 104.638, 110.69, 116.742, 122.793, 128.845, 134.897, 140.949, 147.001, 153.052, 159.104, 165.156, 171.208, 177.259, 183.311, 189.363, 195.415, 201.467, 207.518, 213.57, 219.622, 225.674, 231.725, 237.777, 243.829, 249.881, 255.932, 261.984, 77.4052, 83.457, 89.5087, 95.5605, 101.612, 107.664, 113.716, 119.768, 125.819, 131.871, 137.923, 143.975, 150.026, 156.078, 162.13, 168.182, 174.234, 180.285, 186.337, 192.389, 198.441, 204.492, 210.544, 216.596, 222.648, 228.7, 234.751, 240.803, 246.855, 252.907, 258.958, 265.01, 88.2379, 94.2896, 100.341, 106.393, 112.445, 118.497, 124.549, 130.6, 136.652, 142.704, 148.756, 154.807, 160.859, 166.911, 172.963, 179.014, 185.066, 191.118, 197.17, 203.222, 209.273, 215.325, 221.377, 227.429, 233.48, 239.532, 245.584, 251.636, 257.688, 263.739, 269.791, 275.843, 91.2638, 97.3155, 103.367, 109.419, 115.471, 121.523, 127.574, 133.626, 139.678, 145.73, 151.782, 157.833, 163.885, 169.937, 175.989, 182.04, 188.092, 194.144, 200.196, 206.247, 212.299, 218.351, 224.403, 230.455, 236.506, 242.558, 248.61, 254.662, 260.713, 266.765, 272.817, 278.869]
        }
        VView = {
            'x': [7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 12.696, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 141.863, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 147.059, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 276.217, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 281.413, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 410.571, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 415.767, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 544.921, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 550.117, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 679.261, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 684.457, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 813.603, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 818.799, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 947.933, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129, 953.129],
            'y': [-60.9556, -54.9038, -48.852, -42.8002, -36.7485, -30.6967, -24.6449, -18.5931, -12.5414, -6.4896, -0.437828, 5.61395, 11.6657, 17.7175, 23.7693, 29.821, 35.8728, 41.9246, 47.9764, 54.0281, 60.0799, 66.1317, 72.1835, 78.2352, 84.287, 90.3388, 96.3906, 102.442, 108.494, 114.546, 120.598, 126.649, -63.9815, -57.9297, -51.8779, -45.8261, -39.7744, -33.7226, -27.6708, -21.619, -15.5673, -9.5155, -3.46373, 2.58805, 8.63982, 14.6916, 20.7434, 26.7951, 32.8469, 38.8987, 44.9505, 51.0022, 57.054, 63.1058, 69.1576, 75.2093, 81.2611, 87.3129, 93.3647, 99.4164, 105.468, 111.52, 117.572, 123.624, -31.1465, -25.0947, -19.0429, -12.9911, -6.93937, -0.887599, 5.16417, 11.2159, 17.2677, 23.3195, 29.3713, 35.423, 41.4748, 47.5266, 53.5784, 59.6301, 65.6819, 71.7337, 77.7855, 83.8372, 89.889, 95.9408, 101.993, 108.044, 114.096, 120.148, 126.2, 132.251, 138.303, 144.355, 150.407, 156.459, -34.1724, -28.1206, -22.0688, -16.017, -9.96527, -3.9135, 2.13827, 8.19005, 14.2418, 20.2936, 26.3454, 32.3971, 38.4489, 44.5007, 50.5525, 56.6042, 62.656, 68.7078, 74.7596, 80.8113, 86.8631, 92.9149, 98.9667, 105.018, 111.07, 117.122, 123.174, 129.226, 135.277, 141.329, 147.381, 153.433, -3.99668, 2.05509, 8.10687, 14.1586, 20.2104, 26.2622, 32.314, 38.3657, 44.4175, 50.4693, 56.5211, 62.5728, 68.6246, 74.6764, 80.7282, 86.7799, 92.8317, 98.8835, 104.935, 110.987, 117.039, 123.091, 129.142, 135.194, 141.246, 147.298, 153.349, 159.401, 165.453, 171.505, 177.557, 183.608, -7.02258, -0.970805, 5.08097, 11.1327, 17.1845, 23.2363, 29.2881, 35.3398, 41.3916, 47.4434, 53.4952, 59.5469, 65.5987, 71.6505, 77.7023, 83.754, 89.8058, 95.8576, 101.909, 107.961, 114.013, 120.065, 126.116, 132.168, 138.22, 144.272, 150.324, 156.375, 162.427, 168.479, 174.531, 180.582, 20.4649, 26.5167, 32.5684, 38.6202, 44.672, 50.7238, 56.7755, 62.8273, 68.8791, 74.9309, 80.9826, 87.0344, 93.0862, 99.138, 105.19, 111.242, 117.293, 123.345, 129.397, 135.449, 141.5, 147.552, 153.604, 159.656, 165.707, 171.759, 177.811, 183.863, 189.915, 195.966, 202.018, 208.07, 17.439, 23.4908, 29.5425, 35.5943, 41.6461, 47.6979, 53.7496, 59.8014, 65.8532, 71.905, 77.9567, 84.0085, 90.0603, 96.1121, 102.164, 108.216, 114.267, 120.319, 126.371, 132.423, 138.474, 144.526, 150.578, 156.63, 162.682, 168.733, 174.785, 180.837, 186.889, 192.94, 198.992, 205.044, 42.1751, 48.2269, 54.2786, 60.3304, 66.3822, 72.434, 78.4857, 84.5375, 90.5893, 96.6411, 102.693, 108.745, 114.796, 120.848, 126.9, 132.952, 139.003, 145.055, 151.107, 157.159, 163.211, 169.262, 175.314, 181.366, 187.418, 193.469, 199.521, 205.573, 211.625, 217.677, 223.728, 229.78, 39.1492, 45.201, 51.2527, 57.3045, 63.3563, 69.4081, 75.4598, 81.5116, 87.5634, 93.6152, 99.6669, 105.719, 111.77, 117.822, 123.874, 129.926, 135.978, 142.029, 148.081, 154.133, 160.185, 166.236, 172.288, 178.34, 184.392, 190.444, 196.495, 202.547, 208.599, 214.651, 220.702, 226.754, 61.2672, 67.319, 73.3707, 79.4225, 85.4743, 91.5261, 97.5778, 103.63, 109.681, 115.733, 121.785, 127.837, 133.888, 139.94, 145.992, 152.044, 158.096, 164.147, 170.199, 176.251, 182.303, 188.354, 194.406, 200.458, 206.51, 212.562, 218.613, 224.665, 230.717, 236.769, 242.82, 248.872, 58.2413, 64.2931, 70.3448, 76.3966, 82.4484, 88.5002, 94.5519, 100.604, 106.655, 112.707, 118.759, 124.811, 130.863, 136.914, 142.966, 149.018, 155.07, 161.121, 167.173, 173.225, 179.277, 185.329, 191.38, 197.432, 203.484, 209.536, 215.587, 221.639, 227.691, 233.743, 239.795, 245.846, 77.7342, 83.786, 89.8378, 95.8896, 101.941, 107.993, 114.045, 120.097, 126.148, 132.2, 138.252, 144.304, 150.356, 156.407, 162.459, 168.511, 174.563, 180.614, 186.666, 192.718, 198.77, 204.821, 210.873, 216.925, 222.977, 229.029, 235.08, 241.132, 247.184, 253.236, 259.287, 265.339, 74.7083, 80.7601, 86.8119, 92.8637, 98.9154, 104.967, 111.019, 117.071, 123.123, 129.174, 135.226, 141.278, 147.33, 153.381, 159.433, 165.485, 171.537, 177.589, 183.64, 189.692, 195.744, 201.796, 207.847, 213.899, 219.951, 226.003, 232.054, 238.106, 244.158, 250.21, 256.262, 262.31, 91.5928, 97.6446, 103.696, 109.748, 115.8, 121.852, 127.903, 133.955, 140.007, 146.059, 152.111, 158.162, 164.214, 170.266, 176.318, 182.369, 188.421, 194.473, 200.525, 206.577, 212.628, 218.68, 224.732, 230.784, 236.835, 242.887, 248.939, 254.991, 261.043, 267.094, 273.146, 279.198, 88.5669, 94.6187, 100.67, 106.722, 112.774, 118.826, 124.878, 130.929, 136.981, 143.033, 149.085, 155.136, 161.188, 167.24, 173.292, 179.344, 185.395, 191.447, 197.499, 203.551, 209.602, 215.654, 221.706, 227.758, 233.81, 239.861, 245.913, 251.965, 258.017, 264.068, 270.12, 276.172],
        }
        ## above are in mm, turn to cm
        for dim in ['x','y']:
            for i in range(len(UView[dim])):
                UView[dim][i]/=10
                VView[dim][i]/=10
        ## shift vertically to 0, to avoid negative numbers
        miny = min(UView['y'])
        for i in range(len(UView['y'])):
            UView['y'][i]-=miny
            VView['y'][i]-=miny
        
        for i_module in range(self.n_modules):
            for i_layerU in range(2):        
                layer = i_module*self.n_layersPerModule + i_layerU
                self.layers[layer]=layer
                i_straw0 = self.n_straws*(2*i_module+i_layerU)
                layer_x = UView['x'][i_straw0]
                self.x[layer] = layer_x
                for i_straw in range(self.n_straws):
                    self.y[layer][i_straw] = UView['y'][i_straw0+i_straw]
            for i_layerV in range(2):        
                layer = i_module*self.n_layersPerModule + 2+i_layerV
                self.layers[layer]=layer
                i_straw0 = self.n_straws*(2*i_module+i_layerV)
                layer_x = VView['x'][i_straw0]
                self.x[layer] = layer_x
                for i_straw in range(self.n_straws):
                    self.y[layer][i_straw] = VView['y'][i_straw0+i_straw]

        self.length = VView['x'][-1] - UView['x'][0]
        self.width  = max(VView['y']) - min(VView['y'])
    
    def make_straws_guesst(self):
        ## make straws, based on a guesstimate of their realtive distances.
        ## returns the x,y (length,width) vectors of straw locations
        for i_module in range(self.n_modules):
            for i_layer in range(self.n_layersPerModule):
                layer = i_module*self.n_layersPerModule + i_layer
                self.layers[layer]=layer
                layer_x = i_module*self.module_length + self.straw_radius
                if i_module>0: layer_x += i_module*self.module_distance
                if i_layer>=1: layer_x += self.straw_distance
                if i_layer>=2: layer_x += self.UV_distance
                if i_layer==3: layer_x += self.straw_distance
                self.x[layer] = layer_x
                ## vertical stagger for this layer:
                if (i_layer==1) or (i_layer==2):  stagger = self.straw_distance/2
                else: stagger=0
                for i_straw in range(self.n_straws):
                    #if (i_module<=2) and (i_straw>=16): continue
                    #if (i_module<=4) and (i_straw>=24): continue
                    if i_module==8: self.straws[i_straw]=i_straw
                    self.y[layer][i_straw]=i_module*self.yOffset_PerModule + i_straw*self.straw_distance + stagger

def build_tracker():
    ### Build the detector
    tracker=Detector()
    tracker.make_straws()
    return tracker

def process_bar(num, total, dt):
    import psutil
    rate = float(num)/total
    ratenum = int(50*rate)
    estimate = dt/rate*(1-rate)
    ava = psutil.virtual_memory().available/1024/1024/1024
    r = '\r[{}{}]{}/{} - used {:.1f}s / left {:.1f}s / total {:.1f}s / ava {:.1f}G'.format(f'*'*ratenum,' '*(50-ratenum), num, total, dt, estimate, dt/rate, ava)
    sys.stdout.write(r)
    sys.stdout.flush()

def build_sparse(raw_np, mode='coo'):
    total_n = raw_np.shape[0]
    if(mode=='coo'):
        sp_np = coo_matrix(raw_np.reshape(total_n, -1))
        return sp_np
    if(mode=='csr'):
        sp_np = csr_matrix(raw_np.reshape(total_n, -1))
        return sp_np
    if(mode=='csc'):
        sp_np = csc_matrix(raw_np.reshape(total_n, -1))
        return sp_np

def fillArrays(tree):
    st = time.time()
    total_n = int(len(tree[tree.keys()[0]].array(library='np'))/2)
    evt_hits = np.zeros((total_n,32,32))
    evt_ids  = np.zeros((total_n,32,32))
    evt_wid  = np.zeros((total_n,32,32))
    layer = tree['layer'].array(library='np')
    straw = tree['straw'].array(library='np')
    caltime  = tree['time'].array(library='np')
    width = tree['width'].array(library='np')
    try:
        track = tree['track_id'].array(library='np')
    except:
        pass
    ## Fill the 32x32 hits and ids arrays for this event
    for k in range(total_n):
        for i, (l, s) in enumerate(zip(layer[k].astype(int), straw[k].astype(int))):
            evt_hits[k, l, s] = caltime[k][i]
            evt_wid[k, l, s]  = width[k][i]
            try:
                evt_ids[k, l, s] = track[k][i]
            except:
                pass
        #if(k%1000==0):
        #    process_bar(k+1, total_n, time.time()-st)
    #print()
    return evt_hits, evt_wid, evt_ids

def read_single_root(rootname):
    tree = uproot.open(f'{rootname}:GnnData/time_island_all')
    tree_trk = uproot.open(f'{rootname}:GnnData/time_island')
    st = time.time()
    evts_hits, evt_wid, _ = fillArrays(tree)
    _, _, evt_ids = fillArrays(tree_trk)
    return evts_hits, evt_wid, evt_ids

def fully_connect(node_set):
    N = len(node_set)
    u = []
    v = []
    for i in range(N):
        for j in range(N):
            if i != j:  # 避免自身与自身的配对  
                u.append(node_set[i])
                v.append(node_set[j])
    return u,v

def fully_connect_N(N):
    nodes = np.arange(N)
    u = np.repeat(nodes[:, np.newaxis], N, axis=1).flatten()
    v = np.repeat(nodes[np.newaxis, :], N, axis=0).flatten()
    index = np.where(u == v)[0]
    u = np.delete(u, index)
    v = np.delete(v, index)
    return u, v

def fake_connect(node_set):
    count=0
    for i in range(len(node_set)):
        for j in range(len(node_set)):
            if(j!=i):
                for k in range(len(node_set[i])):
                    for l in range(len(node_set[j])):
                        count+=1
    u = np.zeros(count)
    v = np.zeros(count)
    count=0
    for i in range(len(node_set)):
        for j in range(len(node_set)):
            if(j!=i):
                for k in range(len(node_set[i])):
                    for l in range(len(node_set[j])):
                        u[count] = node_set[i][k]
                        v[count] = node_set[j][l]
                        count+=1
    return u,v


#root to np
def root_to_np(root_name, mode='coo'):
    begin_time = time.time()
    for rootname in glob.glob(f'{root_name}.root'):
        name_hits = f'{root_name}_{mode}_hits.npz'
        name_ids = f'{root_name}_{mode}_ids.npz'
        name_width = f'{root_name}_{mode}_width.npz'
        if(os.path.exists(name_hits)==False or os.path.exists(name_ids)==False or os.path.exists(name_width)==False):
            Hits, Width, IDs = read_single_root(rootname)
            Hits = build_sparse(Hits, mode=mode)
            Width= build_sparse(Width, mode=mode)
            IDs  = build_sparse(IDs, mode=mode)
            save_npz(f'{name_hits}', Hits)
            save_npz(f'{name_width}', Width)
            save_npz(f'{name_ids}', IDs)
        #else:
            #print('\npt file already exists!')
    #print(f'\nroot to pt time used: {time.time()-begin_time:.1f}s')

def fill_info(root_name):
    tree = uproot.open(f'./{root_name}:GnnData/time_island_all')
    fill  = tree['fill'].array(library='np').astype(np.int16).reshape(-1,1)
    run   = tree['runNum'].array(library='np').astype(np.int16).reshape(-1,1)
    subrun= tree['subRunNum'].array(library='np').astype(np.int16).reshape(-1,1)
    evt   = tree['eventNum'].array(library='np').astype(np.int16).reshape(-1,1)
    bunch = tree['bunchNum'].array(library='np').astype(np.int16).reshape(-1,1)
    island= tree['islandNumber'].array(library='np').astype(np.int16).reshape(-1,1)
    graph_feat = np.concatenate((fill, run, subrun, evt, bunch, island), axis=1)
    return graph_feat.reshape(graph_feat.shape[0], 1, 6)

def build_edges(Hits, IDs, Ndata=10000):
    if(os.path.exists(f'./data/src_{Ndata}.npz') and os.path.exists(f'./data/dst_{Ndata}.npz') and os.path.exists(f'./data/edge_label_{Ndata}.npz')):
        src = np.load(f'./data/src_{Ndata}.npz', allow_pickle=True)['arr_0']
        dst = np.load(f'./data/dst_{Ndata}.npz', allow_pickle=True)['arr_0']
        edge_label = np.load(f'./data/edge_label_{Ndata}.npz', allow_pickle=True)['arr_0']
        return src, dst, edge_label
    st=time.time()
    src = []
    dst = []
    u_pos_list = []
    v_pos_list = []
    edge_label = []
    count = 0
    for i, (hit, ID) in enumerate(zip(Hits, IDs)):
        layer, straw = np.where(hit!=0)
        Nnode = len(layer)
        if(Nnode<6):
            continue
        id_feat = ID[layer, straw]
        if(id_feat.max()>=3):
            continue
        u_pos = []
        v_pos = []
        node_set = []
        for k in range(3):
            link = np.where(id_feat==k)[0]
            node_set.append(link.tolist())
            if(len(link)>0 and k>0):
                u_temp, v_temp = fully_connect(link)
                u_pos += u_temp
                v_pos += v_temp
        u,v = fully_connect_N(Nnode)
        src.append(u)
        dst.append(v)
        u_pos_list.append(u_pos)
        v_pos_list.append(v_pos)
        count+=1
        #if(count%1000==0):
            #process_bar(count+1, Ndata*1.4+1, time.time()-st)
        if(count>Ndata*1.4):
            break
    count = 0
    #print()
    st=time.time()
    for i in range(len(src)):
        u = np.array(src[i]).reshape(-1,1)
        v = np.array(dst[i]).reshape(-1,1)
        uv = np.concatenate((u,v), axis=1)
        u_pos = np.array(u_pos_list[i]).reshape(-1,1)
        v_pos = np.array(v_pos_list[i]).reshape(-1,1)
        uv_pos = np.concatenate((u_pos, v_pos), axis=1)
        elabel = np.zeros((len(u), 1))
        uv_pos_uni = np.unique(uv_pos, axis=0)
        indices = []  
        for elem in uv_pos_uni:
            matches = np.where(np.all(uv == elem, axis=1))
            indices.extend(matches[0])  
        elabel[indices] = 1
        edge_label.append(elabel)
        count+=1
        #if(count%1000==0):
            #process_bar(count+1, Ndata*1.4+1, time.time()-st)
        if(count>Ndata*1.4):
            break                    
    #print()
    np.savez(f'./data/src_{Ndata}.npz', np.array(src, dtype=object))
    np.savez(f'./data/dst_{Ndata}.npz', np.array(dst, dtype=object))
    np.savez(f'./data/edge_label_{Ndata}.npz', np.array(edge_label, dtype=object))
    return np.array(src, dtype=object), np.array(dst, dtype=object), np.array(edge_label, dtype=object)

def build_nodes(Hits, IDs, Width, root_name, Ndata):
    if(os.path.exists(f'./data/node_feat_{Ndata}.npz') and os.path.exists(f'./data/node_match_{Ndata}.npz') and os.path.exists(f'./data/node_track_{Ndata}.npz')):
        node_feat = np.load(f'./data/node_feat_{Ndata}.npz', allow_pickle=True)['arr_0']
        node_match = np.load(f'./data/node_match_{Ndata}.npz', allow_pickle=True)['arr_0']
        node_track = np.load(f'./data/node_track_{Ndata}.npz', allow_pickle=True)['arr_0']
        return node_feat, node_match, node_track
    node_feat  = []
    node_track = []
    node_match = []
    count=0
    graph_feat = fill_info(f'{root_name}.root')
    tracker = build_tracker()
    st=time.time()
    for i, (hit, ID, wid) in enumerate(zip(Hits, IDs, Width)):
        layer, straw = np.where(hit!=0)
        Nnode = len(layer)
        if(Nnode<6):
            continue
        id_feat = ID[layer, straw]
        if(id_feat.max()>=3):
            continue        
        l_feat = layer.reshape(-1,1)
        s_feat = straw.reshape(-1,1)
        x_feat = tracker.x[layer].reshape(-1,1)
        y_feat = tracker.y[layer,straw].reshape(-1,1)
        t_feat = (hit[layer,straw]-hit.sum()/len(layer)).reshape(-1,1)
        w_feat = wid[layer,straw].reshape(-1,1)
        id_feat_np = ID[layer,straw].reshape(-1,1)
        node_feat.append(np.concatenate((l_feat, s_feat, x_feat, y_feat, t_feat, w_feat), axis=1).astype(np.float32))
        node_match.append(graph_feat[i])
        node_track.append(id_feat_np)
        count+=1
        #if(count%1000==0):
            #process_bar(count+1, Ndata*1.4+1, time.time()-st)
        if(count>Ndata*1.4):
            break    
    #print()
    np.savez(f'./data/node_feat_{Ndata}.npz', np.array(node_feat, dtype=object))
    np.savez(f'./data/node_match_{Ndata}.npz', np.array(node_match, dtype=object))
    np.savez(f'./data/node_track_{Ndata}.npz', np.array(node_track, dtype=object))
    return np.array(node_feat, dtype=object), np.array(node_match, dtype=object), np.array(node_track, dtype=object)

def build_edges_feat(node_feat, src, dst):
    if(os.path.exists(f'./data/edge_feat_{(len(src)-1)/1.4:.0f}.npz')):
        edge_feat = np.load(f'./data/edge_feat_{(len(src)-1)/1.4:.0f}.npz', allow_pickle=True)['arr_0']
        return edge_feat
    count=0
    st=time.time()
    edge_feat  = []
    for i in range(len(src)):
        u = src[i]
        v = dst[i]
        x_feat = node_feat[i][:,2].reshape(-1,1)
        y_feat = node_feat[i][:,3].reshape(-1,1)
        l_feat = node_feat[i][:,0].reshape(-1,1)
        s_feat = node_feat[i][:,1].reshape(-1,1)
        t_feat = node_feat[i][:,4].reshape(-1,1)
        w_feat = node_feat[i][:,5].reshape(-1,1)
        dDX = np.abs(x_feat[u] - x_feat[v])
        dDY = np.abs(y_feat[u] - y_feat[v])
        dDL = np.abs(l_feat[u] - l_feat[v])
        dDS = np.abs(s_feat[u] - s_feat[v])
        dDT = np.abs(t_feat[u] - t_feat[v])
        dDW = np.abs(w_feat[u] - w_feat[v])
        dcos = dDX/np.sqrt(dDX**2+dDY**2)
        dsin = dDY/np.sqrt(dDX**2+dDY**2)
        edge_feat.append(np.concatenate((dDL,dDS,dDX,dDY,dDT,dDW,dcos,dsin), axis=1).astype(np.float32))
        count+=1
        #if(count%1000==0):
            #process_bar(count+1, len(src), time.time()-st)
    #print()
    np.savez(f'./data/edge_feat_{(len(src)-1)/1.4:.0f}.npz', np.array(edge_feat, dtype=object))
    return np.array(edge_feat, dtype=object)

def root_to_data(root_name, mode, Ndata):
    root_to_np(root_name=root_name, mode=mode)
    IDs = load_npz(f'{root_name}_{mode}_ids.npz').toarray()
    IDs = IDs.reshape(-1,32,32)
    Hits = load_npz(f'{root_name}_{mode}_hits.npz').toarray()
    Hits = Hits.reshape(-1,32,32)
    Width = load_npz(f'{root_name}_{mode}_width.npz').toarray()
    Width = Width.reshape(-1,32,32)
    src, dst, edge_label = build_edges(Hits, IDs, Ndata)
    node_feat, node_match, node_track = build_nodes(Hits, IDs, Width, root_name, Ndata)
    edge_feat = build_edges_feat(node_feat, src, dst)
    return src, dst, node_feat, edge_feat, node_match, node_track, edge_label

if __name__=='__main__':
    root_name = '30996_n2000'
    mode='coo'
    Ndata = 50000
    root_to_data(root_name, mode, Ndata)


