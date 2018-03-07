import numpy as np

"""

---Notes on Network data---

Each RGB colour has training inputs manually inputed into an array:

The data input process is entirelly subjective, for example, the data given tells the network that white on bright yellow looks bad. This is simply an opinion.

The code is written in such that any number of data inputs can be given in each type. The only neccesity is consistency and range. 


"""
training_inputs_W = np.array([[0,0,0],
							  [30,24,24],
							  [61,14,14],
							  [90,46,46],
							  [105,27,27],
							  [90,9,47],
							  [98,63,79],
							  [41,25,41],
							  [113,6,117],
							  [69,6,117],
							  [32,16,19],
							  [25,16,61],
							  [13,5,46],
							  [9,0,46],
							  [50,57,73],
							  [65,65,65],
							  [46,46,46],
							  [30,89,121],
							  [7,47,71],
							  [53,110,112],
							  [51,55,55],
							  [22,118,89],
							  [4,53,38],
							  [54,101,71],
							  [11,88,39],
							  [22,40,14],
							  [35,82,11],
							  [69,88,45],
							  [55,90,10],
							  [73,80,27],
							  [78,72,40],
							  [101,84,0],
							  [82,79,76],
							  [114,52,25]])

training_inputs_G = np.array([[255,0,0],
							  [122,122,122],
							  [179,102,102],
							  [189,45,45],
							  [242,31,87],
							  [203,33,78],
							  [122,114,116],
							  [255,40,213],
							  [179,115,166],
							  [198,101,178],
							  [151,36,201],
							  [39,84,232],
							  [112,117,137],
							  [73,102,196],
							  [0,196,255],
							  [107,141,151],
							  [31,185,164],
							  [115,174,166],
							  [37,219,122],
							  [87,174,127],
							  [87,174,93],
							  [61,189,29],
							  [149,172,145],
							  [185,180,52],
							  [165,179,130],
							  [129,146,88],
							  [189,206,31],
							  [144,148,111],
							  [164,172,92],
							  [219,182,34],
							  [146,133,115],
							  [188,116,22],
							  [162,131,89],
							  [206,89,30]])

training_inputs_B = np.array([[255,255,255],
							  [233,224,224],
							  [250,161,161],
							  [198,198,198],
							  [216,167,180],
							  [206,195,202],
							  [255,134,207],
							  [255,134,244],
							  [222,195,225],
							  [170,118,230],
							  [240,226,255],
							  [132,124,181],
							  [161,166,215],
							  [84,147,240],
							  [159,164,172],
							  [97,196,232],
							  [199,241,255],
							  [189,236,230],
							  [198,208,206],
							  [0,255,171],
							  [0,255,85],
							  [158,245,187],
							  [223,255,227],
							  [205,255,0],
							  [219,255,71],
							  [232,238,210],
							  [255,255,0],
							  [255,255,117],
							  [220,180,157],
							  [236,222,222],
							  [137,255,0],
							  [219,219,186],
							  [212,212,212],
							  [239,239,141],
							  [50,68,215]])

#Creates a training set to allow for ease of access to all data
training_set = [training_inputs_B,training_inputs_G, training_inputs_W]

#Pretrained weights (trained over roughly 2 million iterations)

starting_IH_weights = [[8.91052664, 3.51338965, 2.11430035, -6.3878801],
 					  [1.27930063, -8.16669178, -5.6885955, 6.92942414],
 					  [-6.96130486, 0.80141797, -2.3676587, 5.28949432]]

starting_HO_weights = [[-1.83596863, 3.5460759, 1.17905525, -1.90147355]]




