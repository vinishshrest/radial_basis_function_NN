# %%
import numpy as np 
import matplotlib.pyplot as plt
import random 

# model building packages
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

import os # for path

# set random seed
random.seed(9745)

# gpu detection 
print(tf.config.list_physical_devices("GPU"))

# Set output path 
outpath = os.path.expanduser("~/Dropbox/USmortality/extra_tools/radial_basis_function/results")

# ground truth function
X1_truth = np.linspace(0, 12, 100)

def boundary_truth_ver00(X_vals):

    """Low complexity problem."""
    N_obs = len(X_vals)
    X2_vals = np.sin(X_vals) #+ np.random.normal(0, 0.05, len(X))
    mask1 =  (X_vals > 3) & (X_vals <= 5)  
    mask2 =  (X_vals > 0) & (X_vals <= 1)    
    index = np.zeros(N_obs)
    index[mask1] = 1 
    index2 = np.zeros(N_obs)
    index2[mask2] = 1
    X2_vals = 5 + X2_vals + np.sin(X_vals)**2 + np.cos(X_vals) * np.sin(X_vals)*index + np.random.normal(0, 0.05, len(X2_vals))

    return X2_vals

def boundary_truth_ver01(X_vals):
    """
    According to Claude:
    Complex boundary with multiple challenging features:
    - Multiple regime changes with different dynamics
    - Exponential decay regions
    - Polynomial interactions
    - Trigonometric modulations
    - Sharp transitions and discontinuities
    - Heteroscedastic noise (variance changes with X)
    """
    N_obs = len(X_vals)
    
    # Define multiple regime masks
    regime1 = (X_vals >= 0) & (X_vals < 2)      # Exponential growth region
    regime2 = (X_vals >= 2) & (X_vals < 4)      # Polynomial with trig modulation
    regime3 = (X_vals >= 4) & (X_vals < 6.5)    # Damped oscillation
    regime4 = (X_vals >= 6.5) & (X_vals < 8)    # Sharp nonlinear transition
    regime5 = (X_vals >= 8) & (X_vals <= 10)    # Chaotic-like behavior
    
    # Initialize output
    X2_vals = np.zeros(N_obs)
    
    # Regime 1: Exponential growth with sinusoidal modulation
    X2_vals[regime1] = (
        3 + 
        0.5 * np.exp(0.3 * X_vals[regime1]) + 
        0.8 * np.sin(3 * X_vals[regime1]) +
        0.3 * X_vals[regime1]**2
    )
    
    # Regime 2: Polynomial with trigonometric interaction
    X2_vals[regime2] = (
        5 + 
        -0.5 * (X_vals[regime2] - 3)**3 + 
        2 * np.cos(2 * X_vals[regime2]) * np.sin(X_vals[regime2]) +
        1.5 * np.log(X_vals[regime2] + 1)
    )
    
    # Regime 3: Damped oscillation with exponential envelope
    X2_vals[regime3] = (
        7 + 
        3 * np.exp(-0.4 * (X_vals[regime3] - 4)) * np.sin(4 * X_vals[regime3]) +
        0.2 * (X_vals[regime3] - 5)**2 +
        0.5 * np.cos(X_vals[regime3])**3
    )
    
    # Regime 4: Sharp nonlinear transition with sigmoid-like behavior
    X2_vals[regime4] = (
        6 + 
        4 / (1 + np.exp(-3 * (X_vals[regime4] - 7))) +
        np.sin(5 * X_vals[regime4]) * np.cos(2 * X_vals[regime4]) +
        0.3 * (X_vals[regime4] - 7)**3
    )
    
    # Regime 5: Complex multi-frequency oscillation
    X2_vals[regime5] = (
        10 + 
        1.5 * np.sin(2 * X_vals[regime5]) * np.cos(3 * X_vals[regime5]) +
        0.8 * np.sin(5 * X_vals[regime5]) +
        0.4 * (X_vals[regime5] - 9) * np.cos(X_vals[regime5]) +
        0.2 * (X_vals[regime5] - 8.5)**2
    )
    
    # Add heteroscedastic noise (noise variance increases with X)
    noise_scale = 0.05 + 0.02 * (X_vals / 10)
    X2_vals += np.random.normal(0, noise_scale, N_obs)
    
    # Add occasional outliers (5% of points)
    outlier_mask = np.random.random(N_obs) < 0.05
    X2_vals[outlier_mask] += np.random.normal(0, 0.5, np.sum(outlier_mask))
    
    return X2_vals


def boundary_truth_ver02(X_vals):
    """ According to Claude:
    EXTREME complexity boundary with:
    - 7 distinct regimes with sharp transitions
    - Multiple scales of oscillation (high + low frequency)
    - Discontinuous jumps
    - Local sharp features (spikes, cusps)
    - Regions with different curvature patterns
    - Extreme heteroscedastic noise
    - Clustered outliers in specific regions
    - Non-monotonic transitions between regimes
    """
    N_obs = len(X_vals)
    
    # Define 7 regime masks with overlapping transition zones
    regime1 = (X_vals >= 0) & (X_vals < 1.5)      # Rapid oscillation
    regime2 = (X_vals >= 1.5) & (X_vals < 3.5)    # Smooth polynomial with sudden drop
    regime3 = (X_vals >= 3.5) & (X_vals < 5.0)    # Chaotic multi-frequency
    regime4 = (X_vals >= 5.0) & (X_vals < 7.0)    # Exponential with sharp spike
    regime5 = (X_vals >= 7.0) & (X_vals < 9.0)    # Damped with discontinuity
    regime6 = (X_vals >= 9.0) & (X_vals < 10.5)   # Cusp and sharp corner
    regime7 = (X_vals >= 10.5) & (X_vals <= 12)   # Extreme oscillation decay
    
    X2_vals = np.zeros(N_obs)
    
    # Regime 1: High-frequency oscillation with exponential growth
    X2_vals[regime1] = (
        2 + 
        1.5 * np.exp(0.5 * X_vals[regime1]) +
        2 * np.sin(8 * X_vals[regime1]) * np.cos(5 * X_vals[regime1]) +
        0.5 * np.sin(15 * X_vals[regime1])  # Very high frequency
    )
    
    # Regime 2: Smooth polynomial with SUDDEN DROP at boundary
    X2_vals[regime2] = (
        6 + 
        -1.2 * (X_vals[regime2] - 2.5)**3 + 
        0.8 * (X_vals[regime2] - 2)**2 +
        1.5 * np.sin(3 * X_vals[regime2]) +
        0.8 * np.cos(X_vals[regime2])**2
    )
    # Add discontinuous jump near end of regime
    jump_mask = (X_vals >= 3.2) & (X_vals < 3.5)
    X2_vals[jump_mask] -= 2.5
    
    # Regime 3: Chaotic-looking multi-scale oscillation
    X2_vals[regime3] = (
        5 + 
        2.5 * np.sin(2 * X_vals[regime3]) * np.cos(7 * X_vals[regime3]) +
        1.2 * np.sin(11 * X_vals[regime3]) +
        0.8 * np.cos(4 * X_vals[regime3]) * np.sin(3 * X_vals[regime3]) +
        0.5 * (X_vals[regime3] - 4.25)**2 * np.sin(6 * X_vals[regime3])
    )
    
    # Regime 4: Exponential decay with SHARP SPIKE in middle
    base_vals = (
        8 + 
        3 * np.exp(-0.6 * (X_vals[regime4] - 5)) +
        1.5 * np.sin(4 * X_vals[regime4]) +
        0.3 * (X_vals[regime4] - 6)**2
    )
    # Add sharp spike (like a delta function)
    spike_mask = (X_vals >= 5.8) & (X_vals < 6.2)
    spike_vals = 5 * np.exp(-50 * (X_vals[spike_mask] - 6)**2)
    base_vals[spike_mask[regime4]] += spike_vals
    X2_vals[regime4] = base_vals
    
    # Regime 5: Damped oscillation with DISCONTINUITY in middle
    X2_vals[regime5] = (
        9 + 
        4 * np.exp(-0.3 * (X_vals[regime5] - 7)) * np.sin(5 * X_vals[regime5]) +
        0.8 * np.cos(3 * X_vals[regime5])**3 +
        0.2 * (X_vals[regime5] - 8)**3
    )
    # Add step discontinuity
    step_mask = X_vals >= 8.0
    X2_vals[regime5 & step_mask] += 2.0
    
    # Regime 6: Sharp cusp and corner features
    X2_vals[regime6] = (
        11 + 
        -2 * np.abs(X_vals[regime6] - 9.75)**1.5 +  # V-shaped cusp
        3 / (1 + np.exp(-5 * (X_vals[regime6] - 9.5))) +  # Sharp sigmoid
        1.5 * np.sin(6 * X_vals[regime6]) * np.cos(2 * X_vals[regime6])
    )
    # Add sharp corner
    corner_mask = (X_vals >= 10.0) & (X_vals < 10.5)
    X2_vals[corner_mask] += 1.5 * np.maximum(0, X_vals[corner_mask] - 10.2)
    
    # Regime 7: Extreme multi-frequency with decay envelope
    X2_vals[regime7] = (
        13 + 
        5 * np.exp(-0.8 * (X_vals[regime7] - 10.5)) * (
            np.sin(3 * X_vals[regime7]) * np.cos(7 * X_vals[regime7]) +
            0.8 * np.sin(12 * X_vals[regime7])
        ) +
        0.5 * (X_vals[regime7] - 11)**2 +
        1.2 * np.cos(X_vals[regime7])**4
    )
    
    # EXTREME heteroscedastic noise (variance varies by regime)
    noise_scale = np.zeros(N_obs)
    noise_scale[regime1] = 0.15  # High noise in oscillatory region
    noise_scale[regime2] = 0.08
    noise_scale[regime3] = 0.20  # Very high noise in chaotic region
    noise_scale[regime4] = 0.06
    noise_scale[regime5] = 0.10
    noise_scale[regime6] = 0.05  # Low noise to preserve sharp features
    noise_scale[regime7] = 0.12
    
    X2_vals += np.random.normal(0, noise_scale, N_obs)
    
    # Add CLUSTERED outliers in specific problematic regions
    # Region 1: Near the discontinuity
    outlier_region1 = (X_vals >= 3.0) & (X_vals <= 3.8)
    outlier_mask1 = outlier_region1 & (np.random.random(N_obs) < 0.08)
    X2_vals[outlier_mask1] += np.random.normal(0, 1.0, np.sum(outlier_mask1))
    
    # Region 2: Near the spike
    outlier_region2 = (X_vals >= 5.5) & (X_vals <= 6.5)
    outlier_mask2 = outlier_region2 & (np.random.random(N_obs) < 0.06)
    X2_vals[outlier_mask2] += np.random.normal(0, 0.8, np.sum(outlier_mask2))
    
    # Region 3: In the chaotic zone
    outlier_region3 = (X_vals >= 3.8) & (X_vals <= 4.8)
    outlier_mask3 = outlier_region3 & (np.random.random(N_obs) < 0.10)
    X2_vals[outlier_mask3] += np.random.normal(0, 0.7, np.sum(outlier_mask3))
    
    return X2_vals


X2_truth = boundary_truth_ver02(X1_truth)
X1_truth = X1_truth.reshape(len(X1_truth), 1)
X2_truth = X2_truth.reshape(len(X2_truth), 1)

# 2 features 
X_truth = np.concatenate([X1_truth, X2_truth], axis = 1) 

# plot the true boundary
plt.figure(figsize=(8, 5))
plt.plot(X_truth[:, 0], X_truth[:, 1])
plt.show()


# Create X and Y values (samples)
N = 20000
X1 = np.random.uniform(min(X_truth[:, 0]), max(X_truth[:, 0]), N)
X2 = np.random.uniform(min(X_truth[:, 1]), max(X_truth[:, 1]), N)
X2_comp = boundary_truth_ver02(X1)
mask_comp = X2 > X2_comp
Y_label = np.zeros(N)
Y_label[mask_comp] = 1
Y_col = np.where(Y_label == 0, "b", "r")

# plot the true boundary with points
plt.figure(figsize=(8, 5))
plt.plot(X_truth[:, 0], X_truth[:, 1], c = "g")
plt.scatter(X1, X2, s = 1, c = Y_col)
plt.savefig(outpath + "/true_boundary.png", dpi=300, bbox_inches="tight")
plt.show()

# A blind eye to the boundary
plt.figure(figsize=(8, 5))
plt.scatter(X1, X2, s = 1, c = Y_col)
plt.savefig(outpath + "/samples.png", dpi=300, bbox_inches="tight")
plt.show()



# concatenate the two features
X_samp = np.concatenate([X1.reshape(N, 1), 
                         X2.reshape(N, 1)], axis = 1 )

# normalize before feeding inputs into the radial basis function 
X_samp_norm = (X_samp - np.mean(X_samp, axis = 0)) / np.std(X_samp, axis = 0)

# Plot only the samples. 
# NOTE: We need to come up with the boundary.
plt.figure(figsize=(8, 5))
plt.scatter(X_samp_norm[:, 0], X_samp_norm[:, 1], s = 1, c = Y_col)
plt.show()

###################################################
####################################
#
# Lloyd's algorithm 
# Unsupervised learning (using Xs to learn about data without touching Y)
####################################
###################################################
# pick out initial centers 
k = 100 # number of basis
indx = np.random.choice(len(X_samp_norm), k)
c_init = X_samp_norm[indx]


# calculate the distance between X_samp and the centers
def dist_cal(sample_points, C):

    # Description: Calculates the distance between 2 points (sample_points and C)
    # @Arg sample_points: points (first set of points)
    # @Arg C: centers (another set of points)

    X1_diff = (sample_points[:, 0].reshape(N, 1) - C[:, 0])**2
    X2_diff = (sample_points[:, 1].reshape(N, 1) - C[:, 1])**2

    dist = X1_diff + X2_diff

    return dist

# get the initial distance and index (or the initial cluster assignment)
dist_init = dist_cal(X_samp_norm, c_init) # distance
dist_ind = np.argmin(dist_init, axis = 1) # initial cluster

# get new centers 

def get_new_centers(sample_points, C_num, cluster_ind):

    new_C = np.array([sample_points[cluster_ind == j].mean(axis = 0) for j in range(C_num)])

    # in some cases there might be missing value for centers
    for i in range(C_num):
        if np.isnan(new_C[i])[0] == True:
            new_C[i] = sample_points[np.random.choice(sample_points.shape[0])] # randomly assign new center point
        else:
            print("New center is not missing.")

    return new_C

# initialize the new center
new_C = get_new_centers(sample_points = X_samp_norm, C_num = k, cluster_ind = dist_ind)

# loop for convergence
for i in range(500):

    # get new distance matrix based on new centers
    new_dist_mat = dist_cal(X_samp_norm, new_C)
    # find the new cluster allocation
    new_dist_ind = np.argmin(new_dist_mat, axis = 1) 
    # new centers of one round prior will be the old center
    old_C = new_C
    # get the new centers based on new distance clusters
    new_C = get_new_centers(sample_points = X_samp_norm, C_num = k, cluster_ind = new_dist_ind)

    if(np.allclose(new_C, old_C, rtol = 1e-05, atol = 1e-08)):
        print(f"{i}: Converged")
        break
    else:
        print(f"{i}: has not converged")    


# plot the center (initial assignment)
plt.figure(figsize=(8, 5))
plt.scatter(c_init[:, 0], c_init[:, 1], s = 20, c = "g")
plt.scatter(X_samp_norm[:, 0], X_samp_norm[:, 1], s = 2, c = Y_col, alpha=0.3)
plt.title("Initial assignment of centers")
plt.savefig(outpath + "/fig_initial_cluster_means.png", dpi=300, bbox_inches="tight")
plt.show()


# plot the center
plt.figure(figsize=(8, 5))
plt.scatter(new_C[:, 0], new_C[:, 1], s = 20, c = "g")
plt.scatter(X_samp_norm[:, 0], X_samp_norm[:, 1], s = 2, c = Y_col, alpha = 0.3)
plt.title("Centers after convergence from Lloyd's algorithm")
plt.savefig(outpath + "/fig_converged_cluster_means.png", dpi=300, bbox_inches="tight")
plt.show()


#####################################
###############################

# Get the radial basis 

###############################
#####################################

# get the distance mat based on the converged centers
dist_mat = dist_cal(X_samp_norm, new_C)


def dat_process(gamma_val, Y, distance_mat = dist_mat):

    X_mat = np.exp(-gamma_val * (distance_mat))

    # Split into training and testing sets
    
    # create training and testing mask
    N_obs = len(Y)
    N_train = int(round(N_obs * 0.7, 0))
    train_ind = np.random.choice(N_obs, N_train, replace = False)
    train_mask = np.zeros(N_obs, dtype = bool)
    train_mask[train_ind] = True 
    test_mask = np.where(train_mask == False, True, False)

    X_train = X_mat[train_mask]
    X_test = X_mat[~train_mask]

    # Normalize to have mean 0 and std 1. 
    # NOTE: Only use mean and std of traning data while normalizing.
    X_test = (X_test - np.mean(X_train, axis = 0)) / (np.std(X_train, axis = 0))
    X_train = (X_train - np.mean(X_train, axis = 0)) / (np.std(X_train, axis = 0))

    Y_train = Y[train_mask]
    Y_test = Y[~train_mask]

    return [X_train, X_test, Y_train, Y_test, train_mask, test_mask]

# get the processed data
processed_dat = dat_process(gamma_val = 10, Y = Y_label)
X_train_dat, X_test_dat, Y_train_dat, Y_test_dat = processed_dat[0:4]
X_train_mask = processed_dat[4]
X_test_mask = processed_dat[5]

# Get raw train and test data using masks 
X_raw_train = X_samp[X_train_mask]
X_raw_test = X_samp[X_test_mask]

# Plot the training and testing data to make sure that they are representative 
plt.figure(figsize=(8, 5))
plt.scatter(X_raw_train[:, 0], X_raw_train[:, 1], c = "b", s = 1)
plt.scatter(X_raw_test[:, 0], X_raw_test[:, 1], c = "r", s = 1)
plt.show()

# input dimension 
input_dim = k

# layout of the model 
def nn_model(features_dim = input_dim):

    mod = keras.Sequential([
        layers.Dense(64, activation = "relu", input_shape = (features_dim, )), 
        layers.Dense(32, activation = "relu"), 
        layers.Dense(16, activation = "relu"), 
        layers.Dense(1, activation = "sigmoid")
    ])

    return mod

# get validation set 
def val_set(X, Y):

    N_obs = X.shape[0] 
    N_part_train = int(round(N_obs * 0.6, 0)) # number of partial training samples (70% of train sample)
    ind_part_train = np.random.choice(N_obs, N_part_train, replace = False)
    mask_part_train = np.zeros(N_obs, dtype = bool)
    mask_part_train[ind_part_train] = True

    X_part_train = X[mask_part_train]
    X_val = X[~mask_part_train]

    Y_part_train = Y[mask_part_train]
    Y_val = Y[~mask_part_train]

    return [X_part_train, X_val, Y_part_train, Y_val]

# split the training data into partial training and 
# validation sets
partial_X_train, X_val, partial_Y_train, Y_val = val_set(X = X_train_dat, Y = Y_train_dat)

# call model 
model = nn_model()

# compile model
model.compile(optimizer = "adam", 
              loss = "binary_crossentropy", 
              metrics = ["accuracy"])

# model fit 
num_epoch = 500
history = model.fit(partial_X_train, 
                    partial_Y_train, 
                    epochs = num_epoch, 
                    batch_size = 32, 
                    validation_data = (X_val, Y_val))

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label = "train accuracy")
plt.plot(history.history["val_accuracy"], label = "val accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig(outpath + "/model_metrics_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label = "train loss")
plt.plot(history.history["val_loss"], label = "val loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig(outpath + "/model_metrics_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------
# Build the Final Model
# ---------------------------------------------------
# call model
final_model = nn_model()

# compile
final_model.compile(optimizer = "adam", 
                    loss = "binary_crossentropy", 
                    metrics = ["accuracy"])

# model fit 
final_history = final_model.fit(X_train_dat, Y_train_dat, 
                                epochs = 50, 
                                batch_size = 32)

# evaluate on test data 
final_model.evaluate(X_test_dat, Y_test_dat)
predict_test = model.predict(X_test_dat)
plt.hist(predict_test)


# ---------------------------------------------------
# Decision Boundary Plot in Original Space
# ---------------------------------------------------

# 1. Create grid over X1, X2 (normalized space)
x_min, x_max = X_samp_norm[:, 0].min(), X_samp_norm[:, 0].max()
y_min, y_max = X_samp_norm[:, 1].min(), X_samp_norm[:, 1].max()

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

# 2. Compute distance from grid → RBF centers
dist_grid = ((grid_points[:, 0][:, None] - new_C[:, 0])**2 +
             (grid_points[:, 1][:, None] - new_C[:, 1])**2)

gamma = 10
X_grid = np.exp(-gamma * dist_grid)

# 3. Normalize using TRAINING statistics (important!)
X_grid = (X_grid - X_train_dat.mean(axis=0)) / X_train_dat.std(axis=0)

# 4. Predict with neural network
Z = final_model.predict(X_grid).reshape(xx.shape)

# 5. Plot decision boundary on real inputs
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, levels=[0,0.5,1], alpha=0.3, colors=["blue","red"])
plt.scatter(X_samp_norm[:,0], X_samp_norm[:,1], s=1, c=Y_col)
plt.title("RBF Neural Network - Learned Decision Boundary")
plt.savefig(outpath + "/final_boundary_detection.png", dpi=300, bbox_inches="tight")
plt.show()