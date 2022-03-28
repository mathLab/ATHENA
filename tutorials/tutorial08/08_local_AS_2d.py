
import numpy as np
import matplotlib.pyplot as plt

from athena import Normalizer, TopDownHierarchical, KMeansAS, KMedoidsAS, ClassifyAS, plot_scores

# import logging
# logging.basicConfig(filename='divisive.log', level=logging.DEBUG)

def plot_scores_(possible_clusters, config):
    """
    method='kmeans' or 'kmedoids' or 'divisive'
    """
    scores = []
    for n_clusters in possible_clusters:
        if config['method'] == 'kmeans':
            config['n_clusters'] = n_clusters
            cluster_as = KMeansAS(config=config)
        elif config['method'] == 'kmedoids':
            config['n_clusters'] = n_clusters
            cluster_as = KMedoidsAS(config=config)
        elif config['method'] == 'top-down':
            config['max_clusters'] = n_clusters
            config['min_children'] = n_clusters
            config['max_children'] = n_clusters
            cluster_as = TopDownHierarchical(config=config)
        cluster_as.fit()
        score = cluster_as.compute_scores(inputs_test, outputs_test)
        scores.append(score)
    scores = np.asarray(scores)
    print("Clustering with {} method".format(config['method']))
    print('r2_local = {}'.format(scores[:, 1]))
    print('mae_local = {}'.format(scores[:, 3]))
    plot_scores(possible_clusters, scores, plot_config)

def quartic_2d(X, n):
    # Unnormalize inputs
    x = n.inverse_transform(X)

    f = x[:, 0]**4 - x[:, 1]**4
    df = np.empty(x.shape)
    df[:, 0] = 4.0 * x[:, 0]**3
    df[:, 1] = -4.0 * x[:, 1]**3
    return f, df

np.random.seed(42)

n_params = 2
n_samples = 400
n_samples_test = 600

lb = np.array(n_params * [0])
ub = np.array(n_params * [1])

scaler = Normalizer(lb, ub)
real_inputs = np.random.uniform(scaler.lb, scaler.ub, (n_samples, n_params))
inputs = scaler.fit_transform(real_inputs)


# output values (f) and gradients (df)
outputs, gradients = quartic_2d(inputs, scaler)

real_inputs_test = np.random.uniform(scaler.lb, scaler.ub,
                                     (n_samples_test, n_params))
inputs_test = scaler.fit_transform(real_inputs_test)
outputs_test = quartic_2d(inputs_test, scaler)[0]

config = {
    'method': 'kmedoids',
    'n_clusters': 6,
    'as_dim': 1,
    'random_state': 0,
    'max_clusters': 5,
    'max_red_dim': 2,
    'max_children': 3,
    'min_children': 3,
    'min_local': 10,
    'dim_cut': 0.7,
    'normalization': 'uniform',
    'metric': 'as',
    'refinement_criterion': 'force',
    'as_dim_criterion': 'residual',
    'minimum_score': 2,
    'score_tolerance': 0.99,
    'max_dim_refine_further':2,
    'inputs_test': inputs_test,
    'outputs_test': outputs_test,
    'inputs': inputs,
    'outputs': outputs,
    'gradients': gradients,
}

plot_config = {
    'figsize': (10, 5),
    'title': 'Quartic 2D',
    'mae_bottom': 40,
    'mae_top': 100,
    'r2_bottom': 0.67,
    'r2_top': 1.03,
    'loc': 7,
    'main': 'r2',
    'filename': 'quartic_2d_r2_top-down_2_10.pdf'
}

# TopDownHierarchical clusters
cluster_as = TopDownHierarchical(config=config)
cluster_as.fit()
cluster_as.plot_clusters(with_test=True)

# classification with local AS
agg = ClassifyAS(inputs,
                 gradients,
                 n_neighbours=6,
                 threshold=0.9999,
                 neighbour_resampling=5)
n_c, labels = agg.fit()
print("Number of components: ", n_c)
agg.plot()
score = agg.plot_decision_boundaries()
print("Classification error on train set {}".format(score))
plt.show()

# convergence study
cl_min, cl_max = 2, 8
possible_clusters = np.arange(cl_min, cl_max, 1)

# k-means
config['method'] = 'kmeans'
plot_config['filename'] = 'quartic_2d_r2_kmeans_{}_{}.pdf'.format(cl_min, cl_max)
plot_scores_(possible_clusters, config)

# k-medoids
config['method'] = 'kmedoids'
plot_config['filename'] = 'quartic_2d_r2_kmedoids_{}_{}.pdf'.format(cl_min, cl_max)
plot_scores_(possible_clusters, config)

# top-down clustering
config['method'] = 'top-down'
plot_config['filename'] = 'quartic_2d_r2_top-down_{}_{}.pdf'.format(cl_min, cl_max)
plot_scores_(possible_clusters, config)