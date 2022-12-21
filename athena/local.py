"""
Module local Active Subspaces.

:References:

    - Romor Francesco, Marco Tezzele, and Gianluigi Rozza.
      "A local approach to parameter space reduction for regression and classification tasks." arXiv preprint arXiv:2107.10867 (2021).

"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import sqrtm, inv

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score

import GPy
from athena import Normalizer, ActiveSubspaces

_log = logging.getLogger('hierarchical')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

plt.rcParams.update({'font.size': 14})


class MaximumASDimensionReached(Exception):
    """Raised when all clusters have maximum AS dimension."""


class ClusterBase():
    """Local Active Subspaces clustering Base class.
    """
    def __init__(self):

        self.inputs = None
        self.outputs = None
        self.gradients = None
        self.inputs_test = None
        self.outputs_test = None
        self.inputs_val = None
        self.outputs_val = None

        self.outputs_dim = None
        self.method = None  # gradients evaluation method
        self.as_dim = None

        self.clustering = None  # implemented KMeans or KMedoids

        # cluster labels start from 0
        self.labels = None  # labels list of each input
        self.unique_labels = None  # set of labels

        self.full_as = None
        self.full_gpr = None
        self.local_ass = None
        self.local_gprs = None

        self.max_clusters = None
        self.random_state = None

    def _init_local_as(self,
                       inputs=None,
                       outputs=None,
                       gradients=None,
                       config=None):
        assert inputs is not None and outputs is not None, 'inputs or outputs argument is None'

        if not isinstance(inputs, np.ndarray) or not isinstance(
                outputs, np.ndarray):
            raise TypeError('Inputs and outputs type is not np.ndarray.')

        self.inputs = inputs
        self.outputs = outputs.squeeze()
        if gradients is None or isinstance(gradients, np.ndarray):
            self.gradients = gradients
        else:
            raise TypeError('Gradients type is not np.ndarray.')

        self.method = 'exact' if self.gradients is not None else 'local'

        self.as_dim = min(config['as_dim'], self.inputs.shape[1])

        self.max_clusters = config['max_clusters']
        self.random_state = config['random_state']

        # test samples used to evalute prediction errors
        self.inputs_test = config['inputs_test'] if config.get(
            'inputs_test', None) is not None else self.inputs
        self.outputs_test = config['outputs_test'] if config.get(
            'outputs_test', None) is not None else self.outputs

        # validation samples used to tune the local AS' regressions
        self.inputs_val = config['inputs_val'] if config.get(
            'inputs_val', None) is not None else self.inputs
        self.outputs_val = config['outputs_val'] if config.get(
            'outputs_val', None) is not None else self.outputs

        self.outputs_dim = self.outputs.shape[1] if len(
            self.outputs.shape) == 2 else 1

    def fit(self, inputs=None, outputs=None, gradients=None, config=None):
        """
        """
        self._init_local_as(inputs, outputs, gradients, config)
        self._fit_global()
        self._fit_clustering()
        self._fit_reduction()
        self._fit_regressions()

    def _fit_clustering(self):
        """Init unique_labels, labels, method predict"""
        raise NotImplementedError(
            f'Subclass must implement abstract method {self.__class__.__name__}._fit_clustering'
        )

    def _fit_global(self, plot=False):
        """Compute global Active Subspace"""
        self.full_as = ActiveSubspaces(dim=self.as_dim, method=self.method)
        self.full_as.fit(inputs=self.inputs,
                         outputs=self.outputs,
                         gradients=self.gradients)
        if plot:
            self.full_as.plot_eigenvalues()
            self.full_as.plot_sufficient_summary(self.inputs, self.outputs)

    def _fit_reduction(self):
        """Initialize list of local Active Subspaces."""
        self.local_ass = []
        for label in self.unique_labels:
            class_member_mask = (self.labels == label)

            ass = ActiveSubspaces(dim=self.as_dim, method=self.method)
            if self.gradients is None:
                ass.fit(inputs=self.inputs[class_member_mask],
                        outputs=self.outputs[class_member_mask])
            else:
                ass.fit(gradients=self.gradients[class_member_mask])

            self.local_ass.append(ass)

    def _fit_regressions(self, plot=False):
        """Compute local response surfaces with Gaussian process regression."""
        self.full_gpr = GPy.models.GPRegression(self.full_as.transform(
            self.inputs)[0],
                                                self.outputs.reshape(-1, 1),
                                                normalizer=True)
        self.full_gpr.optimize_restarts(10, verbose=False)
        if plot:
            self.full_gpr.plot()
            plt.show()

        self.local_gprs = []

        # fit n_clusters gprs
        for label, ass in zip(self.unique_labels, self.local_ass):
            class_member_mask = (self.labels == label)
            gpr = GPy.models.GPRegression(
                ass.transform(self.inputs[class_member_mask])[0],
                self.outputs[class_member_mask].reshape(-1, 1),
                normalizer=True)
            gpr.optimize_restarts(10, verbose=False)
            self.local_gprs.append(gpr)

    def compute_scores(self, inputs_test, outputs_test):
        """"""
        full_pred = self.full_gpr.predict(
            self.full_as.transform(inputs_test)[0])[0]
        r2_full = r2_score(outputs_test,
                           full_pred,
                           multioutput='uniform_average')

        labels_test = self.clustering.predict(inputs_test)
        local_pred = np.empty(outputs_test.shape)
        for i, label in enumerate(labels_test):
            local_pred[i] = self.local_gprs[label].predict(
                self.local_ass[label].transform(inputs_test[i].reshape(
                    1, -1))[0])[0]

        r2_local = r2_score(outputs_test,
                            local_pred,
                            multioutput='uniform_average')

        mae_full = mean_absolute_error(full_pred, outputs_test)
        mae_local = mean_absolute_error(local_pred, outputs_test)

        return np.array([r2_full, r2_local, mae_full, mae_local])

    @staticmethod
    def cluster_metric(x, y):
        return np.linalg.norm(x - y) if x[-1] == y[-1] else 1000

    # TODO plot clusters with dimensions higher than 2 with TSNE ?
    def plot_clusters(self, save=False, title='2d_clusters', plot=True):
        """Plot clusters of 2d data."""
        assert self.inputs.shape[1] == 2
        plt.figure(figsize=(8, 8))
        plt.scatter(self.inputs[:, 0], self.inputs[:, 1], c=self.labels)
        plt.grid(linestyle='dotted')
        plt.tight_layout()
        if save:
            plt.savefig(f'{title}.pdf')
        if plot: plt.show()


class KMeansAS(ClusterBase):
    """Clustering with k-means"""
    def __init__(self):
        super().__init__()
        self.centers = None

    def _fit_clustering(self):
        self.clustering = KMeans(n_clusters=self.max_clusters,
                                 random_state=self.random_state)
        self.clustering.fit(self.inputs)

        self.labels = self.clustering.labels_
        self.unique_labels = list(set(self.labels))
        self.centers = self.clustering.cluster_centers_
        _log.debug(f"Number of clusters k-means: {self.max_clusters}")

        silhouette_ = silhouette_score(self.inputs, self.labels)
        _log.debug(
            "Silhouette Coefficient k-means: {:0.3f}".format(silhouette_))


class KMedoidsAS(ClusterBase):
    """Clustering with k-medoids"""
    def __init__(self):
        super().__init__()
        self.centers = None

    def _fit_clustering(self):
        self.clustering = KMedoids(metric=self.as_metric,
                                   n_clusters=self.max_clusters,
                                   random_state=self.random_state)
        self.clustering.fit(self.inputs)

        self.labels = self.clustering.labels_
        self.unique_labels = list(set(self.labels))
        self.centers = self.clustering.cluster_centers_
        _log.debug(f"Number of clusters k-means: {self.max_clusters}")

        silhouette_ = silhouette_score(self.inputs, self.labels)
        _log.debug(
            "Silhouette Coefficient k-means: {:0.3f}".format(silhouette_))

    def as_metric(self, X, Y):
        "AS weighted Euclidean metric"
        return np.linalg.norm(
            self.full_as.evects.T.dot(X - Y) * self.full_as.evals)


class TopDownHierarchicalAS(ClusterBase):
    def __init__(self):
        """TODO check states logic.
        1. 2 and 4 are exclusives
        2. 0 and 4 have the same role
        3. if 4 and 1 are present, 4 has priority over 1

        :param dict config: 'normalization': uniform, gaussian, root 'metric': as, CI, euclidean, kmeans
        'refinement_criterion': global, mean, force
        'as_dim_criterion': spectral_gap, residual
        """
        super().__init__()

        # clustering parameters
        self.max_clusters = None
        self.max_red_dim = None
        self.max_children = None
        self.min_children = None
        self.min_local = None  # minimum number of elements
        self.score_tolerance = None
        self.dim_cut = None

        self.normalization = None  # uniform, gaussian, root
        self.metric = None  # as, CI, euclidean
        self.refinement_criterion = None  #global, mean, force
        self.as_dim_criterion = None  # spectral_gap, residual

        self.minimum_score = None  # it increases the cluster dimension until the score threshold is reached
        self.max_dim_refine_further = None
        self.verbose = False

        self.total_clusters = 0
        self.state_d = {
            '0': "running",
            '1': "minimum number of elements violated",
            '2': "score tolerance achieved",
            '3': "maximum number of clusters reached",
            '4': "score tolerance not achieved",
            '5': "too little samples to cluster with given clusters range",
            '6': "no clustering performed",
            '7': "node not refined"
        }

        self.root = None

    def _init_hierarchical(self, config):
        """"""
        try:
            # clustering parameters
            self.max_clusters = config['max_clusters']
            self.max_red_dim = config['max_red_dim']
            self.max_children = config['max_children']
            self.min_children = config['min_children']
            self.min_local = config['min_local']
            self.score_tolerance = config['score_tolerance']
            self.dim_cut = config['dim_cut']
            self.normalization = config['normalization']
            self.metric = config['metric']
            self.refinement_criterion = config['refinement_criterion']
            self.as_dim_criterion = config['as_dim_criterion']

        except KeyError as k:
            raise KeyError(
                f"Missing mandatory keyword {k.args[0]} for class TopDownHierarchical"
            ) from k

        self.minimum_score = config.get('minimum_score', None)
        self.max_dim_refine_further = config.get('max_dim_refine_further',
                                                 self.inputs.shape[1])
        self.verbose = config.get('verbose', True)

    def fit(self, inputs=None, outputs=None, gradients=None, config=None):
        """
        """
        self._init_local_as(inputs, outputs, gradients, config)
        self._init_hierarchical(config)
        self._fit_global()
        scores, leaves_dimensions_list = self._fit_clustering()
        return scores, leaves_dimensions_list

    def _fit_clustering(self, print_states=False, plot=False):
        self.root = TopDownNode(None, np.arange(self.inputs.shape[0]),
                                np.arange(self.inputs_val.shape[0]), self)

        self.total_clusters += 1

        # check if without clustering the tolerance is achieved
        if self.root.score > self.score_tolerance:
            self.root.state = 2
            state = {6}  # init states set
            _log.debug("No clustering")
        else:
            self.root.state = 4
            children_fifo = [self.root]
            _log.debug("Add root")

            # continue until children_fifo is empty
            while (children_fifo != []):
                _log.debug(
                    f"Before pop total queue length is: {len(children_fifo)},\n total clusters: {str(self.total_clusters)}"
                )

                node = children_fifo.pop(0)
                state, children = node.refine_cluster()

                _log.debug(
                    f"After pop {str(state)} and len children list added is {len(children)} total clusters: {str(self.total_clusters)}"
                )

                if 3 in state:
                    break
                elif 7 in state:
                    continue
                elif 4 in state:  # refine further the present node
                    self.total_clusters += len(children) - 1
                    children_fifo.extend(child for child in children
                                         if child.state == 4)
                elif 2 in state:  # tolerance reached, continue
                    self.total_clusters += len(children) - 1
                elif any(
                        x in state
                        for x in [1, 5]):  # abort further clustering conditions
                    continue

        res = self.compute_scores(self.inputs_test, self.outputs_test)
        n_leaves, leaves_dim, _ = self._print_leaves_score()

        if self.verbose:
            print("Hierarchical top-down clustering completed with states")
            for sta in state:
                print(sta, " : ", self.state_d[str(sta)])
            print("n_leaves: {:d}".format(n_leaves))
            print(f"Test score: {res}")

        if print_states:
            self._print_state()
        if plot:
            self.plot_clusters()

        if self.minimum_score is not None:
            self.refine_further(self.minimum_score)

        return res, leaves_dim

    def refine_one_step(self):
        """Increase the dimension of the Active Subspace once, when possible."""
        class LeafUpdate(object):
            def __init__(self):
                self.score = 0
                self.leaves_list = []

            def __call__(self, node):
                self.score = max(self.score, node.score)
                self.leaves_list.append(node)

        leaf_update = LeafUpdate()
        self.breadth_first_search(leaf_update, self.void_func)
        leaf_update.leaves_list.sort(key=lambda leaf: leaf.score)

        index = 0
        for leaf in leaf_update.leaves_list:
            if leaf.r_dim < self.inputs.shape[1]:
                break
            else:
                index += 1

        if len(leaf_update.leaves_list) == index:
            print("All leaves have max AS dimension")
            raise MaximumASDimensionReached
        else:
            node_to_update = leaf_update.leaves_list[index]

            # change as dimension node_to_update.ss.dim += 1
            node_to_update.r_dim += 1

    def refine_further(self, minimum_score, plot=False):
        """Increase the dimension of the local AS until the tolerance on the
        prediction error is reached."""
        print("Start refining: increasing the as dimension when possible.")

        class CallRefine(object):
            def __init__(self, minimum_score):
                self.min = minimum_score

            def __call__(self, node):
                node.refine_further(self.min)

        call_refine = CallRefine(minimum_score)
        self.breadth_first_search(call_refine, self.void_func)

        n_leaves = self._print_leaves_score()[0]
        res = self.compute_scores(self.inputs_test, self.outputs_test)

        if self.verbose:
            print("Finished increasing as dimension when possible.")
            print("n_leaves: ", n_leaves)
            print("Test score: ", res)

        if plot: self.plot_clusters()

    def _print_state(self):
        """Print the states of every node."""
        children_fifo = [self.root]
        while (children_fifo != []):
            node = children_fifo.pop(0)
            print(node.state, " : ", self.state_d[str(node.state)])
            children_fifo.extend(node.children)

    def _print_state_debug(self):
        "Save in the logger debug info from every node."
        children_fifo = [self.root]
        while (children_fifo != []):
            node = children_fifo.pop(0)
            _log.debug(f"{str(node.state)} : {self.state_d[str(node.state)]}")
            children_fifo.extend(node.children)

    def _print_leaves_score(self):
        """Print the information of every leaf."""
        class ComputeScore(object):
            def __init__(self):
                self.n_leaves = 0
                self.leaves_dim = []
                self.n_elems = []

            def __call__(self, node):
                self.n_leaves += 1
                print("score is: {:.2f}".format(node.score), " r_dim is: ",
                      node.r_dim, " state is: ", node.state, " n_elem : ",
                      node.ind.shape[0])
                self.leaves_dim.append(node.r_dim)
                self.n_elems.append(node.ind.shape[0])

        compute_score = ComputeScore()
        self.breadth_first_search(compute_score, self.void_func)

        return compute_score.n_leaves, compute_score.leaves_dim, compute_score.n_elems

    def assign_leaf_labels(self):
        """Assign integer labels to the leaves."""
        class LeafLabels(object):
            def __init__(self):
                self.labels_counter = 0

            def __call__(self, node):
                node.leaf_label = self.labels_counter
                self.labels_counter += 1

        leaf_labels = LeafLabels()
        self.breadth_first_search(leaf_labels, self.void_func)

    def reset_gprs(self):
        """Reset the GPRs of every leaf and root."""
        def reset_gpr(node):
            node.gpr = None
            node.ss = None

        self.breadth_first_search(reset_gpr, self.void_func, reset_gpr)

    def plot_clusters(self,
                      with_test=False,
                      save_data=True,
                      plot=True,
                      save=True):
        class SaveLeafInfo(object):
            def __init__(self):
                self.n_leaves = 0
                self.n_elems = []
                self.scores = []
                self.r_dims = []
                self.leaf_labels = []
                self.saved_data = []

            def __call__(self, node):
                self.n_leaves += 1
                self.n_elems.append(node.ind.shape[0])
                self.scores.append(node.score)
                self.r_dims.append(node.r_dim)
                self.leaf_labels.append(node.leaf_label)
                self.saved_data.append(node.ind)

        self.assign_leaf_labels()
        leaves_info = SaveLeafInfo()
        self.breadth_first_search(leaves_info, self.void_func)

        if save_data:
            to_be_saved = []
            for i in range(len(leaves_info.saved_data)):
                col_ones = np.ones(
                    (self.inputs[leaves_info.saved_data[i]].shape[0], 1))
                to_be_saved.append(
                    np.hstack((self.inputs[leaves_info.saved_data[i]],
                               leaves_info.scores[i] * col_ones,
                               leaves_info.r_dims[i] * col_ones,
                               leaves_info.leaf_labels[i] * col_ones)))
            arr = np.vstack(to_be_saved)
            _log.debug(f"Saved data with shape: {arr.shape}")
            np.save("clusters_r2_asdim_labels.npy", to_be_saved)

        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(111)
        colors = cm.rainbow(np.linspace(0, 1, self.max_clusters))

        for i in range(len(leaves_info.saved_data)):
            ax1.scatter(self.inputs[leaves_info.saved_data[i]][:, 0],
                        self.inputs[leaves_info.saved_data[i]][:, 1],
                        c=colors[leaves_info.leaf_labels[i]].reshape(1, -1),
                        label="r2={:.2f}, as_dim={}".format(
                            leaves_info.scores[i], leaves_info.r_dims[i]))

        if with_test:
            labels = self.predict(self.inputs_test)[1]
            for idl in range(self.total_clusters):
                mask = idl == labels
                ax1.scatter(self.inputs_test[mask, 0],
                            self.inputs_test[mask, 1],
                            c=colors[int(labels[mask][0])].reshape(1, -1),
                            alpha=0.5,
                            marker='x')

        plt.legend()
        plt.title(
            f"Hierarchical top-down clustering, total clusters= {self.total_clusters} "
        )
        plt.tight_layout()

        if save: plt.savefig("clusters_top_down.png")
        if plot: plt.show()
        plt.close()

    # overload
    def compute_scores(self,
                       test_inputs,
                       test_outputs,
                       custom_score=None,
                       classification_criterion='default'):
        """Compute the r2 scores associated to the predictions of the
        test_inputs variable. The classification on the local cluster is
        performed with the default cluster method (option 'default') or the
        minimum distance (option 'min_dist')."""
        if custom_score is None:

            def score(x, y):
                return np.mean(
                    np.clip(r2_score(x, y, multioutput='raw_values'), 0, 1))

            custom_score_ = score
        else:
            custom_score_ = custom_score

        full_pred = np.zeros_like(test_outputs)
        local_pred = np.zeros_like(test_outputs)
        for i_test in range(test_inputs.shape[0]):
            full_pred[i_test] = self.root.predict(test_inputs[i_test])

            children_fifo = [self.root]
            while (children_fifo != []):
                node = children_fifo.pop(0)
                if node.children == []:
                    local_pred[i_test] = node.predict(test_inputs[i_test])
                else:
                    test_normalized = node.normalizer(
                        inputs=test_inputs[i_test])[0]

                    if classification_criterion == 'default':
                        idx = node.cluster.predict(
                            np.atleast_2d(test_normalized))[0]
                    elif classification_criterion == 'min_dist':
                        idx = self._cluster_min_dist(test_inputs[i_test],
                                                     node.children)
                    else:
                        raise ValueError(
                            "Pass 'default' or 'min_dist' as classification criterions."
                        )

                    children_fifo.extend([node.children[idx]])

        r2_full = custom_score_(test_outputs, full_pred)
        r2_local = custom_score_(test_outputs, local_pred)
        mae_full = mean_absolute_error(test_outputs, full_pred)
        mae_local = mean_absolute_error(test_outputs, local_pred)

        scores = np.array([r2_full, r2_local, mae_full, mae_local])
        return scores

    def predict(self, test_inputs):
        predictions = np.zeros((test_inputs.shape[0], test_inputs.shape[1]))
        labels = np.zeros(test_inputs.shape[0])
        for i_test in range(test_inputs.shape[0]):
            children_fifo = [self.root]
            while (children_fifo != []):
                node = children_fifo.pop(0)
                if node.children == []:
                    predictions[i_test] = node.predict(test_inputs[i_test])
                    labels[i_test] = node.leaf_label
                else:
                    test_normalized = node.normalizer(
                        inputs=test_inputs[i_test])[0]
                    idx = node.cluster.predict(
                        np.atleast_2d(test_normalized))[0]

                    children_fifo.extend([node.children[idx]])

        return predictions, labels

    def _cluster_min_dist(self, test, clusters):
        group_list = [self.inputs[node.ind] for node in clusters]
        return self.set_min_dist(test, group_list)

    @staticmethod
    def set_min_dist(test, groups):
        "Returns the index of the group with the shortes distance from test."
        dist_group = np.zeros(len(groups))
        for i_cl, cluster in enumerate(groups):
            dist_group[i_cl] = np.min(
                np.linalg.norm(cluster - test.reshape(1, -1), axis=1))
        return np.argmin(dist_group)

    def breadth_first_search(self, to_do_leaf, to_do_inside, to_do_root=None):
        if to_do_root:
            to_do_root(self.root)
        children_fifo = [self.root]
        while (children_fifo != []):
            node = children_fifo.pop(0)
            if node.children == []:
                to_do_leaf(node)
            else:
                to_do_inside(node.children)
                children_fifo.extend(node.children)

    @staticmethod
    def void_func(*args, **kwargs):
        pass


class TopDownNode():
    def __init__(self, parent, node_indexes, val_indexes, tree_obj):
        """A TopDownNode is defined by the indexes of the triplets (inputs,
        outputs, gradients) of the training data and the parent node. The root
        has None parent. The invariants are:
        1. r_dim is between 1 and max_red_dim
        2. the score is above the score_tolerance
        3. len(children) is between min_children and max_children
        4. each child cluster has at least min_local elements

        The state of a child is 4 if the cluster needs to be refined or 2 if the
        tolerance is achieved and the cluster does not need further refinement.

        TODO: for the moment val_indexes refer to the training input dataset, it
        can be extended to arbitrary val sets but then the classification of the
        val set has to be done inside each refine_cluster (maybe as a tree
        classifier).

        """
        self.parent = parent
        self.children = []
        self.cluster = None

        self.ind = node_indexes
        self.t_ind = val_indexes

        self.r_dim = None
        self.score = None

        self._gpr = None  # GPR object
        self._ss = None  # AS object

        self.hierarchical = tree_obj
        self.state = None

        self.type = "root" if self.parent is None else self.hierarchical.normalization
        _log.debug(
            f"Validation set length: {self.hierarchical.inputs_val[self.t_ind].shape}"
        )

        ####### Estabilish invariants r_dim and score
        self.normalizer = self.NormalizeDivisive(self.type, self.ind,
                                                 self.hierarchical.inputs)

        inp_normalized, out, grad_normalized = self.normalizer(
            self.ind, self.hierarchical.inputs, self.hierarchical.outputs,
            self.hierarchical.gradients)

        # 1. r_dim is between 1 and max_red_dim
        ass = self.compute_as(inp_normalized, out, grad_normalized)
        # save essential info of AS without saving the whole of it
        self.evects = ass.evects
        self.evals = ass.evals / np.sum(ass.evals)

        self.cum_evals = np.array(
            [np.sum(self.evals[:i + 1]) for i in range(self.evals.shape[0])])

        if self.hierarchical.as_dim_criterion == "residual":
            self.r_dim = self.as_dim(self.hierarchical.max_red_dim,
                                     threshold=self.hierarchical.dim_cut)
        elif self.hierarchical.as_dim_criterion == "spectral_gap":
            self.r_dim = self.as_dim_gap(self.hierarchical.max_red_dim)
        else:
            raise ValueError(
                "The criterion for active subspace dimension evaluation must be among: residual, spectral_gap."
            )

        # change as dimension
        ass.dim = self.r_dim
        ass._partition()
        _log.debug(f"reduced dim is: {str(self.r_dim)}")

        # 2. the score is above the score_tolerance
        gpr = GPy.models.GPRegression(ass.transform(inp_normalized)[0],
                                      out.reshape(inp_normalized.shape[0], -1),
                                      normalizer=False,
                                      noise_var=1)
        gpr.optimize_restarts(10, verbose=False)

        t_inp_normalized = self.normalizer(self.t_ind,
                                           self.hierarchical.inputs_val,
                                           self.hierarchical.outputs_val)[0]

        local_pred = gpr.predict(ass.transform(t_inp_normalized)[0])[0]
        self.score = np.mean(
            np.clip(
                r2_score(self.hierarchical.outputs_val[self.t_ind],
                         local_pred,
                         multioutput='raw_values'), 0, 1))

        _log.debug(f"r2 score is: {str(self.score)}")

    def refine_further(self, minimum_score):
        _log.debug(f"r2 score inside refine_further is: {str(self.score)}")

        # create self.ss if it does not exist, through property decorator
        while (self.score < minimum_score
               and self.ss.dim + 1 <= self.hierarchical.max_dim_refine_further):
            # change as dimension
            self.ss.dim += 1
            self.r_dim += 1

            # ActiveSubspaces class returns throws ValueError when the dimension
            # of the active subspace is equal to the dimension of the whole
            # domain
            try:
                self.ss._partition()
            except ValueError:
                self.ss.W1 = self.ss.evects
                self.ss.W2 = None

            _log.debug("reduced dim inside refine_further loop is: " +
                       str(self.r_dim))

            t_inp_normalized = self.normalizer(self.t_ind,
                                               self.hierarchical.inputs_val,
                                               self.hierarchical.outputs_val)[0]
            # re-optimize gpr
            self._gpr = None

            # property decorator on self.gpr is called
            local_pred = self.gpr.predict(
                self.ss.transform(t_inp_normalized)[0])[0]
            self.score = r2_score(self.hierarchical.outputs_val[self.t_ind],
                                  local_pred,
                                  multioutput='uniform_average')

            _log.debug(f"r2 score inside while cycle is: {str(self.score)}")

    class NormalizeDivisive():
        """Inner class for normalization of inputs, gradients w.r.t. local
        clusters"""
        def __init__(self, norm_type, ind, inputs):
            self.type = norm_type

            if self.type == 'gaussian':
                _log.debug("Gaussian normalization")
                # ! it is improbable that the inverse does not exist if the
                # input points are more than 2 and well-distributed
                inp = inputs[ind]
                self.mean = np.mean(inp, axis=0, keepdims=True)
                cov = np.cov(inp.T)
                self.sqrt_cov = sqrtm(cov)
                self.inv_sqrt_cov = inv(self.sqrt_cov)

            elif self.type == 'uniform':
                _log.debug("Uniform normalization")
                inp = inputs[ind]
                lb = np.min(inp, axis=0, keepdims=False)
                ub = np.max(inp, axis=0, keepdims=False)
                self.nor = Normalizer(lb, ub)

            elif self.type == 'root':
                _log.debug("Without normalization")
            else:
                raise ValueError(
                    "A proper flag was not passed to normalize class: " +
                    self.type)

        def __call__(self, ind=None, inputs=None, outputs=None, gradients=None):
            if ind is not None:
                inp = inputs[ind]
                out = outputs[ind]
                if gradients is not None:
                    grad = gradients[ind]
                else:
                    grad = None
                    grad_normalized = None
            else:
                inp = inputs
                out = outputs
                if gradients is not None:
                    grad = gradients
                else:
                    grad = None
                    grad_normalized = None

            if self.type == 'gaussian':
                inp_normalized = (inp - self.mean).dot(self.inv_sqrt_cov)
                if grad is not None:
                    grad_normalized = grad.dot(self.sqrt_cov)

                return inp_normalized, out, grad_normalized

            elif self.type == 'uniform':
                inp_normalized = self.nor.fit_transform(inp)
                if grad is not None:
                    grad_normalized = grad * (
                        (self.nor.ub - self.nor.lb) / 2).reshape(1, -1)

                return inp_normalized, out, grad_normalized

            elif self.type == 'root':
                return inp, out, grad

    @property  # lazy property
    def gpr(self):
        if self._gpr is not None:
            return self._gpr
        inp_normalized, out, _ = self.normalizer(self.ind,
                                                 self.hierarchical.inputs,
                                                 self.hierarchical.outputs)

        gpr = GPy.models.GPRegression(self.ss.transform(inp_normalized)[0],
                                      out.reshape(inp_normalized.shape[0], -1),
                                      normalizer=False)
        gpr.optimize_restarts(20, verbose=False)
        self._gpr = gpr
        return gpr

    @gpr.setter
    def gpr(self, value):
        self._gpr = value

    @gpr.deleter  # lazy property
    def gpr(self):
        del self._gpr

    @property  # lazy property
    def ss(self):
        # TODO check type and behaviour e.g. return only if state is 2 or 4
        if self._ss is not None:
            return self._ss
        inp_normalized, out, grad_normalized = self.normalizer(
            self.ind, self.hierarchical.inputs, self.hierarchical.outputs,
            self.hierarchical.gradients)

        if grad_normalized is not None:
            ss = ActiveSubspaces(dim=self.r_dim)
            ss.fit(gradients=grad_normalized)
        else:
            ss = ActiveSubspaces(dim=self.r_dim, method='local')
            ss.fit(inputs=inp_normalized, outputs=out)

        self._ss = ss
        return ss

    @ss.setter
    def ss(self, value):
        self._ss = value

    def refine_cluster(self):
        """From inputs outputs pairs or inputs outputs gradients triplets,
        evaluates the number of kMedoids clusters in number between min_children
        and max_children. The metric used is the local as_metric.

        It returns a non empy list of children if state contains 4. It returns
        an empty list of children if state contains 2 (score tolerance achieved)
        or 3 (max clusters used) or 5 (clustering cannot be performed) or 1 (too
        ephimeral clusters)."""

        max_total_clusters = self.hierarchical.max_clusters
        min_clusters = self.hierarchical.min_children
        max_clusters = self.hierarchical.max_children
        min_local = self.hierarchical.min_local
        score_thre = self.hierarchical.score_tolerance
        priority = self.hierarchical.refinement_criterion

        state = {0}
        res_max = self.score  # best score among choice of n_clusters
        cluster_ = None

        _log.debug("Start refining from {} to {} clusters".format(
            min_clusters, max_clusters))
        for n_clusters in range(min_clusters, max_clusters + 1):

            _log.debug("Refine cluster with " + str(n_clusters) +
                       " clusters. Total would be: " +
                       str(self.hierarchical.total_clusters + n_clusters - 1) +
                       " \nRes max: {:.3f}".format(res_max) + " N_children: " +
                       str(len(self.children)))

            # check if max_clusters would be reached
            if max_total_clusters < self.hierarchical.total_clusters + n_clusters - 1:
                # state contains 3 only if during the for loop no refinement is
                # chosen
                if max_total_clusters == self.hierarchical.total_clusters + len(
                        self.children) - 1:
                    state.add(3)
                elif self.children == []:
                    state.add(7)

                _log.debug("Max clusters violation : " + str(state) +
                           " and list length " + str(len(self.children)))
                state.discard(0)
                return state, self.children

            # check if clustering is possible
            if self.ind.shape[0] < self.hierarchical.total_clusters + n_clusters:
                state.add(5)
                _log.debug("Refine returns 5 : " + str(state) +
                           " and list length " + str(len(self.children)))
                state.discard(0)
                return state, self.children

            # cluster with metric of choice
            if self.hierarchical.metric == 'as':
                _log.debug("AS metric")
                cluster_ = KMedoids(metric=self.as_metric,
                                    n_clusters=n_clusters,
                                    random_state=self.hierarchical.random_state)

                local_inp = self.normalizer(self.ind, self.hierarchical.inputs,
                                            self.hierarchical.outputs)[0]
            elif self.hierarchical.metric == 'kmeans':
                _log.debug("kmeans")
                cluster_ = KMeans(n_clusters=n_clusters,
                                  random_state=self.hierarchical.random_state)

                local_inp = self.normalizer(self.ind, self.hierarchical.inputs,
                                            self.hierarchical.outputs)[0]
            # C1 norm with outputs included
            elif self.hierarchical.metric == 'C1':
                _log.debug("C1 metric")
                cluster_ = KMedoids(n_clusters=n_clusters,
                                    random_state=self.hierarchical.random_state)

                inp_normalized, out, grad_normalized = self.normalizer(
                    self.ind, self.hierarchical.inputs,
                    self.hierarchical.outputs, self.hierarchical.gradients)
                local_inp = np.hstack(
                    (inp_normalized, out.reshape(-1, 1), grad_normalized))

            elif self.hierarchical.metric == 'euclidean':
                _log.debug("Euclidean metric")
                cluster_ = KMedoids(n_clusters=n_clusters,
                                    random_state=self.hierarchical.random_state)

                local_inp = self.normalizer(self.ind, self.hierarchical.inputs,
                                            self.hierarchical.outputs)[0]
            else:
                raise ValueError("Wrong metric value: " +
                                 self.hierarchical.metric +
                                 ".Possible values are as, C1, euclidean.")

            cluster_.fit(local_inp)
            labels = cluster_.labels_
            unique_labels = list(set(labels))
            _log.debug("Print labels: " + str(unique_labels))

            state.discard(1)
            for label in unique_labels:
                # check min number of elements
                n_local = list(labels).count(label)
                if n_local < min_local:
                    state.add(1)
                    break

            # check number of elements
            if 1 in state:
                _log.debug("Continue breaks 1 " + str(n_local) + ": " +
                           str(state) + " and list length " +
                           str(len(self.children)))
                continue

            # create children list
            children_ = []
            res_ = []
            state_ = []
            val_indexes = cluster_.predict(
                self.hierarchical.inputs_val[self.t_ind])
            for label in unique_labels:
                mask = labels == label
                mask_val = val_indexes == label
                m_ind = self.ind[mask]
                m_val = self.t_ind[mask_val]

                # score is evaluated on the train data, when the validation set
                # is not rich enough to be spread among all the clusters
                if len(m_val) == 0: m_val = m_ind

                node = TopDownNode(self, m_ind, m_val, self.hierarchical)
                children_.extend([node])

                # check if at least one cluster has not reached tolerance
                res_.append(node.score)
                if node.score >= score_thre:
                    state_.append(2)
                    node.state = 2
                else:
                    state_.append(4)
                    node.state = 4

            _log.debug("Children states: " + str(state_))

            # update state status
            state.discard(2)
            state.discard(4)
            state.add(max(state_))

            # choose to refine based on the specified priority
            if priority == "global":
                ret_val, res_max, state = self.global_children_priority(
                    children_, res_max, res_, state_, cluster_, state)
            elif priority == "mean":
                ret_val, res_max, state = self.mean_children_priority(
                    children_, res_max, res_, state_, cluster_, state)
            elif priority == "force":
                ret_val, res_max, state = self.force_refinement_priority(
                    children_, res_max, res_, state_, cluster_, state)
            else:
                raise ValueError(
                    "Wrong priority value. Possible values are global, mean, force."
                )

            if ret_val == 'break':
                break
            elif ret_val == 'continue':
                continue
            else:
                raise ValueError(
                    "Invalid return value from priority criteria for refinement."
                )

        # if no refinement is chosen, return state 7
        if self.children == []:
            state.add(7)
            state.discard(2)
            state.discard(4)
            _log.debug("Refine returns 7 : " + str(state) +
                       " and list length " + str(len(self.children)))

        _log.debug("Refine returns 2,4,1 : " + str(state) +
                   " and list length " + str(len(self.children)))

        state.discard(0)
        return state, self.children

    def global_children_priority(self, children, res_max, res, children_states,
                                 cluster, state):
        """ The refinement is chosen if the r2 score of the divisive node is
        bested by the r2 score on the whole predictions of the children
        nodes."""

        # costly r2 evaluation with gpr
        pred = np.zeros(self.hierarchical.inputs.shape[0])
        for node in children:
            pred[node.ind] = node.predict(
                self.hierarchical.inputs[node.ind]).squeeze()

        pred = pred[self.ind]
        full_children_r2 = r2_score(self.hierarchical.outputs[self.ind],
                                    pred,
                                    multioutput='uniform_average')

        if res_max < full_children_r2:
            _log.debug(f"Best score bested: {res_max} < {full_children_r2}")
            res_max = full_children_r2
            self.children = children
            self.cluster = cluster
        else:
            _log.debug(f"Best score not bested: {res_max} < {full_children_r2}")

        if res_max >= self.hierarchical.score_tolerance:
            return_value = "break"
        else:
            return_value = "continue"

        return return_value, res_max, state

    def mean_children_priority(self, children, res_max, res, children_states,
                               cluster, state):
        """The refinement is chosen if the local r2 scores are above the
        threshold or if res_max is below the average of the local r2 scores. It
        can happen that refinement is not performed because the parent score is
        the best."""
        res_ = sum(res) * (1 / len(res))
        max_state = max(children_states)

        # stop refinement: tolerance achieved for the first time w.r.t.
        # n_clusters loop
        if max_state == 2:
            _log.debug("Leaf reached")
            self.children = children
            self.cluster = cluster
            return_value = "break"

        elif max_state == 4:
            _log.debug("Leaf not reached, children states" +
                       str(children_states))

            # if the overall score is bested change children DEBUG changed to
            # 1.01 instead of * 1
            if res_max < res_:
                _log.debug(f"Bested: {str(res_max)} < {str(res_)}")
                res_max = res_
                self.children = children
                self.cluster = cluster
            else:
                _log.debug(f"Not bested: {str(res_max)}  {str(res_)}")

            return_value = "continue"

        return return_value, res_max, state

    def force_refinement_priority(self, children, res_max, res, children_states,
                                  cluster, state):
        """The refinement is chosen if the local r2 scores are above the
        threshold or if res_max is below the average of the local r2 scores. It
        can NOT happen that refinement is not performed because the parent score
        is the best.
        """
        res_ = sum(res) * (1 / len(res))
        max_state = max(children_states)

        # stop refinement: tolerance achieved for the first time w.r.t.
        # n_clusters loop
        if max_state == 2:
            _log.debug("Leaf reached")
            self.children = children
            self.cluster = cluster
            return_value = "break"

        elif max_state == 4:
            _log.debug("Leaf not reached, children states" +
                       str(children_states))

            # The following lines distinguish force_refinement_priority from
            # mean_refinement_priority. If it is the first iteration w.r.t
            # n_clusters loop from min_clusters to max_clusters, update the
            # children list anyway (even if the refinement is not associated to
            # the best score).
            if self.children == []:
                self.children = children
                self.cluster = cluster

            # if the overall score is bested change children
            if res_max < res_:
                _log.debug(f"Bested: {str(res_max)} < {str(res_)}")
                res_max = res_
                self.children = children
                self.cluster = cluster
            else:
                _log.debug(f"Not Bested: {str(res_max)} > {str(res_)}")

            return_value = "continue"

        return return_value, res_max, state

    @staticmethod
    def compute_as(inputs, outputs, gradients):
        """Returns ss"""
        if gradients is not None:
            ss = ActiveSubspaces(dim=1)
            ss.fit(gradients=gradients)
        else:
            ss = ActiveSubspaces(dim=1, method="local")
            ss.fit(inputs=inputs, outputs=outputs)

        return ss

    def as_dim_gap(self, max_dim):
        r_dim = 0
        assert max_dim < self.evals.shape[
            0], "max_red_dim cannot be equal or grater than the input dimension"
        max_gap = self.evals[0] - self.evals[1]
        _log.debug(f"Spectral gaps: {str(max_gap)}")
        r_dim = 1
        i = 1
        while (i < max_dim):
            gap = self.evals[i] - self.evals[i + 1]
            _log.debug(f"Spectral gaps: {str(gap)}")
            if gap > max_gap:
                max_gap = gap
                r_dim = i + 1
            i += 1
        _log.debug(f"Returned: {str(r_dim)}")
        return r_dim

    def as_dim(self, max_dim, threshold=0.95):
        r_dim = 1
        while (r_dim <= max_dim):
            _log.debug(
                "Spectral cumulatives: {:.6f}, threshold: {:.6f}, red dim {}".
                format(self.cum_evals[r_dim - 1], threshold, r_dim))

            if self.cum_evals[r_dim - 1] >= threshold:
                _log.debug(f"break {r_dim}")
                break
            r_dim += 1
        else:
            _log.debug(f"else {r_dim}")
            r_dim -= 1
        _log.debug(f"Returned: {r_dim}")
        return r_dim

    def as_metric(self, X, Y):
        """Metrics that agglomerate clusters in the direction trasversal to the
        active subspace."""
        return np.linalg.norm(self.evects.T.dot(X - Y) * self.evals)

    def predict(self, test):
        test_normalized = self.normalizer(inputs=test)[0]

        # DEBUG inp = self.hierarchical.inputs[self.ind] inp_normalized = (inp -
        # self.mean).dot(self.inv_sqrt_cov) fig = plt.figure() ax1 =
        # fig.add_subplot(111)
        # # ax1.scatter(test_normalized[:, 0], test_normalized[:, 1], c='r')
        # # ax1.scatter(inp_normalized[:, 0], inp_normalized[:, 1], c='b')
        # ax1.scatter(self.mean[:, 0], self.mean[:, 1], c='g') plt.show()

        reduced = self.ss.transform(test_normalized)[0]
        return self.gpr.predict(np.atleast_2d(reduced))[0]


def plot_scores(clusters, scores, config, plot=True, save=False):
    """
    Assumed order is (r2_full, r2_local, mae_full, mae_local) for every row
    """
    fig, ax = plt.subplots(1, 1, figsize=config['figsize'])
    plt.title(config['title'])

    # right axis
    ax_right = ax.twinx()

    # if r2 then only mae reduction, otherwise viceversa
    if config['main'] == 'r2':
        reduction = (scores.T[2] - scores.T[3]) * 100 / scores.T[2]
        ax_right.set_ylabel('Mean absolute error reduction [%]', color='red')
        ax_right.set_ylim(bottom=config['mae_bottom'], top=config['mae_top'])

        ax.plot(clusters, scores.T[0], '--', c='blue',
                label='Global AS')  # full
        ax.plot(clusters, scores.T[1], '-o', c='blue',
                label='Local AS')  # local
        ax.set_ylim(bottom=config['r2_bottom'], top=config['r2_top'])
        ax.set_ylabel(r'R$^2$ score', color='blue')
        ax.set_yticks([0.1 * i for i in range(11)])

    if config['main'] == 'mae':
        reduction = (scores.T[1] - scores.T[0]) * 100 / scores.T[0]
        ax_right.set_ylabel(r'R$^2$ improvement [%]', color='red')
        ax_right.set_ylim(bottom=config['r2_bottom'], top=config['r2_top'])

        ax.plot(clusters, scores.T[2], '--', c='blue',
                label='Global AS')  # full
        ax.plot(clusters, scores.T[3], '-o', c='blue',
                label='Local AS')  # local
        ax.set_ylim(bottom=config['mae_bottom'], top=config['mae_top'])
        ax.set_ylabel('Mean absolute error', color='blue')

    ax_right.plot(clusters, reduction, '-o', c='red')
    ax_right.tick_params(axis='y', labelcolor='red')

    ax.set_xlabel('Number of clusters')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_xticks(clusters)

    ax.grid(linestyle='dotted')
    ax.legend(loc=config['loc'])
    plt.tight_layout()
    if save: plt.savefig(config['filename'])
    if plot: plt.show()
