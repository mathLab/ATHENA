"""
Module local Active Subspaces.

:References:

    - Romor, Francesco, Marco Tezzele, and Gianluigi Rozza.
      "A local approach to parameter space reduction for regression and classification tasks." arXiv preprint arXiv:2107.10867 (2021).

"""
from typing import List
from statistics import mean

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import linalg

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances_argmin

import GPy
from athena import Normalizer, ActiveSubspaces

import logging
# ! add in the script: logging.basicConfig(filename='divisive.log',
# level=logging.DEBUG)

_log = logging.getLogger('divisive')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

plt.rcParams.update({'font.size': 14})


class MaximumASDimensionReached(Exception):
    """Raised when all clusters have maximum AS dimension."""


class ClusterBase():
    """Local Active Subspaces clustering Base class.
    """
    def __init__(self, config):
        if config.get('inputs', None) or config.get('outputs', None):
            raise TypeError('inputs or outputs argument is None')

        self.inputs = config['inputs']
        self.outputs = config['outputs'].squeeze()
        self.gradients = config['gradients'] if config.get(
            'gradients', None) is not None else None

        self.as_dim = config['as_dim']
        self.method = 'exact' if config.get('gradients',
                                            None) is not None else 'local'

        self.n_clusters = config['n_clusters']
        self.random_state = config['random_state']

        self.clustering = None
        self.labels = None  # labels start from 0
        self.unique_labels = None

        self.full_as = None
        self.full_gpr = None
        self.local_ass = None
        self.local_gprs = None

        self.inputs_test = config['inputs_test'] if config.get(
            'inputs_test', None) is not None else self.inputs
        self.outputs_test = config['outputs_test'] if config.get(
            'outputs_test', None) is not None else self.outputs

        self.inputs_val = config['inputs_val'] if config.get(
            'inputs_val', None) is not None else self.inputs
        self.outputs_val = config['outputs_val'] if config.get(
            'outputs_val', None) is not None else self.outputs

        if len(self.outputs.shape) == 2:
            self.outputs_dim = self.outputs.shape[1]
        else:
            self.outputs_dim = 1

    def fit(self):
        """
        """
        self._fit_global()
        self._fit_clustering()
        self._fit_reduction()
        self._fit_regressions()

    def _fit_clustering(self):
        """Init unique_labels, labels, method predict"""
        raise NotImplementedError(
            'Subclass must implement abstract method {}._fit_clustering'.format(
                self.__class__.__name__))

    def _fit_global(self, plot=False):
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

        scores = np.array([r2_full, r2_local, mae_full, mae_local])
        return scores

    # TODO plot utilities for dimenions greater than 2
    def plot_clusters(self, save=False, title='2d_clusters'):
        assert self.inputs.shape[1] == 2
        plt.scatter(self.inputs[:, 0], self.inputs[:, 1], c=self.labels)
        plt.grid(linestyle='dotted')
        plt.tight_layout()
        if save: plt.savefig('{}.pdf'.format(title))
        plt.show()


class DecisionTreeAS(ClusterBase):
    def __init__(self, inputs, outputs, gradients, config):
        super().__init__(inputs, outputs, gradients, config)

    def _fit_clustering(self):
        # bins = np.linspace(np.amin(self.outputs), np.amax(self.outputs),
        # self.n_clusters+1) self.labels = np.digitize(self.outputs, bins)
        # # include the maximum in the last bin
        # self.labels[self.labels == self.n_clusters+1] = self.n_clusters
        # # shift classes starting from 0
        # self.labels -= 1

        clsuters_output = KMeans(n_clusters=self.n_clusters,
                                 random_state=self.random_state)
        clsuters_output.fit(self.outputs.reshape(-1, 1))

        self.labels = clsuters_output.labels_

        self.unique_labels = list(set(self.labels))

        self.clustering = DecisionTreeClassifier().fit(self.inputs,
                                                       self.labels)
        # r = export_text(self.clustering) print(r)


class KMeansAS(ClusterBase):
    """"""
    def __init__(self, config):
        super(KMeansAS).__init__(config)

    def _fit_clustering(self):
        self.clustering = KMeans(n_clusters=self.n_clusters,
                                 random_state=self.random_state)
        self.clustering.fit(self.inputs)

        self.labels = self.clustering.labels_
        self.unique_labels = list(set(self.labels))
        # self.centers = self.clustering.cluster_centers_

        print('Number of clusters: %d' % self.n_clusters)
        silhouette_ = silhouette_score(self.inputs, self.labels)
        print('Silhouette Coefficient: %0.3f' % silhouette_)


class KMedoidsAS(ClusterBase):
    def __init__(self, config):
        super(KMedoidsAS).__init__(config)

    def _fit_clustering(self):
        self.clustering = KMedoids(metric=self.as_metric,
                                   n_clusters=self.n_clusters,
                                   random_state=self.random_state)
        self.clustering.fit(self.inputs)

        self.labels = self.clustering.labels_
        self.unique_labels = list(set(self.labels))
        # self.centers = self.clustering.cluster_centers_

        silhouette_ = silhouette_score(self.inputs, self.labels)
        print('Number of clusters: %d' % self.n_clusters)
        # print('Silhouette Coefficient: %0.3f' % silhouette_)

    def as_metric(self, X, Y):
        return np.linalg.norm(
            self.full_as.evects.T.dot(X - Y) * self.full_as.evals)


class KmedoidsCustom(KMedoids):
    """Custom overloaded predict method from scikit. It is needed when the
    training metric is supervised (depending on input, output and gradients
    information) but the classification metric it is not (depending only on
    inputs)."""
    def __init__(self, *args, predict_metric=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_metric = predict_metric

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed' New data to
            predict.

        Returns
        -------
        labels : array, shape = (n_query,) Index of the cluster each sample
            belongs to.
        """
        X = check_array(X, accept_sparse=["csr", "csc"])

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return np.argmin(X[:, self.medoid_indices_], axis=1)
        else:
            check_is_fitted(self, "cluster_centers_")

            # Return data points to clusters based on which cluster assignment
            # yields the smallest distance
            input_dim = X.shape[1]

            return pairwise_distances_argmin(
                X,
                Y=self.cluster_centers_[:, :input_dim],
                metric=self.predict_metric)


class Divisive(ClusterBase):
    def __init__(self, inputs, outputs, config, gradients=None):
        """TODO check states logic.
        1. 2 and 4 are exlusives
        2. 0 and 4 have the same role
        3. if 4 and 1 are present, 4 has priority over 1

        :param dict config: 'normalization': uniform, gaussian, root 'metric':
            as, CI, euclidean, kmeans 'refinement_criterion': global, mean,
            force 'as_dim_criterion': spectral_gap, residual
        """
        super().__init__(inputs, outputs, config, gradients)
        self.max_clusters = config['max_clusters']
        self.max_red_dim = config['max_red_dim']
        self.max_children = config['max_children']
        self.min_children = config['min_children']
        self.min_local = config['min_local']
        self.score_tolerance = config['score_tolerance']
        self.normalization = config['normalization']  # uniform, gaussian, root
        self.metric = config['metric']  # as, CI, euclidean
        self.dim_cut = config['dim_cut']
        self.refinement_criterion = config[
            'refinement_criterion']  #global, mean, force
        self.as_dim_criterion = config[
            'as_dim_criterion']  # spectral_gap, residual
        self.minimum_score = config['minimum_score'] if 1 > config[
            'minimum_score'] > 0 else None  # it increases the cluster dimension until the score threshold is reached
        self.max_dim_refine_further = config[
            'max_dim_refine_further'] if config[
                'max_dim_refine_further'] is not None else self.inputs.shape[1]
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

    def fit(self):
        self._fit_global()
        scores, leaves_dimensions_list = self._fit_clustering()
        return scores, leaves_dimensions_list

    def _fit_clustering(self):
        self.root = DivisiveNode(None, np.arange(self.inputs.shape[0]),
                                 np.arange(self.inputs_val.shape[0]), self)

        self.total_clusters += 1

        # check if without clustering the tolerance is achieved
        if self.root.score > self.score_tolerance:
            self.root.state = 2
            state = {6}
            _log.debug("No clustering")
        else:
            self.root.state = 4
            children_fifo = [self.root]
            _log.debug("Add root")

            # continue until state 3 is reached or children_fifo is empty TODO
            # fix logical mistakes in the use states
            while (children_fifo != []):
                _log.debug("Befor pop total queue length is: " +
                           str(len(children_fifo)) + " total clusters: " +
                           str(self.total_clusters))

                node = children_fifo.pop(0)
                state, children = node.refine_cluster()

                _log.debug("After pop " + str(state) +
                           " and len children list added is " +
                           str(len(children)) + " total clusters: " +
                           str(self.total_clusters))

                if 3 in state:
                    break
                elif 7 in state:
                    continue
                elif 4 in state:
                    self.total_clusters += len(children) - 1
                    for child in children:
                        if child.state == 4:
                            children_fifo.append(child)
                elif 2 in state:
                    self.total_clusters += len(children) - 1
                elif any(x in state for x in [1, 5]):
                    continue

        print(
            "Hierarchical divisive clustering completed with state {}".format(
                state))
        print("Max clusters: ", self.max_clusters)
        # self._print_state()
        n_leaves, leaves_dim, _ = self._print_leaves_score()
        print("n_leaves: ", n_leaves)
        # self.plot_clusters()
        res = self.compute_scores(self.inputs_test, self.outputs_test)
        print("Test score: ", res)

        if self.minimum_score is not None:
            self.refine_further(self.minimum_score)

        return res, leaves_dim

    def refine_one_step(self):
        class LeafUpdate(object):
            def __init__(self):
                self.score = 0
                self.leaves_list = []

            def __call__(self, node):
                self.score = max(self.score, node.score)
                self.leaves_list.append(node)

        def void_func(*args, **kwargs):
            pass

        leaf_update = LeafUpdate()
        self.breadth_first_search(leaf_update, void_func)
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

            # ActiveSubspaces clas returns throws ValueError when the dimension
            # of the active subspace is equal to the dimension of the whole
            # domain try: node_to_update.ss._partition() except ValueError:
            # node_to_update.ss.W1 = node_to_update.ss.evects
            # node_to_update.ss.W2 = None

            # _log.debug("reduced dim inside refine_further loop is: " +
            #            str(node_to_update.r_dim))

            # t_inp_normalized = node_to_update.normalizer(node_to_update.t_ind,
            #     node_to_update.divisive.inputs_val,
            #     node_to_update.divisive.outputs_val)[0]
            # # re-optimize gpr
            # node_to_update._gpr = None

            # # property decorator on node_to_update.gpr is called
            # local_pred =
            #     node_to_update.gpr.predict(node_to_update.ss.transform(t_inp_normalized)[0])[0]
            #     node_to_update.score =
            #     r2_score(node_to_update.divisive.outputs_val[node_to_update.t_ind],
            #     local_pred, multioutput='uniform_average')

    def refine_further(self, minimum_score):
        print("Increasing the as dimension when possible.")

        class CallRefine(object):
            def __init__(self, minimum_core):
                self.min = minimum_score

            def __call__(self, node):
                node.refine_further(self.min)

        def void_func(*args, **kwargs):
            pass

        call_refine = CallRefine(minimum_score)
        self.breadth_first_search(call_refine, void_func)
        print("Finished increasing as dimension when possible.")

        n_leaves, leaves_dim, _ = self._print_leaves_score()
        print("n_leaves: ", n_leaves)

        # self.plot_clusters()

        res = self.compute_scores(self.inputs_test, self.outputs_test)
        print("Test score: ", res)

    def _print_state(self):
        children_fifo = [self.root]
        while (children_fifo != []):
            node = children_fifo.pop(0)
            print(node.state, " : ", self.state_d[str(node.state)])
            children_fifo.extend(node.children)

    def _print_leaves_score(self):
        children_fifo = [self.root]
        n_leaves = 0
        leaves_dim = []
        n_elems = []
        while (children_fifo != []):
            node = children_fifo.pop(0)
            if node.children == []:
                n_leaves += 1
                print("score is: {:.2f}".format(node.score), " r_dim is: ",
                      node.r_dim, " state is: ", node.state, " n_elem : ",
                      node.ind.shape[0])
                leaves_dim.append(node.r_dim)
                n_elems.append(node.ind.shape[0])
            children_fifo.extend(node.children)
        return n_leaves, leaves_dim, n_elems

    def _print_state_debug(self):
        children_fifo = [self.root]
        while (children_fifo != []):
            node = children_fifo.pop(0)
            _log.debug(str(node.state) + " : " + self.state_d[str(node.state)])
            children_fifo.extend(node.children)

    def assign_leaf_labels(self):
        class LeafLabels(object):
            def __init__(self):
                self.labels_counter = 0

            def __call__(self, node):
                node.leaf_label = self.labels_counter
                self.labels_counter += 1

        def void_func(*args, **kwargs):
            pass

        leaf_labels = LeafLabels()
        self.breadth_first_search(leaf_labels, void_func)

    def reset_gprs(self):
        def reset_gpr(node):
            node.gpr = None
            node.ss = None
            pass

        def void_func(*args, **kwargs):
            pass

        self.breadth_first_search(reset_gpr, void_func, reset_gpr)

    def plot_clusters(self, with_test=False, save_data=True):
        children_fifo = [self.root]
        # it = -1
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(111)
        colors = cm.rainbow(np.linspace(0, 1, self.max_clusters))

        self.assign_leaf_labels()
        if save_data: save_dat = None

        while (children_fifo != []):
            node = children_fifo.pop(0)
            if node.children == []:
                # it += 1
                data = self.inputs[node.ind]
                ax1.scatter(data[:, 0],
                            data[:, 1],
                            c=colors[node.leaf_label],
                            label="r2={:.2f}, as_dim={}".format(
                                node.score, node.r_dim))
                if save_data:
                    col_ones = np.ones((data.shape[0], 1))
                    dats = np.hstack(
                        (data, node.score * col_ones, node.r_dim * col_ones,
                         node.leaf_label * col_ones))
                    if save_dat is None:
                        save_dat = dats
                    else:
                        save_dat = np.vstack((save_dat, dats))

            children_fifo.extend(node.children)

        print("Saved data with shape: ", save_dat.shape)
        np.save("clusters_r2_asdim_labels.npy", save_dat)

        if with_test:
            predictions = np.zeros(
                (self.inputs_test.shape[0], self.inputs_test.shape[1] + 1))
            for i_test in range(self.inputs_test.shape[0]):
                children_fifo = [self.root]
                while (children_fifo != []):
                    node = children_fifo.pop(0)
                    if node.children == []:
                        predictions[i_test, :-1] = self.inputs_test[i_test]
                        predictions[i_test, -1] = node.leaf_label
                    else:
                        test_normalized = node.normalizer(
                            inputs=self.inputs_test[i_test])[0]
                        idx = node.cluster.predict(
                            np.atleast_2d(test_normalized))[0]
                        children_fifo.extend([node.children[idx]])
            ax1.scatter(predictions[:, 0],
                        predictions[:, 1],
                        c=[
                            colors[int(predictions[i, -1])]
                            for i in range(self.inputs_test.shape[0])
                        ],
                        alpha=0.5)

        plt.legend()
        plt.title("Divisive clustering, total clusters= {} ".format(
            self.total_clusters))
        plt.tight_layout()
        # TODO pretty plot + specific image name
        plt.savefig("clusters_divisive.png")
        plt.close()

    # overload
    def compute_scores(self, test_inputs, test_outputs, custom_score=None):
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
                    idx = node.cluster.predict(
                        np.atleast_2d(test_normalized))[0]

                    # TODO custom alternatives for classification idx =
                    # self._cluster_min_dist(test_inputs[i_test], node.children)
                    children_fifo.extend([node.children[idx]])

        r2_full = custom_score_(test_outputs, full_pred)
        r2_local = custom_score_(test_outputs, local_pred)
        mae_full = mean_absolute_error(test_outputs, full_pred)
        mae_local = mean_absolute_error(test_outputs, local_pred)

        scores = np.array([r2_full, r2_local, mae_full, mae_local])
        return scores

    def predict(self, test_inputs):
        predictions = np.zeros(test_inputs.shape[0])
        for i_test in range(test_inputs.shape[0]):
            children_fifo = [self.root]
            while (children_fifo != []):
                node = children_fifo.pop(0)
                if node.children == []:
                    predictions[i_test] = node.predict(test_inputs[i_test])
                else:
                    test_normalized = node.normalizer(
                        inputs=test_inputs[i_test])[0]
                    idx = node.cluster.predict(
                        np.atleast_2d(test_normalized))[0]

                    # TODO custom alternatives for classification idx =
                    # self._cluster_min_dist(test_inputs[i_test], node.children)

                    # DEBUG fig = plt.figure() ax1 = fig.add_subplot(111) colors
                    # = cm.rainbow(np.linspace(0, 1, len(node.children) + 1))
                    # ax1.scatter(test_inputs[i_test][0],
                    # test_inputs[i_test][1], c=colors[0], label="test")

                    # for i, node in enumerate(node.children):
                    #     ax1.scatter(self.inputs[node.ind][:, 0],
                    #     self.inputs[node.ind][:, 1], c=colors[i + 1],
                    #     label=str(i)) plt.title(str(idx)) plt.legend()
                    #     plt.show()

                    children_fifo.extend([node.children[idx]])

        return predictions

    def _cluster_min_dist(self, test, clusters: List):
        group_list = []
        for node in clusters:
            group_list.append(self.inputs[node.ind])
        return self.set_min_dist(test, group_list)

    @staticmethod
    def set_min_dist(test, groups: List):
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


class DivisiveNode():
    def __init__(self, parent, node_indexes, val_indexes, divisive_obj):
        """A DivisiveNode is defined by the indexes of the triplets (inputs,
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
        self.ind = node_indexes
        self.t_ind = val_indexes
        self.r_dim = None
        self.score = None
        self.children = []
        self.divisive = divisive_obj
        self.state = None
        self._gpr = None
        self._ss = None
        self.cluster = None
        self.type = "root" if self.parent is None else self.divisive.normalization
        _log.debug("Validation set length: {}".format(
            self.divisive.inputs_val[self.t_ind].shape))

        ####### Estabilish invariants r_dim and score
        self.normalizer = self.NormalizeDivisive(self.type, self.ind,
                                                 self.divisive.inputs)

        inp_normalized, out, grad_normalized = self.normalizer(
            self.ind, self.divisive.inputs, self.divisive.outputs,
            self.divisive.gradients)

        # 1. r_dim is between 1 and max_red_dim
        ass = self.compute_as(inp_normalized, out, grad_normalized)
        # save essential info of AS without saving the whole of it
        self.evects = ass.evects
        self.evals = ass.evals / np.sum(ass.evals)

        # sufficiency summary for debug if out.shape[0] == 1:
        # ass.plot_sufficient_summary(inp_normalized, out) else:
        # ass.plot_sufficient_summary(inp_normalized, out[:, 10])

        self.cum_evals = np.array(
            [np.sum(self.evals[:i + 1]) for i in range(self.evals.shape[0])])

        if self.divisive.as_dim_criterion == "residual":
            self.r_dim = self.as_dim(self.divisive.max_red_dim,
                                     threshold=self.divisive.dim_cut)
        elif self.divisive.as_dim_criterion == "spectral_gap":
            self.r_dim = self.as_dim_gap(self.divisive.max_red_dim)
        else:
            raise ValueError(
                "The criterion for active subspace dimension criterion must be among: residual, spectral_gap."
            )

        # change as dimension
        ass.dim = self.r_dim
        ass._partition()
        _log.debug("reduced dim is: " + str(self.r_dim))

        # 2. the score is above the score_tolerance
        gpr = GPy.models.GPRegression(ass.transform(inp_normalized)[0],
                                      out.reshape(inp_normalized.shape[0], -1),
                                      normalizer=False,
                                      noise_var=1)
        gpr.optimize_restarts(10, verbose=False)

        t_inp_normalized = self.normalizer(self.t_ind,
                                           self.divisive.inputs_val,
                                           self.divisive.outputs_val)[0]

        local_pred = gpr.predict(ass.transform(t_inp_normalized)[0])[0]
        self.score = np.mean(
            np.clip(
                r2_score(self.divisive.outputs_val[self.t_ind],
                         local_pred,
                         multioutput='raw_values'), 0, 1))

        _log.debug("r2 score is: " + str(self.score))

    def refine_further(self, minimum_score):
        _log.debug("r2 score inside refine_further is: " + str(self.score))

        # create self.ss if it does not exist, thanks to property decorator
        while (self.score < minimum_score
               and self.ss.dim + 1 <= self.divisive.max_dim_refine_further):
            # change as dimension
            self.ss.dim += 1
            self.r_dim += 1

            # ActiveSubspaces clas returns throws ValueError when the dimension
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
                                               self.divisive.inputs_val,
                                               self.divisive.outputs_val)[0]
            # re-optimize gpr
            self._gpr = None

            # property decorator on self.gpr is called
            local_pred = self.gpr.predict(
                self.ss.transform(t_inp_normalized)[0])[0]
            self.score = r2_score(self.divisive.outputs_val[self.t_ind],
                                  local_pred,
                                  multioutput='uniform_average')

            _log.debug("r2 score inside while cycle is: " + str(self.score))

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
                self.sqrt_cov = linalg.sqrtm(cov)
                self.inv_sqrt_cov = linalg.inv(self.sqrt_cov)

            elif self.type == 'uniform':
                _log.debug("Uniform normalization")
                inp = inputs[ind]
                lb = np.min(inp, axis=0, keepdims=False)
                ub = np.max(inp, axis=0, keepdims=False)
                self.nor = Normalizer(lb, ub)

            elif self.type == 'root':
                _log.debug("Without normalization")
                pass
            else:
                raise ValueError(
                    "A proper flag was not passed to normalize class: " +
                    self.type)

        def __call__(self,
                     ind=None,
                     inputs=None,
                     outputs=None,
                     gradients=None):
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
        # TODO check type and behaviour e.g. return only if state is 2 or 4
        if self._gpr is None:
            inp_normalized, out, _ = self.normalizer(self.ind,
                                                     self.divisive.inputs,
                                                     self.divisive.outputs)

            gpr = GPy.models.GPRegression(self.ss.transform(inp_normalized)[0],
                                          out.reshape(inp_normalized.shape[0],
                                                      -1),
                                          normalizer=False)
            gpr.optimize_restarts(20, verbose=False)
            self._gpr = gpr
            return gpr
        else:
            return self._gpr

    @gpr.setter
    def gpr(self, value):
        self._gpr = value

    @gpr.deleter  # lazy property
    def gpr(self):
        del self._gpr

    @property  # lazy property
    def ss(self):
        # TODO check type and behaviour e.g. return only if state is 2 or 4
        if self._ss is None:
            inp_normalized, out, grad_normalized = self.normalizer(
                self.ind, self.divisive.inputs, self.divisive.outputs,
                self.divisive.gradients)

            if grad_normalized is not None:
                ss = ActiveSubspaces(dim=self.r_dim)
                ss.fit(gradients=grad_normalized)
            else:
                ss = ActiveSubspaces(dim=self.r_dim, method='local')
                ss.fit(inputs=inp_normalized, outputs=out)

            self._ss = ss
            return ss
        else:
            return self._ss

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

        max_total_clusters = self.divisive.max_clusters
        min_clusters = self.divisive.min_children
        max_clusters = self.divisive.max_children
        min_local = self.divisive.min_local
        score_thre = self.divisive.score_tolerance
        priority = self.divisive.refinement_criterion

        state = {0}
        res_max = self.score  # best score among choice of n_clusters
        cluster_ = None

        _log.debug("Start refining from {} to {} clusters".format(
            min_clusters, max_clusters))
        for n_clusters in range(min_clusters, max_clusters + 1):

            _log.debug("Refine cluster with " + str(n_clusters) +
                       " clusters. Total would be: " +
                       str(self.divisive.total_clusters + n_clusters - 1) +
                       " \nRes max: {:.3f}".format(res_max) + " N_children: " +
                       str(len(self.children)))

            # check if max_clusters would be reached
            if max_total_clusters < self.divisive.total_clusters + n_clusters - 1:
                # state contains 3 only if during the for loop no refinement is
                # chosen
                if max_total_clusters == self.divisive.total_clusters + len(
                        self.children) - 1:
                    state.add(3)
                elif self.children == []:
                    state.add(7)

                _log.debug("Max clusters violation : " + str(state) +
                           " and list length " + str(len(self.children)))
                return state, self.children

            # check if clustering is possible
            if self.ind.shape[0] < self.divisive.total_clusters + n_clusters:
                state.add(5)
                _log.debug("Refine returns 5 : " + str(state) +
                           " and list length " + str(len(self.children)))
                return state, self.children

            # cluster with metric of choice
            if self.divisive.metric == 'as':
                _log.debug("AS metric")
                cluster_ = KMedoids(metric=self.as_metric,
                                    n_clusters=n_clusters,
                                    random_state=self.divisive.random_state)

                local_inp = self.normalizer(self.ind, self.divisive.inputs,
                                            self.divisive.outputs)[0]
            elif self.divisive.metric == 'kmeans':
                _log.debug("kmeans")
                cluster_ = KMeans(n_clusters=n_clusters,
                                  random_state=self.divisive.random_state)

                local_inp = self.normalizer(self.ind, self.divisive.inputs,
                                            self.divisive.outputs)[0]
            # C1 norm with outputs included
            elif self.divisive.metric == 'C1':
                _log.debug("C1 metric")
                cluster_ = KMedoids(n_clusters=n_clusters,
                                    random_state=self.divisive.random_state)

                inp_normalized, out, grad_normalized = self.normalizer(
                    self.ind, self.divisive.inputs, self.divisive.outputs,
                    self.divisive.gradients)
                local_inp = np.hstack(
                    (inp_normalized, out.reshape(-1, 1), grad_normalized))

            elif self.divisive.metric == 'euclidean':
                _log.debug("Euclidean metric")
                cluster_ = KMedoids(n_clusters=n_clusters,
                                    random_state=self.divisive.random_state)

                local_inp = self.normalizer(self.ind, self.divisive.inputs,
                                            self.divisive.outputs)[0]
            else:
                raise ValueError("Wrong metric value: " +
                                 self.divisive.metric +
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
                self.divisive.inputs_val[self.t_ind])
            for label in unique_labels:
                mask = labels == label
                mask_val = val_indexes == label
                m_ind = self.ind[mask]
                # TODO change when an independent validation set is passed

                m_val = self.t_ind[mask_val]
                node = DivisiveNode(self, m_ind, m_ind, self.divisive)
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
        return state, self.children

    def global_children_priority(self, children, res_max, res, children_states,
                                 cluster, state):
        """ The refinement is chosen if the r2 score of the divisive node is
        bested by the r2 score on the whole predictions of the children
        nodes."""

        # costly r2 evaluation with gpr
        pred = np.zeros(self.divisive.inputs.shape[0])
        for node in children:
            pred[node.ind] = node.predict(
                self.divisive.inputs[node.ind]).squeeze()

        pred = pred[self.ind]
        full_children_r2 = r2_score(self.divisive.outputs[self.ind],
                                    pred,
                                    multioutput='uniform_average')

        # full_children_r2 = 1 -
        #     np.sum(np.array([node.t_ind.shape[0]*(1-node.score) *
        #     np.var(node.divisive.outputs_val[node.t_ind]) for node in
        #     children])) / (np.var(self.divisive.outputs_val[self.t_ind]) *
        #     self.t_ind.shape[0])

        # in practice it may be useful to multply full_children_r2 by a value >
        # 1
        if res_max < full_children_r2:
            _log.debug("Best score bested: {} < {}".format(
                res_max, full_children_r2))
            res_max = full_children_r2
            self.children = children
            self.cluster = cluster
        else:
            _log.debug("Best score not bested: {} < {}".format(
                res_max, full_children_r2))

        if res_max >= self.divisive.score_tolerance:
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
        res_ = mean(res)
        max_state = max(children_states)

        # stop refinement: tolerance achieved for the first time w.r.t.
        # n_clusters loop
        if 2 == max_state:
            _log.debug("Leaf reached")
            self.children = children
            self.cluster = cluster
            return_value = "break"

        elif 4 == max_state:
            _log.debug("Leaf not reached, children states" +
                       str(children_states))

            # if the overall score is bested change children DEBUG changed to
            # 1.01 instead of * 1
            if res_max < res_:
                _log.debug("Bested: " + str(res_max) + " < " + str(res_))
                res_max = res_
                self.children = children
                self.cluster = cluster
            else:
                _log.debug("Not bested: " + str(res_max) + "  " + str(res_))

            return_value = "continue"

        return return_value, res_max, state

    def force_refinement_priority(self, children, res_max, res,
                                  children_states, cluster, state):
        """The refinement is chosen if the local r2 scores are above the
        threshold or if res_max is below the average of the local r2 scores. It
        can NOT happen that refinement is not performed because the parent score
        is the best.
        """
        res_ = mean(res)
        max_state = max(children_states)

        # stop refinement: tolerance achieved for the first time w.r.t.
        # n_clusters loop
        if 2 == max_state:
            _log.debug("Leaf reached")
            self.children = children
            self.cluster = cluster
            return_value = "break"

        elif 4 == max_state:
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
                _log.debug("Bested: " + str(res_max) + " < " + str(res_))
                res_max = res_
                self.children = children
                self.cluster = cluster
            else:
                _log.debug("Not Bested: " + str(res_max) + " > " + str(res_))

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
        _log.debug("Spectral gaps: " + str(max_gap))
        r_dim = 1
        i = 1
        while (i < max_dim):
            gap = self.evals[i] - self.evals[i + 1]
            _log.debug("Spectral gaps: " + str(gap))
            if gap > max_gap:
                max_gap = gap
                r_dim = i + 1
            i += 1
        _log.debug("Returned: " + str(r_dim))
        return r_dim

    def as_dim(self, max_dim, threshold=0.95):
        r_dim = 0
        # assert max_dim < self.evals.shape[0], "max_red_dim cannot be equal or
        #     grater than the input dimension"

        r_dim = 1
        while (r_dim <= max_dim):
            _log.debug(
                "Spectral cumulatives: {:.6f}, threshold: {:.6f}, red dim {}".
                format(self.cum_evals[r_dim - 1], threshold, r_dim))

            if self.cum_evals[r_dim - 1] >= threshold:
                _log.debug("break {}".format(r_dim))
                break
            r_dim += 1
        else:
            _log.debug("else {}".format(r_dim))
            r_dim -= 1
        _log.debug("Returned: " + str(r_dim))
        return r_dim

    def as_metric(self, X, Y):
        """Metrics that agglomerate clusters in the direction trasversal to the
        active subspace."""
        return np.linalg.norm(self.evects.T.dot(X - Y) * self.evals)

    def predict(self, test):
        test_normalized = self.normalizer(inputs=test)[0]

        # DEBUG inp = self.divisive.inputs[self.ind] inp_normalized = (inp -
        # self.mean).dot(self.inv_sqrt_cov) fig = plt.figure() ax1 =
        # fig.add_subplot(111)
        # # ax1.scatter(test_normalized[:, 0], test_normalized[:, 1], c='r')
        # # ax1.scatter(inp_normalized[:, 0], inp_normalized[:, 1], c='b')
        # ax1.scatter(self.mean[:, 0], self.mean[:, 1], c='g') plt.show()

        reduced = self.ss.transform(test_normalized)[0]
        return self.gpr.predict(np.atleast_2d(reduced))[0]


def plot_scores(clusters, scores, config):
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
    plt.savefig(config['filename'])
    # plt.show()
