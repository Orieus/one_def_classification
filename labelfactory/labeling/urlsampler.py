# -*- coding: utf-8 -*-
import numpy as np
import ipdb
import sys

# My own libraries
from labelfactory.activelearning.activelearner import ActiveLearner


class URLsampler(object):

    """ This class provides a method to take a given number of urls from a list,
        based on a given active learning algorithm.

    """

    def __init__(self, ref_class, preds, labels, urls, markers, weights,
                 alphabet, type_al, alth, p_al, p_relabel, tourneysize):

        """
        This method initialize the sampler object. As part of this process, it
        creates the AL objects required for the sample generation.

        :Attributes:

            :ref_class: Reference category for the sample
            :preds:     Dictionary containing, for each class and each wid, the
                        current predictions (thougn only predictions about the
                        ref_class are used)
            :labels:    Dict containing, for each wid, the new labels.
            :urls:      Dict containing, for each wid, its url
            :markers:   Dict containing, for each wid, its marker. An URL
                        obtained by random sampling has zero marker. If it has
                        been obtained through active learning, its marker is 1.
            :weights:   Dict containing, for each wid, its weight. Weights must
                        be used in sample averages to correct the bias of the
                        active learning algorithm.
            :alphabet:  Dict containing the symbol of each possible class.
            :type_al:   Type of active learning algorithm.
            :alth:      Active learning threshold
            :p_al:      Probability of active learning (wrt random sampling)
            :p_relabel: Probability of resampling
            :tourneysize: Size of tourney (only for type_al = 'tourney')
            :parentcat: Dict containing, for each class, its parent
        """

        ####################
        # Initial attributes

        # Attributes taken from the argument
        self.preds = preds
        self.labels = labels
        self.urls = urls
        self.markers = markers
        self.weights = weights
        self.ref_class = ref_class

        self._yes = alphabet['yes']
        self._no = alphabet['no']
        self._unknown = alphabet['unknown']
        self._error = alphabet['error']

        self.p_al = p_al

        ######################
        # Create the AL object
        self.alearn = ActiveLearner(type_al, p_relabel, alth, p_al,
                                    tourneysize)

    def get_urls_batch(self, max_urls=10,):

        """ Loads at most max_urls from the dataset.

        :Attributes:

            :max_urls:   max number of urls to label

        Returns:
            :newurls:    A list of urls to label
            :newwids:    Its corresponding wids
            :newqueries: A dict containing some other info about each new url
                         to label: the url (again), and the marker and relabel
                         indicators
            Also, self.y, self.weights and self.markers are updated.

        Note: newurls, newwids and newqueries arrays have the same size, equal
              to the number of new urls to label. But note that the size of
              weights_new is equal to the size of the whole dataset loaded into
              the sampler, because each sampled url changes all weights.
        """

        # Read all web identifiers (wids)
        # I preferer wids = [*self.urls] here, but I need a python-2-compatible
        # command...
        wids = list(self.urls)
        num_urls = len(self.urls)

        # Initialize
        scores = np.zeros(num_urls)
        weights = np.zeros(num_urls)
        self.y = np.zeros(num_urls)  # Labels for active learning

        # The active learning toolbox uses labels 1 and -1 for the positive and
        # negative class, and 0 for the unlabeled samples. This dictionary is
        # used for the conversion from the input data alphabet
        AL_alphabet = {self._yes: 1, self._no: -1, self._unknown: 0,
                       self._error: 0}

        # Read current scores and labels
        for i, wi in enumerate(wids):

            # Read i-th score
            pred_i = self.preds[self.ref_class][wi]
            if pred_i is not None and not np.isnan(pred_i):
                # WARNING:
                # If there is no prediction, it is set to 0. This is not
                # actually the best option. Another alternative should be
                # implemented.
                # print "Assigning zero score when pred is None"
                scores[i] = pred_i

            if wi in self.weights:

                # Read i-th weight
                weight_i = self.weights[wi]

                # Correct "None" weights
                if weight_i is None or np.isnan(weight_i):

                    if (self.labels[self.ref_class][wi]
                            in {self._yes, self._no}):
                        # If weight_i is None, it is set to 1/num_urls.
                        # That is, we take the worst-case assumption that
                        # the sample was selected with probability 1.
                        weights[i] = 1.0 / num_urls

                    # Note that if the label is not _yes or _no, the weight
                    # takes the default value, zero.
                    # (Actually, this value is not relevant, since the weigh
                    # of an unlabeled sample is never used. Setting it to zero
                    # is useful to sum all active weights with sum(weights)
                    # without interference from the unlabeled sample weights).

                else:
                    weights[i] = weight_i

            # Read and convert labels to the alphabet used by the AL
            if wi in self.labels[self.ref_class]:
                if self.labels[self.ref_class][wi] in AL_alphabet:
                    self.y[i] = AL_alphabet[self.labels[self.ref_class][wi]]
                elif type(self.labels[self.ref_class][wi]) == int:
                    # This is for the multilabel case. All positive labels
                    # are mapped to 1.
                    self.y[i] = np.sign(self.labels[self.ref_class][wi])

        print('Loading scores for {0} urls'.format(num_urls))
        print('Loading {} labels'.format(len(self.labels[self.ref_class])))

        ##################
        # Active learning:
        indexes, relabels, markers, weights_new = self.alearn.get_queries(
            max_urls, self.y, scores, weights)

        # Return sublist of urls acording to AL indexes
        newwids = [wids[i] for i in indexes]
        newurls = [self.urls[w] for w in newwids]

        newqueries = {}

        n = len(indexes)
        for k in range(n):
            newqueries[newwids[k]] = {'url': newurls[k],
                                      'marker': markers[k],
                                      'relabel': relabels[k]}

        # Update sampler
        for k in range(n):
            self.markers[newwids[k]] = markers[k]
        for i, wi in enumerate(wids):
            self.weights[wi] = weights_new[i]

        # Print the four types of webs to label
        print("Obteniendo {} urls nuevas para etiquetar.".format(n))

        print("New labels. Random sampling: ")
        print([newwids[k] for k in range(n) if relabels[k] == 0 and
               markers[k] == 0])

        print("Re-labels. Random sampling: ")
        print([newwids[k] for k in range(n) if relabels[k] == 1 and
               markers[k] == 0])

        print("New labels. Active Learning: ")
        print([newwids[k] for k in range(n) if relabels[k] == 0 and
               markers[k] == 1])

        print("Re-labels. Active Learning: ")
        print([newwids[k] for k in range(n) if relabels[k] == 1 and
               markers[k] == 1])

        return newurls, newwids, newqueries

    def get_single_url(self, target_wid):

        """ Loads a single url from from the dataset.

        :Attributes:
            :target_wid: the wid of a single url to label.

        Returns:
            :newurls:    A list of urls to label
            :newwids:    Its corresponding wids
            :newqueries: A dict containing some other info about each new url
                         to label: the url (again), and the marker and relabel
                         indicators
        """

        # Make sure that the target url has been previously labeled.
        # The code is not ready to label urls without a previous label,
        # because that would requiere to change the active learning weights
        if target_wid not in self.markers:
            sys.exit("The target url has no label. Labeling of a specific " +
                     "url should be done to revise existing labels only")

        newwids = [target_wid]
        newurls = [self.urls[target_wid]]
        newqueries = {}
        newqueries[target_wid] = {'url': self.urls[target_wid],
                                  'marker': self.markers[target_wid],
                                  'relabel': 1}

        # Print the four types of webs to label
        print("Relabeling {0} ".format(newurls[0]))

        return newurls, newwids, newqueries

    def get_urls_RSAL(self, max_urls=10):

        """ Loads at most max_urls from the dataset.

            This is an old version of get_urls. The main difference between
            them is that, while get_urls_batch creates a unique AL object for
            both active learning (AL) and random sampling (RS), get_urls_RSAL
            uses one AL object for each kind of sampler. This way, the random
            sampler is fed with all samples that have not been previously
            labeled by RS. Thus, AL samples can be relabeld by the RS. The
            difference is statistically relevant:

            * Using get_urls_RSAL, those urls sampled with marker=0 are iid
              samples from the original distribution.
            * Using get_urls_batch, we can still take samples using RS (by
              stating self.p_AL < 1). However, is some samples have been
              previously obtainted by AL, the RS samples are not purely random,
              because they are biased by the AL algorith (which has removed
              samples from the unlabeled dataset in a non random way).
              However, ger_urls_batch returns a weight for each url, that can
              be used to correct the AL bias by averaging.

            For large datasets, if the proportion of labeled samples is small,
            the bias effect introduced by the AL can be ignored, and the RS
            samples obtained by get_urls_bath are amost purely random.

        :Attributes:

            :max_urls:   max number of urls to label

        Returns:
            :newurls:   A list of urls to label
            :newwids:   Its corresponding wids
            :newqueries: Its current labels

        """

        # Read all web identifiers (wids)
        wids = self.urls.keys()
        num_urls = len(self.urls)

        # Initialize
        scores = np.zeros(num_urls)
        self.y = np.zeros(num_urls)  # Labels for active learning
        self.z = np.zeros(num_urls)  # Labels for random sampling

        # Read current scores and labels
        for i, wi in enumerate(wids):

            pred_i = self.preds[self.ref_class][wi]

            if pred_i is None:
                # WARNING:
                # If there is no prediction, it is set to 0. This is not
                # actually the best option. Another alternative should be
                # implemented.
                # print "Assigning zero score when pred is None"
                scores[i] = 0
            else:
                scores[i] = pred_i

            # if wi in self.labels[self.ref_class]:
            self.y[i] = self.labels[self.ref_class][wi]

            # The current label set for random sampling includes only those
            # labels also obtained by random sampling.
            if self.markers[wi] == 0:
                if self.labels[self.ref_class][wi] == self._yes:
                    self.z[i] = 1
                else:
                    self.z[i] = -1

        print('Cargando scores sobre {} urls'.format(num_urls))
        print('Cargando {} etiquetas'.format(len(self.labels[self.ref_class])))

        # Feed scores and labels to the active learners to get some new queries
        # Number of queries to take by al (active learning) and by rs (random
        # sampling)
        n_queries_al = max_urls*self.p_al     # No. of queries by al
        n_queries_rs = max_urls-n_queries_al  # No. of queries by rs

        ##################
        # Active learning:
        indexes, relabels, markers, weights = self.alearn.get_queries(
            n_queries_al, self.y, scores)

        # Return sublist of urls acording to AL indexes
        # self.urls_AL = [self.all_urls[i] for i in indexes]
        newwids = [wids[i] for i in indexes]
        newurls = [self.urls[w] for w in newwids]

        newqueries = {}

        n = len(indexes)
        for k in range(n):
            newqueries[newwids[k]] = {'url': newurls[k],
                                      'marker': 1,
                                      'relabel': relabels[k],
                                      'weight': weights[k]}

        #################
        # Random sampling
        indexes0, relabels0, markers0, weights0 = \
            self.randomsampler.get_queries(n_queries_rs, self.z, scores)

        # Return sublist of urls acording to AL indexes
        # self.urls_AL = [self.all_urls[i] for i in indexes]
        wids_RS = [wids[i] for i in indexes0]
        urls_RS = [self.urls[w] for w in wids_RS]

        n0 = len(indexes0)
        for k in range(n0):
            newqueries[wids_RS[k]] = {'url': urls_RS[k],
                                      'marker': 0,
                                      'relabel': relabels0[k],
                                      'weight': weights0[k]}

        newurls.extend(urls_RS)
        newwids.extend(wids_RS)

        # Print the four types of webs to label
        print("Obteniendo {} urls nuevas para etiquetar.".format(n + n0))

        print("Etiquetas nuevas. Muestreo aleatorio: ")
        print([wids_RS[k] for k in range(n0) if relabels0[k] == 0])

        print("Re-etiquetas. Muestreo aleatorio: ")
        print([wids_RS[k] for k in range(n0) if relabels0[k] == 1])

        print("Etiquetas nuevas. Active Learning: ")
        print([newwids[k] for k in range(n) if relabels[k] == 0])

        print("Re-etiquetas. Active Learning: ")
        print([newwids[k] for k in range(n) if relabels[k] == 1])

        return newurls, newwids, newqueries
