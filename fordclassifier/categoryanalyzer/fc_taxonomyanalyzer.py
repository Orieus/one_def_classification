import pandas as pd
import numpy as np
import wikipedia
import os

from tabulate import tabulate

# Local imports
from fordclassifier.evaluator.evaluatorClass import Evaluator
from fordclassifier.corpusanalyzer.textproc import Lemmatizer

# Only for debugging
import pdb


class FCTaxonomyAnalyzer(object):
    """
    This class contains those methods used to process categorical from the
    corpus database used in the MINECO project (which containts abstracts of
    research projects in a particular DB structure).

    Thus, they are specific for the particular data structure used in this
    project.
    """

    def __init__(self, categories=None, cf=None, path2project=None,
                 f_struct_cls=None):

        self.categories = categories
        self.cf = cf
        self.path2project = path2project
        self.f_struct_cls = f_struct_cls

        return

    def compareLabels(self, df_labels, df_projs, label_field, sep=',',
                      cat_model='multilabel'):

        """
        Computes the confusion matrix for the input labels with respect to the
        true labels taken from the human labeling tool

        Args:
            label_field: Columns in DB 'proyectos' containing the labels to
                         evaluate. Available options are:
                            - Unesco2Ford
                            - FORDclassif
            sep: Separator used to separate categories in the DB
        """

        # Read manual labels
        refs = df_labels['Referencia'].tolist()
        df_projs = df_projs[df_projs['Referencia'].isin(refs)]

        # Join in a single dataframe the df with the manual labels and that
        # with the labels to be evaluated.
        # Only those projects with a manual label are taken into account.
        df2 = df_labels.join(
            df_projs.set_index('Referencia'), on='Referencia')

        # Remove samples without class labels
        idx = [i for i, x in enumerate(df2[label_field].values)
               if x is not None]
        man_labels0 = [df2['labelstr'].values[i] for i in idx]
        u2f_labels0 = [df2[label_field].values[i] for i in idx]

        n0 = len(refs) - len(u2f_labels0)
        if n0 > 0:
            print(('---- Warning: {0} samples have no assigned ' +
                   'categories.').format(n0))
            print('     They have been discarded for the computation of the ' +
                  'confusion matrix.')

        # Extract labels:
        man_labels1 = [x.split('++') for x in man_labels0]
        u2f_labels1 = [x.split(sep) for x in u2f_labels0]

        if cat_model == 'multiclass' and label_field == 'Unesco2Ford':
            # Take only items with a single label.
            print('---- NOTE: Only projects with a single u2f label will ' +
                  'be used to compute errors')
            idx = [i for i, x in enumerate(u2f_labels1) if len(x) == 1]
            man_labels1 = [man_labels1[i] for i in idx]
            u2f_labels1 = [u2f_labels1[i] for i in idx]

        # Filter projects that contain level 1-categories
        man_labs_filtered = []
        u2f_labs_filtered = []
        for n, x in enumerate(zip(man_labels1, u2f_labels1)):
            if (len([z for z in x[0] if z.endswith('_')]) == 0 and
                    len([z for z in x[1] if z.endswith('_')]) == 0):
                man_labs_filtered.append(x[0])
                u2f_labs_filtered.append(x[1])

        n_filtered = len(u2f_labels1) - len(u2f_labs_filtered)
        if n_filtered > 0:
            print(('---- Warning: {0} samples contain level-1 FORD ' +
                   'categories.').format(n_filtered))
            print('      They have been discarded for the computation of ' +
                  'the confusion matrix.')

        # Compute confusion matrices
        cats = [x for x in self.categories if not x.endswith('_')]

        fname = 'CONF_{}_'.format(label_field) + '_NoOrder'
        EV = Evaluator(self.path2project, self.f_struct_cls)

        Mconf_NoOrder = EV.compute_confusion_matrix_multilabel_v2(
            man_labs_filtered, u2f_labs_filtered, fname + '.pkl',
            sorted_categories=cats, order_sensitive=False, verbose=True)
        EV.draw_confusion_matrix(
            Mconf_NoOrder, fname + '.png', sorted_categories=cats)
        EV.draw_confusion_matrix(
            Mconf_NoOrder, fname + '_un.png', sorted_categories=cats,
            normalize=False)

        fname = 'CONF_{}_'.format(label_field) + '_Order'
        EV = Evaluator(self.path2project, self.f_struct_cls)
        Mconf_Order = EV.compute_confusion_matrix_multilabel_v2(
            man_labs_filtered, u2f_labs_filtered, fname + '.pkl',
            sorted_categories=cats, order_sensitive=True, verbose=True)
        EV.draw_confusion_matrix(
            Mconf_Order, fname + '.png', sorted_categories=cats)
        EV.draw_confusion_matrix(
            Mconf_Order, fname + '_un.png', sorted_categories=cats,
            normalize=False)

        df_ranked_abs, df_ranked_rel = EV.compute_sorted_errors(
            Mconf_Order, cats)
        df_ranked_abs = df_ranked_abs.rename(columns={'Clasif': label_field})
        df_ranked_rel = df_ranked_rel.rename(columns={'Clasif': label_field})

        cols = ['Cat. real', label_field, 'Err/total (%)', 'Error/cat (%)',
                'Peso muestral']

        tag = {'Unesco2Ford': 'u2f', 'FORDclassif': 'fc'}
        f_path = os.path.join(self.path2project, self.f_struct_cls['results'],
                              f'ranked_{tag[label_field]}_abs.xlsx')
        df_ranked_abs.to_excel(f_path)
        print("---- Ranked absolute errors:")
        print(df_ranked_abs.head())

        f_path = os.path.join(self.path2project, self.f_struct_cls['results'],
                              f'ranked_{tag[label_field]}_rel.xlsx')
        df_ranked_rel.to_excel(f_path)
        print("---- Ranked relative errors:")
        print(df_ranked_rel.head())

        # Compute errors
        Er_NoOrder = 1 - np.trace(Mconf_NoOrder) / np.sum(Mconf_NoOrder)
        Er_Order = 1 - np.trace(Mconf_Order) / np.sum(Mconf_Order)

        # ##################
        # Compute EMD errors

        # Level FORD 1 errors
        # path2tax = os.path.join(self.f_struct['import'], 'taxonomy')
        path2tax = '../MINECO2018/source_data/taxonomy'
        fpath = os.path.join(path2tax, 'S_Matrix_Ford1.xlsx')
        emd_F1NoOrder = EV.compute_EMD_error(
            man_labs_filtered, u2f_labs_filtered, fpath, order_sensitive=False)
        print(f"---- Level 1 error rate: {emd_F1NoOrder}")

        # Level FORD 1 errors
        # path2tax = os.path.join(self.f_struct['import'], 'taxonomy')
        fpath = os.path.join(path2tax, 'S_Matrix_sim.xlsx')
        emd_csNoOrder = EV.compute_EMD_error(
            man_labs_filtered, u2f_labs_filtered, fpath, order_sensitive=False)
        print(f"---- Cost-sensitive error rate: {emd_csNoOrder}")

        # Combined Level FORD 1 - CS errors
        fpath = [os.path.join(path2tax, 'S_Matrix_Ford1.xlsx'),
                 os.path.join(path2tax, 'S_Matrix_sim.xlsx')]
        emd_csNoOrder2 = EV.compute_EMD_error(
            man_labs_filtered, u2f_labs_filtered, fpath, order_sensitive=False)
        print(f"---- Cost-sensitive error rate: {emd_csNoOrder2}")

        return Er_Order, Er_NoOrder

    def saveLabelPairs(self, df_labels, df_projs, single_u2f=False):

        """
        Returns a data frame with project data and their labels

        Unesco2ford label
        Args:
            df_labels   Dataframe with the manual labels
            df_projs    Dataframe with project info, u2f labels and automatic
                        classifications
            single_u2f: If True, only those projects with a single u2f label
                        are saved to file
        """

        # Read manual labels
        refs = df_labels['Referencia'].tolist()
        df_projs = df_projs[df_projs['Referencia'].isin(refs)]

        # Join in a single dataframe the df with the manual labels and that
        # with the labels to be evaluated.
        # Only those projects with a manual label are taken into account.
        df2 = df_labels.join(
            df_projs.set_index('Referencia'), on='Referencia')

        # Remove samples without class labels
        idx = [i for i, x in enumerate(df2['Unesco2Ford'].values)
               if x is not None]
        man_labels0 = [df2['labelstr'].values[i] for i in idx]
        u2f_labels0 = [df2['Unesco2Ford'].values[i] for i in idx]
        refs0 = [df2['Referencia'].values[i] for i in idx]

        # Extract labels:
        man_labels1 = [x.split('++') for x in man_labels0]
        u2f_labels1 = [x.split(',') for x in u2f_labels0]

        if single_u2f:
            # Take only items with a single label.
            print('---- NOTE: Only projects with a single u2f label will ' +
                  'be used to compute errors')
            idx = [i for i, x in enumerate(u2f_labels1) if len(x) == 1]
            man_labels1 = [man_labels1[i] for i in idx]
            u2f_labels1 = [u2f_labels1[i] for i in idx]
            refs0 = [refs0[i] for i in idx]

        # Filter projects that contain level 1-categories
        refs0_filtered = []
        for n, x in enumerate(zip(man_labels1, u2f_labels1, refs0)):
            if (len([z for z in x[0] if z.endswith('_')]) == 0 and
                    len([z for z in x[1] if z.endswith('_')]) == 0):
                refs0_filtered.append(x[2])

        df_out = df2[df2['Referencia'].isin(refs0)]
        # Reorder columns
        df_out = df_out[['Referencia', 'Unesco2Ford', 'labelstr',
                         'FORDclassif', 'Titulo', 'Resumen', 'userId']]

        return df_out

    def lemmatizeDefs(self, cf, df):

        # Lemmatize the concatenation of Name and Definition columns
        print('---- Lemmatizing and saving category definitions to database:')

        # Initialize lemmatizer object
        languages = self.cf.get('PREPROC', 'languages')
        lemmatizer_tool = self.cf.get('PREPROC', 'lemmatizer_tool')
        hunspelldic = self.cf.get('PREPROC', 'hunspelldic')
        stw_file = self.cf.get('PREPROC', 'stw_file')
        ngram_file = self.cf.get('PREPROC', 'ngram_file')
        dict_eq_file = self.cf.get('PREPROC', 'dict_eq_file')
        tilde_dictio = self.cf.get('PREPROC', 'tilde_dictio')

        lm = Lemmatizer(languages, lemmatizer_tool, hunspelldic, stw_file,
                        ngram_file, dict_eq_file, tilde_dictio)

        # str(x[2]) is used because some cell maye be NoneType
        data = list(df.values)
        lemmatizedRes = list(map(lambda x: (x[0], lm.processESstr(
            x[1] + ' ' + str(x[2]), keepsentence=True, removenumbers=True)),
            data))

        df_lemmas = pd.DataFrame(lemmatizedRes,
                                 columns=['Reference', 'Def_lemmas'])

        return df_lemmas

    def expandCategories(self, df):
        '''
        Expands category definitions using text taken from the Wikipedia.

        It assumes that the taxonomy table contains pointers to the
        wikipedia pages defining each category
        '''

        # #################
        # Reading Wikipedia

        # We take data from the Spanish Wikipedia
        wikipedia.set_lang('es')

        df = df.set_index('Reference')

        # Read wikipedia pages for all categories in the taxonomy
        wikidefs = []
        for cat in df.index:
            print('----- ----- Category {}'.format(df.loc[cat, 'Name']))

            # Compute a single-string definition of category cat
            wdef = ''

            if df.loc[cat, 'WikiRef'] is not None:
                # Take the list of wikipedia tags for category cat
                # (a '+'-separated string of wikipedia titles)
                refs = df.loc[cat, 'WikiRef'].split('+')
                print('               Refs {}'.format(refs))

                # Read all wikipedia pages related to the category
                for ref in refs:
                    page = wikipedia.page(ref)
                    print('               Page {}'.format(page.title))
                    wdef += page.title + ' ' + page.content + ' '

            wikidefs.append((cat, wdef))

        # #############
        # Lemmatization

        # Lemmatize the concatenation of Name and Definition columns
        print('---- Lemmatizing expanded category definitions')
        # Initialize lemmatizer object
        languages = self.cf.get('PREPROC', 'languages')
        lemmatizer_tool = self.cf.get('PREPROC', 'lemmatizer_tool')
        hunspelldic = self.cf.get('PREPROC', 'hunspelldic')
        stw_file = self.cf.get('PREPROC', 'stw_file')
        ngram_file = self.cf.get('PREPROC', 'ngram_file')
        dict_eq_file = self.cf.get('PREPROC', 'dict_eq_file')
        tilde_dictio = self.cf.get('PREPROC', 'tilde_dictio')

        lm = Lemmatizer(languages, lemmatizer_tool, hunspelldic, stw_file,
                        ngram_file, dict_eq_file, tilde_dictio)

        wikidef_lemmas = list(map(
            lambda x: (x[0], lm.processESstr(x[1], keepsentence=True,
                                             removenumbers=True)),
            wikidefs))

        # Output dataframe to be sent to the database
        df_wdefs = pd.DataFrame(wikidef_lemmas,
                                columns=['Reference', 'WikiDefs'])

        return df_wdefs
