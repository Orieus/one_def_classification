import itertools
import pandas as pd
try:
    import Levenshtein
except:
    print("---- WARNING: Error importing Levenshtein.")
    print("              Detector of coordinated projects not available")
    print("              You can try pip install python-Levenshtein")

# Local imports
from fordclassifier.corpusanalyzer.bow import Bow
from fordclassifier.corpusanalyzer.textproc import Lemmatizer


class FCCorpusAnalyzer(object):
    """
    This class contains those methods used to process some data from the
    corpus database used in the MINECO project (which containts abstracts of
    research projects in a particular DB structure).

    Thus, they are specific for the particular data structure used in this
    project.
    """

    def __init__(self):

        return

    def detectCoord(self, df, refs):

        def find_structure(ref):
            structure = ''
            for i, el in enumerate(ref):
                if el.isalpha():
                    if i > 0 and structure[-1] == '-':
                        structure += el
                    else:
                        structure += 'a'
                elif el.isnumeric():
                    structure += '0'
                elif el == '-':
                    structure += '-'
                else:
                    structure += '?'
            return structure

        print('---- ---- Finding data structure.')
        ref_str_tit_res = list(map(
            lambda x: (x[0], find_structure(x[0]), x[1], x[2]), df.values))

        # refs_str = list(map(lambda x: x[1], ref_str_tit_res))
        # count = Counter(refs_str)

        # print(count)
        # ## 'aaa0000-00000': 34164,      Proyectos de Plan Nacional antiguos
        # ########## 'aaa0000-00000-C0-0-P': 2290,   POTENCIALES COORDINADOS de
        #                               los nuevos (Proyectos de Investigación)
        # ########## 'aaa0000-00000-C0-0-R': 4887,   POTENCIALES COORDINADOS de
        #                                            los nuevos (Retos)
        # ########## 'aaa0000-00000-C00-00': 16879,  POTENCIALES COORDINADOS
        #                                            ANTIGUOS
        # ########## 'aaa0000-00000-Caa': 290,            ?????
        # ## 'aaa0000-00000-E': 8938,
        # ## 'aaa0000-00000-Eaa': 4607,           EXPLORA
        # ## 'aaa0000-00000-Jaa': 2619,           ??????
        # ## 'aaa0000-00000-P': 9603,             Proyectos de Plan Nacional
        # ## 'aaa0000-00000-R': 11125,            Proyectos de Plan Nacional
        # ## 'aaa0000-00000-Raaa': 1100,          ???????
        # ## 'aaa0000-A-00000': 19,
        # ## 'aaa0000-C-00000': 36,
        # ########## 'aaaa-0000-000': 701,         POTENCIALES COORDINADOS
        # ########## 'aaaa-0000-000-C00-00': 152,  POTENCIALES COORDINADOS
        # ## 'aaaa0000-000': 30,
        # ########## 'aaaa0000-000-C00-00': 45,    POTENCIALES COORDINADOS
        # ########## 'aaaa0000-00000': 737,
        # ## 'aaaa0000-A-00000': 10

        # Los coordinados deben estar en las mismas categorías. Hacemos
        # busqueda de acuerdo a las siguientes reglas:
        # Examinamos que la diferencia del número de caracteres sea superior a
        # un umbral; si no lo es calculamos la
        # distancia entre los resúmenes

        # Para minimizar el número de comparaciones a revisar, y viendo la
        # estructura de coordinados de la lista anterior

        th = 50
        min_len = 10
        th_leve = 0.9

        coord_list = []

        # Iteramos diferentes estructuras, considerando únicamente las que
        # presentan coordinados
        print('---- ---- Iterating structure 1 of 3.')
        for structure in ['aaaa0000-00000', 'aaaa0000-000-C00-00']:

            print(structure)

            selected = list(filter(
                lambda x: x[1] == structure, ref_str_tit_res))

            lens = list(map(
                lambda x: len(x[3].replace('\r\n', '\n').strip()), selected))

            for i, sel1 in enumerate(selected):
                if lens[i] > min_len:
                    coord_i = [sel1[0]]
                    for j, sel2 in enumerate(selected):
                        if ((i != j) & ((abs(lens[i] - lens[j])) < th) &
                                (sel1[0][:8] == sel2[0][:8])):
                            if (Levenshtein.ratio(sel1[-1].strip(),
                                                  sel2[-1].strip())) > th_leve:
                                coord_i.append(sel2[0])
                    if (len(coord_i) > 1):
                        coord_list.append(coord_i)

        print('---- ---- Iterating structure 2 of 3.')
        for structure in ['aaaa-0000-000-C00-00', 'aaaa-0000-000']:

            print(structure)

            selected = list(filter(
                lambda x: x[1] == structure, ref_str_tit_res))

            lens = list(map(
                lambda x: len(x[3].replace('\r\n', '\n').strip()), selected))

            for i, sel1 in enumerate(selected):
                if lens[i] > min_len:
                    coord_i = [sel1[0]]
                    for j, sel2 in enumerate(selected):
                        if ((i != j) & ((abs(lens[i] - lens[j])) < th) &
                                (sel1[0][:9] == sel2[0][:9])):
                            if (Levenshtein.ratio(sel1[-1].strip(),
                                                  sel2[-1].strip())) > th_leve:
                                coord_i.append(sel2[0])
                    if (len(coord_i) > 1):
                        coord_list.append(coord_i)

        print('---- ---- Iterating structure 3 of 3.')
        for structure in ['aaa0000-00000-Caa', 'aaa0000-00000-Eaa',
                          'aaa0000-00000-C0-0-P', 'aaa0000-00000-C0-0-R',
                          'aaa0000-00000-E', 'aaa0000-00000-C00-00', ]:

            print(structure)
            selected = list(filter(
                lambda x: x[1] == structure, ref_str_tit_res))

            lens = list(map(
                lambda x: len(x[3].replace('\r\n', '\n').strip()), selected))

            for i, sel1 in enumerate(selected):
                if lens[i] > min_len:
                    coord_i = [sel1[0]]
                    for j, sel2 in enumerate(selected):
                        if ((i != j) & ((abs(lens[i] - lens[j])) < th) &
                                (sel1[0][:7] == sel2[0][:7])):
                            if (Levenshtein.ratio(sel1[-1].strip(),
                                                  sel2[-1].strip())) > th_leve:
                                coord_i.append(sel2[0])
                    if (len(coord_i) > 1):
                        coord_list.append(coord_i)

        ######################
        # Remove duplicates from list
        print('---- ---- Post-processing')
        coord_list = list(map(sorted, coord_list))
        coord_list.sort()
        coord_list = list(
            coord_list for coord_list, _ in itertools.groupby(coord_list))

        # Consolidate groups of projects (if A coordinated with B and B
        # coordinated with C, then A coordinated with C)
        coord_list2 = []
        for el in coord_list:
            new_item = el
            for el2 in coord_list:
                if set.intersection(set(el), set(el2)):
                    new_item = list(set(el+el2))
            coord_list2.append(new_item)

        coord_list = list(map(sorted, coord_list2))
        coord_list.sort()
        coord_list = list(coord_list for coord_list, _ in itertools.groupby(
            coord_list))

        def detecta_year(proj_ref):
            for i, ch in enumerate(proj_ref):
                if ch.isnumeric():
                    return int(proj_ref[i: i+4])

        coord_list2 = []
        for el in coord_list:
            coord_list2.append((el, detecta_year(el[0])))

        # Assign each project in a group to its group representative
        ref2coord = dict(zip(refs, refs))
        for group in coord_list2:
            for p in group[0]:
                # Each group is represented by the first project in the list
                ref2coord[p] = group[0][0]
        refcoord = list(ref2coord.items())

        return refcoord, coord_list2

    def computeBow(self, df, min_df=2, title_mul=1, verbose=True):
        """
        Computes the BoW from a corpus dataframe with at least three columns:
            - Resumen_lemas (all of them in Spanish)
            - Titulo_lemas (some of them may be in english)
            - Titulo_lang (specify the languaje of each title)

        Args:
            df:     Dataframe of projects
            min_df: Minimum document frequency for the bow computation
            title_mul: Multiplier of the title words
        """

        # Separate DB columns in different variables
        # refs = list(df.loc[:, 'Referencia'])
        titulo_lang = list(df.loc[:, 'Titulo_lang'])
        resumen_lemas = list(df.loc[:, 'Resumen_lemas'])
        titulo_lemas = list(df.loc[:, 'Titulo_lemas'])

        # Convert corpus in a list of strings
        docs = []
        for lang, tit, res in zip(titulo_lang, titulo_lemas, resumen_lemas):

            # Add title (if in Spanish)
            if lang == 'es':
                text = title_mul * (tit + ' ') + '\n' + res
            else:
                text = res

            text = text.lower().replace('\n', ' ')
            # ### BUG?
            text = text.replace('.', ' ')

            docs.append(text)

        # ####################
        # Compute and save BoW

        # Initialize BoW object
        if verbose:
            print("---- Computing TFIDF...")
        bow = Bow(min_df)

        # Compute BoW and vocabularies from docs
        Xtfidf = bow.fit(docs)
        vocab = bow.tfidf.vocabulary_
        inv_vocab = bow.obtain_inv_vocab()

        return Xtfidf, vocab, inv_vocab

    def computeLemmas(self, cf, refs, abstracts, titles):

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

        # ################
        # Lemmatize titles
        print('---- ---- (1/4) Detecting language from titles')
        corpus_lang = lm.langDetection(titles)

        print('---- ---- (2/4) Lemmatizing titles')
        corpus_lemmas = lm.lemmatizeES(titles, chunksize=100)
        lemmatizedRes = list(zip(refs, corpus_lang, corpus_lemmas))
        df_lemas_tit = pd.DataFrame(lemmatizedRes, columns=[
            'Referencia', 'Titulo_lang', 'Titulo_lemas'])

        # ####################
        # Processing abstracts
        print('---- ---- (3/4) Detecting language from abstracts')
        corpus_lang = lm.langDetection(abstracts)

        print('---- ---- (4/4) Lemmatizing abstracts')
        corpus_lemmas = lm.lemmatizeES(abstracts, chunksize=100)
        lemmatizedRes = list(zip(refs, corpus_lang, corpus_lemmas))
        df_lemas_abs = pd.DataFrame(lemmatizedRes, columns=[
                'Referencia', 'Resumen_lang', 'Resumen_lemas'])

        return df_lemas_tit, df_lemas_abs
