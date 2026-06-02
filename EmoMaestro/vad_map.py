import csv
import math
from warnings import warn
import util

class VADMap:
    def __init__(self, emotions_set):
        self.__emotions_set = emotions_set
        self.__csv_path = "datasets/warriner_s13428-012-0314-x.csv"
        self.__map = None
        self.__range = None
        self.__set_map()
        self.__set_range(tuple([(-1, 1)] * 3))

    def get_emo_vad(self, emo):
        emo = emo.lower()

        v = float(self.__map.get(emo)[0])
        a = float(self.__map.get(emo)[1])
        d = float(self.__map.get(emo)[2])

        return v, a, d

    def get_emotions(self):
        return tuple(self.__map.keys())

    def get_high_emo(self, coord):
        high_emo = None
        min_dist = None

        for emo, vad in self.__map.items():
            cmp_diff = 0

            for i in range(len(vad)):
                cmp_diff += pow(coord[i] - vad[i], 2)

            dist = math.sqrt(cmp_diff)

            if min_dist is None or dist < min_dist:
                min_dist = dist
                high_emo = emo

        return high_emo

    def __set_map(self):
        csv_path = self.__csv_path
        emotions = self.__emotions_set
        w_col = "Word"
        v_col = "V.Mean.Sum"
        a_col = "A.Mean.Sum"
        d_col = "D.Mean.Sum"
        p_min, p_max = 1, 9
        self.__range = tuple([(p_min, p_max)] * 3)

        vad_map = dict()

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                emo = row[w_col]

                if emo in emotions:
                    v, a, d = float(row[v_col]), float(row[a_col]), float(row[d_col])

                    vad_map[emo] = (v, a, d)

                    emotions.remove(emo)

        if emotions:
            warn(f"Emotions not found in {csv_path.split('/')[-1]}: {emotions}")

        self.__map = vad_map

    def __set_range(self, new_range):
        for emo, vad in self.__map.items():
            n_vad = []

            for i, p in enumerate(vad):
                o_min, o_max = self.__range[i][0], self.__range[i][1]
                n_min, n_max = new_range[i][0], new_range[i][1]

                new_p = (p - o_min) * (n_max - n_min) / (o_max - o_min) + n_min

                n_vad.append(new_p)

            self.__map[emo] = tuple(n_vad)