import os
import pandas

from tqdm.auto import tqdm

from config import Config


class BaseDataFrame:

    """Базовый класс для работы с датафреймом"""

    def __init__(self):
        self._df = None

    @property
    def df(self):
        if self._df is not None:
            return self._df
        else:
            self.generate_dataframe()
            return self._df

    def generate_dataframe(self):
        minor: list = []
        major: list = []
        critical: list = []
        blocker: list = []
        final_classes_list: list = []

        for file_name in tqdm(os.listdir(Config.path_minor)):
            with open(f'{Config.path_minor}/{file_name}', 'r', encoding='utf-8') as file:
                data = file.read()
                minor.append(data)

        for file_name in tqdm(os.listdir(Config.path_major)):
            with open(f'{Config.path_major}/{file_name}', 'r', encoding='utf-8') as file:
                data = file.read()
                major.append(data)

        for file_name in tqdm(os.listdir(Config.path_critical)):
            with open(f'{Config.path_critical}/{file_name}', 'r', encoding='utf-8') as file:
                data = file.read()
                critical.append(data)

        for file_name in tqdm(os.listdir(Config.path_blocker)):
            with open(f'{Config.path_blocker}/{file_name}', 'r', encoding='utf-8') as file:
                data = file.read()
                blocker.append(data)

        for i in range(len(minor)):
            final_classes_list.append(1)

        for i in range(len(major)):
            final_classes_list.append(2)

        for i in range(len(critical)):
            final_classes_list.append(3)

        for i in range(len(blocker)):
            final_classes_list.append(4)

        final_description_list = minor + major + critical + blocker
        final_data: dict = {"description": final_description_list, "class_cri": final_classes_list}

        self._df = pandas.DataFrame(data=final_data)
