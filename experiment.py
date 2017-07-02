import os


class Experiment:
    def __init__(self, root, base_folder):
        self.root = root
        self.base_path = os.path.join(self.root, base_folder)
        self.__get_experiment_id()
        self.dir = self.__get_experiment_dir()
        self.last_restore_point, self.restore_dir = self.__get_last_restore_point()

        print("experiment_dir: {}".format(self.dir))

    def __get_experiment_id(self):
        self.experiment_id = 1
        path = os.path.join(self.base_path, str(self.experiment_id))
        while os.path.isdir(path):
            self.experiment_id += 1
            path = os.path.join(self.base_path, str(self.experiment_id))

    def __get_experiment_dir(self):
        return os.path.join(self.base_path, str(self.experiment_id))

    def __get_last_restore_point(self):
        folder = os.path.join(self.base_path, str(self.experiment_id - 1))
        meta_file = None
        id_ = self.experiment_id

        while meta_file is None:
            for root, directory, files in os.walk(folder):
                for name in files:
                    if name.endswith('.meta'):
                        meta_file = os.path.join(folder, name)
            folder = os.path.join(self.base_path, str(id_))
            id_ -= 1
            if id_ < 0:
                folder = None
                break
        return meta_file, folder
