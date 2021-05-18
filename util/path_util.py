from pathlib import Path

from definitions import DATA_DIR, OUTPUT_DIR


class PathUtil:
    """
    provide a way to get a path to specific directive
    """

    @staticmethod
    def doc(pro_name, version):
        doc_output_dir = Path(DATA_DIR) / "doc"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_path = str(
            doc_output_dir / ("{pro}.{version}.dc".format(pro=pro_name, version=version)))
        return doc_path

    @staticmethod
    def wiki_emb_path():
        doc_output_dir = Path(DATA_DIR) / "wiki_word2vec"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        return doc_output_dir

    @staticmethod
    def output_trained_tfIdf_model_dir():
        trained_tfIdf_model_path = Path(DATA_DIR) / "tfidf_model" / "train"
        trained_tfIdf_model_path.mkdir(exist_ok=True, parents=True)
        return trained_tfIdf_model_path

    @staticmethod
    def output_corpus_tfIdf_model_dir():
        trained_tfIdf_model_path = Path(DATA_DIR) / "tfidf_model" / "corpus"
        trained_tfIdf_model_path.mkdir(exist_ok=True, parents=True)
        return trained_tfIdf_model_path

    @staticmethod
    def output_document_tfIdf_model_dir():
        trained_tfIdf_model_path = Path(DATA_DIR) / "tfidf_model" / "document"
        trained_tfIdf_model_path.mkdir(exist_ok=True, parents=True)
        return trained_tfIdf_model_path

    @staticmethod
    def output_RankSVM_model_dir():
        trained_tfIdf_model_path = Path(DATA_DIR) / "RankSVM_model"
        trained_tfIdf_model_path.mkdir(exist_ok=True, parents=True)
        return trained_tfIdf_model_path

    @staticmethod
    def output_index_model_dir():
        trained_tfIdf_model_path = Path(DATA_DIR) / "index_model"
        trained_tfIdf_model_path.mkdir(exist_ok=True, parents=True)
        return trained_tfIdf_model_path
