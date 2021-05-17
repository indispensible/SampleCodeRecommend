from pathlib import Path

from definitions import DATA_DIR, OUTPUT_DIR


class PathUtil:
    """
    provide a way to get a path to specific directive
    """

    @staticmethod
    def annotation_sentence():
        annotation_sentence = Path(DATA_DIR) / "annotation_data" / "annotation_sentence.json"
        return str(annotation_sentence)

    @staticmethod
    def graph_data(pro_name, version):
        graph_data_output_dir = Path(DATA_DIR) / "graph"
        graph_data_output_dir.mkdir(exist_ok=True, parents=True)

        graph_data_path = str(graph_data_output_dir / "{pro}.{version}.graph".format(pro=pro_name, version=version))
        return graph_data_path

    @staticmethod
    def name_searcher(pro_name, version):
        name_searcher_output_dir = Path(DATA_DIR) / "name_searcher"
        name_searcher_output_dir.mkdir(exist_ok=True, parents=True)

        name_searcher_path = str(name_searcher_output_dir / "{pro}_name_searcher.{v}".format(pro=pro_name, v=version))
        return name_searcher_path

    @staticmethod
    def name_cache(pro_name, version):
        name_cache_output_dir = Path(DATA_DIR) / "name_cache"
        name_cache_output_dir.mkdir(exist_ok=True, parents=True)

        name_cache_path = str(name_cache_output_dir / "{pro}_name_cache.{v}.pickle".format(pro=pro_name, v=version))
        return name_cache_path

    @staticmethod
    def multi_document_collection(pro_name, version="v2"):
        doc_output_dir = Path(DATA_DIR) / "doc"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_name = doc_output_dir / "{pro}.{v}.dc".format(pro=pro_name, v=version)
        return str(doc_name)

    @staticmethod
    def doc_path():
        doc_output_dir = Path(DATA_DIR) / "doc"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        return str(doc_output_dir)

    @staticmethod
    def json_doc_path():
        doc_output_dir = Path(DATA_DIR) / "doc" / "jdk"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        return str(doc_output_dir)

    def trainging_data_path(version, data_version):
        training_data_path = Path(OUTPUT_DIR) / "training_data" / "training_data.{v1}.{v2}.json".format(
            v1=version, v2=data_version)
        return str(training_data_path)

    @staticmethod
    def trainging_jsonl_path(version, data_version):
        training_data_path = Path(OUTPUT_DIR) / "training_data" / "training_data.{v1}.{v2}.jsonl".format(
            v1=version, v2=data_version)
        return str(training_data_path)

    @staticmethod
    def empirical_research_jsonl_path(version, data_version):
        empirical_search_json1_path =\
            Path(OUTPUT_DIR) / "empirical_research_data" / "empirical_research_data.{v1}.{v2}.jsonl".format(
                v1=version, v2=data_version
            )
        return str(empirical_search_json1_path)

    @staticmethod
    def annotation_corpus_path():
        corpus_path = Path(DATA_DIR) / "annotation_data" / "corpus.json"
        return str(corpus_path)

    @staticmethod
    def annotation_path():
        annotation_path = Path(DATA_DIR) / "annotation_data"
        return str(annotation_path)

    @staticmethod
    def annotation_label_set(data_version, extract_type):
        label_path = Path(DATA_DIR) / "annotation_data" / "{v1}/{v2}/relation_extract_labels.json".format(
            v1=data_version, v2=extract_type)
        return str(label_path)

    @staticmethod
    def annotation_data_path(data_version, extract_type):
        annotation_data_path = Path(DATA_DIR) / "annotation_data" / "{v1}/{v2}".format(
            v1=data_version, v2=extract_type)
        annotation_data_path.mkdir(exist_ok=True, parents=True)
        return str(annotation_data_path)

    @staticmethod
    def annotation_same_sentence_json_path(data_version, extract_type):
        annotation_data_path = Path(DATA_DIR) / "annotation_data" / "{v1}/{v2}/same_tag_sentence.json".format(
            v1=data_version, v2=extract_type)
        return str(annotation_data_path)

    @staticmethod
    def annotation_different_sentence_json_path(data_version, extract_type):
        annotation_data_path = Path(DATA_DIR) / "annotation_data" / "{v1}/{v2}/different_tag_sentence.json".format(
            v1=data_version, v2=extract_type)
        return str(annotation_data_path)

    @staticmethod
    def annotation_arbitrament_path(data_version, extract_type):
        annotation_data_path = Path(DATA_DIR) / "annotation_data" / "{v1}/{v2}/arbitrament.json1".format(
            v1=data_version, v2=extract_type)
        return str(annotation_data_path)

    @staticmethod
    def spacy_model(version, n_iter, drop):
        model_version_path = Path(OUTPUT_DIR) / "spacy_model" / "{v}".format(v=version)
        model_version_path.mkdir(exist_ok=True, parents=True)
        model_path = model_version_path / "{n}_{d}".format(n=str(n_iter), d=str(int(drop * 100)))
        model_path.mkdir(exist_ok=True, parents=True)
        return str(model_path)

    @staticmethod
    def spacy_train_data_path(version):
        annotation_data_dir = Path(OUTPUT_DIR) / "spacy_model" / "train_data" / "{v}".format(v=version)
        return str(annotation_data_dir / "train_data.json")

    @staticmethod
    def spacy_predict_data_path(version):
        annotation_data_dir = Path(OUTPUT_DIR) / "spacy_model" / "train_data" / "{v}".format(v=version)
        return str(annotation_data_dir / "predict_data.json")

    @staticmethod
    def bert_embedding_model():
        model_version_path = Path(DATA_DIR) / "bert_embedding"
        return str(model_version_path)

    @staticmethod
    def base_bert_embedding_model():
        model_version_path = Path(DATA_DIR) / "bert_embedding" / "base_embedding"
        return str(model_version_path)

    @staticmethod
    def bi_lstm_crf_ner_model(version):
        model_version_path = Path(OUTPUT_DIR) / "bert_model" / "{v}".format(v=version)
        model_version_path.mkdir(exist_ok=True, parents=True)
        model_path = model_version_path / "ner_model"
        return str(model_path)

    @staticmethod
    def bert_ner_model(version, batch_size, epochs):
        model_version_path = Path(OUTPUT_DIR) / "bert_model" / "{v}".format(v=version)
        model_version_path.mkdir(exist_ok=True, parents=True)
        model_path = model_version_path / "bert_ner_model" / "{batch_size}_{epochs}".format(batch_size=str(batch_size),
                                                                                            epochs=str(epochs))
        model_path.mkdir(exist_ok=True, parents=True)
        return str(model_path)

    @staticmethod
    def bert_train_data_path(version):
        annotation_data_dir = Path(OUTPUT_DIR) / "bert_model" / "train_data" / "{v}".format(v=version)
        annotation_data_dir.mkdir(exist_ok=True, parents=True)
        return str(annotation_data_dir / "train_data.json")

    @staticmethod
    def bert_predict_data_path(version):
        annotation_data_dir = Path(OUTPUT_DIR) / "bert_model" / "train_data" / "{v}".format(v=version)
        annotation_data_dir.mkdir(exist_ok=True, parents=True)
        return str(annotation_data_dir / "predict_data.json")

    @staticmethod
    def bert_valid_data_path(version):
        annotation_data_dir = Path(OUTPUT_DIR) / "bert_model" / "train_data" / "{v}".format(v=version)
        annotation_data_dir.mkdir(exist_ok=True, parents=True)
        return str(annotation_data_dir / "valid_data.json")

    @staticmethod
    def doc(pro_name, version):
        doc_output_dir = Path(DATA_DIR) / "doc"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_path = str(
            doc_output_dir / ("{pro}.{version}.dc".format(pro=pro_name, version=version)))
        return doc_path

    @staticmethod
    def doc_with_datetime(pro_name, version, datetime):
        doc_output_dir = Path(DATA_DIR) / "doc" / pro_name
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_path = str(
            doc_output_dir / ("{pro}.{version}.{datetime}.dc".format(pro=pro_name, version=version, datetime=datetime)))
        return doc_path

    @staticmethod
    def graph_with_datetime(pro_name, version, datetime):
        doc_output_dir = Path(DATA_DIR) / "graph" / pro_name
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_path = str(
            doc_output_dir / (
                "{pro}.{version}.{datetime}.graph".format(pro=pro_name, version=version, datetime=datetime)))
        return doc_path

    @staticmethod
    def input_wordemb(version='v5'):
        doc_output_dir = Path(DATA_DIR) / "wiki_word2vec"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_path = str(doc_output_dir / ("{version}.bin".format(version=version)))
        return doc_path

    @staticmethod
    def wiki_emb_path():
        doc_output_dir = Path(DATA_DIR) / "wiki_word2vec"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        return doc_output_dir

    @staticmethod
    def input_name_cache(pro_name='jdk', version='v5'):
        # todo: 要废弃
        name_cache_output_dir = Path(DATA_DIR) / "name_cache"
        name_cache_output_dir.mkdir(exist_ok=True, parents=True)

        name_cache_path = str(name_cache_output_dir / "{pro}_name_cache.{v}.pickle".format(pro=pro_name, v=version))
        return name_cache_path

    @staticmethod
    def input_graph_data(pro_name="jdk", version="v5"):
        doc_output_dir = Path(DATA_DIR) / "graph" / pro_name
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_path = str(
            doc_output_dir / ("{pro}.{version}.graph".format(pro=pro_name, version=version)))
        return doc_path

    @staticmethod
    def input_doc_collection(pro_name="jdk", version="v5"):
        doc_output_dir = Path(DATA_DIR) / "doc"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_path = str(
            doc_output_dir / ("{pro}.{version}.dc".format(pro=pro_name, version=version)))
        return doc_path

    @staticmethod
    def fast_text_directive_classifier_model(profile='Model.bin'):
        model_output_dir = Path(DATA_DIR) / "directive_classifier"
        model_output_dir.mkdir(exist_ok=True, parents=True)

        model_path = str(model_output_dir / "{pro}".format(pro=profile))
        return model_path

    @staticmethod
    def fast_text_train_data_path():
        training_data_path = Path(DATA_DIR) / "directive_classifier" / "evaluation" / "train.txt"
        return str(training_data_path)

    @staticmethod
    def fast_text_test_data_path():
        testing_data_path = Path(DATA_DIR) / "directive_classifier" / "evaluation" / "test.txt"
        return str(testing_data_path)

    @staticmethod
    def ner_annotation_data():
        doc_output_dir = Path(DATA_DIR) / "ner"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_path = str(
            doc_output_dir / "ner-annotation.json")
        return doc_path

    @staticmethod
    def pre_train_w2v_model():
        pre_train_w2v_model_path = Path(DATA_DIR) / "wiki_word2vec" / "pretrain_300.bin"
        return str(pre_train_w2v_model_path)

    @staticmethod
    def output_trained_w2v_model(version):
        trained_w2v_model_path = Path(DATA_DIR) / "word2vec_model" / "train"
        trained_w2v_model_path.mkdir(exist_ok=True, parents=True)
        model_path = str(trained_w2v_model_path / "{version}.tunrd.wordemb".format(version=version))
        return str(model_path)

    @staticmethod
    def fast_text_directive_result_from_drone_path():
        directive_result_from_drone_path = Path(DATA_DIR) / "directive_classifier" / "directive_result_from_drone.json"
        return str(directive_result_from_drone_path)

    @staticmethod
    def fast_text_directive_result_from_train_data_path():
        directive_result_from_train_data_path = Path(
            DATA_DIR) / "directive_classifier" / "directive_result_from_train_data.json"
        return str(directive_result_from_train_data_path)

    @staticmethod
    def directive_model_path():
        directive_model_path = Path(
            DATA_DIR) / "directive_classifier" / "Model.bin"
        return str(directive_model_path)

    @staticmethod
    def entity_link(model='test'):
        entity_link = Path(DATA_DIR) / "entity_link" / "test" / "test.json"
        return str(entity_link)

    @staticmethod
    def entity_link_regression_model(model_dir='ridge', model_name="ridge"):
        doc_output_dir = Path(DATA_DIR) / "entity_link" / model_dir
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        entity_link_regression_model = str(doc_output_dir / "{file}.model").format(file=model_name)
        return entity_link_regression_model

    @staticmethod
    def train_decision_tree_data_path(pro_name="entity_link", file_name="evaluation_for_jdk_v6_1"):
        doc_output_dir = Path(DATA_DIR) / pro_name / "evaluation"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_path = str(doc_output_dir / ("{file}.json".format(file=file_name)))
        return doc_path

    @staticmethod
    def decision_tree_model_path(pro_name="entity_link", file_name="decision_tree", max_depth=3):
        doc_output_dir = Path(DATA_DIR) / pro_name / "decision_tree"
        doc_output_dir.mkdir(exist_ok=True, parents=True)
        doc_path = str(doc_output_dir / ("{file}.{max_depth}.model".format(file=file_name, max_depth=max_depth)))
        return doc_path

    @staticmethod
    def bert_result_dir_path(version, batch_size, epochs):
        annotation_data_dir = Path(OUTPUT_DIR) / "bert_model" / "result_data" / "{v}/{b}_{e}".format(
            v=version, b=batch_size, e=epochs)
        annotation_data_dir.mkdir(exist_ok=True, parents=True)
        return str(annotation_data_dir)

    @staticmethod
    def experiment_dir_path(experiment_id=1):
        experiment_data_dir = Path(DATA_DIR) / "experiment" / "RQ_{e_id}".format(e_id=experiment_id)
        experiment_data_dir.mkdir(exist_ok=True, parents=True)
        return str(experiment_data_dir)


    @staticmethod
    def empirical_research_key_word_json_path():
        empirical_research_key_word_json_path = Path(
            OUTPUT_DIR) / "empirical_research_data" / "empirical_research_key_word.json"
        return str(empirical_research_key_word_json_path)

    @staticmethod
    def empirical_research_field_list_json_path():
        empirical_research_field_list_json_path = Path(
            OUTPUT_DIR) / "empirical_research_data" / "sentence_field_list.json"
        return str(empirical_research_field_list_json_path)

    @staticmethod
    def output_trained_tfIdf_model_dir():
        trained_tfIdf_model_path = Path(DATA_DIR) / "tfidf_model" / "train"
        trained_tfIdf_model_path.mkdir(exist_ok=True, parents=True)
        return trained_tfIdf_model_path

    @staticmethod
    def simple_code_dir_file(code_file):
        sample_code_dir_path = Path(DATA_DIR) / "sample_code"
        sample_code_dir_path.mkdir(exist_ok=True, parents=True)
        sample_code_path = sample_code_dir_path / code_file
        return sample_code_path

    @staticmethod
    def json_data(file_name):
        json_data_output_dir = Path(DATA_DIR) / "json"
        json_data_output_dir.mkdir(exist_ok=True, parents=True)

        json_data_path = str(json_data_output_dir / "{file}.json".format(file=file_name))
        return json_data_path

    @staticmethod
    def jdk_graph_data(version="v1"):
        jdk_graph_data_dir = Path(OUTPUT_DIR) / "graph" / "jdk8"
        jdk_graph_data_dir.mkdir(exist_ok=True, parents=True)

        jdk_graph_data_path = str(jdk_graph_data_dir / ("jdk8" + "." + version + ".graph"))
        return jdk_graph_data_path

    @staticmethod
    def android_graph_data(version="v1"):
        jdk_graph_data_dir = Path(OUTPUT_DIR) / "graph" / "android27"
        jdk_graph_data_dir.mkdir(exist_ok=True, parents=True)

        jdk_graph_data_path = str(jdk_graph_data_dir / ("android27" + "." + version + ".graph"))
        return jdk_graph_data_path

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

    @staticmethod
    def output_es_data_dir():
        trained_es_data_path = Path(DATA_DIR) / "ES"
        trained_es_data_path.mkdir(exist_ok=True, parents=True)
        return trained_es_data_path
