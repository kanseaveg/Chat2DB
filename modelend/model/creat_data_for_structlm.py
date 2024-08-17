import json
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from schema_item_classifier import *
from t5_inference import *
import numpy as np
from ..utils.args import parse_init
import argparse
import time, re
from transformers import RobertaTokenizerFast
from classifier_model import MyClassifier
from transformers.trainer_utils import set_seed
from transformers import T5TokenizerFast, T5ForConditionalGeneration, MT5ForConditionalGeneration
from tokenizers import AddedToken
from sql_metadata import Parser
from ..utils.common_utils import list2dict, get_db_schemas
from ..utils.bridge_content_encoder import get_database_matches
from ..target.third_party.spider.evaluation import main as spider_evaluate
from ..target.third_party.test_suite.evaluation import evaluate as ts_evaluate
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)


def get_db_contents(question, table_name_original, column_names_original, db_id, db_path):
    matched_contents = []
    # extract matched contents for each column
    for column_name_original in column_names_original:
        matches = get_database_matches(
            question,
            table_name_original,
            column_name_original,
            db_path + "/{}/{}.sqlite".format(db_id, db_id)
        )
        matches = sorted(matches)
        matched_contents.append(matches)

    return matched_contents
def preprocessing(opt, nl:str, db_id:str,db_schemas:dict):
    db_schemas = get_db_schemas(db_schemas[db_id])

    data = {"question": nl, "db_id": db_id}
    question = data["question"].replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "'").replace(
        "\u201d", "'").strip()
    db_id = data["db_id"]
    sql, norm_sql, sql_skeleton = "", "", ""
    natsql, norm_natsql, natsql_skeleton = "", "", ""
    natsql_used_columns, natsql_tokens = [], []
    preprocessed_data = {}
    preprocessed_data["question"] = question
    preprocessed_data["db_id"] = db_id
    preprocessed_data["sql"] = sql
    preprocessed_data["norm_sql"] = norm_sql
    preprocessed_data["sql_skeleton"] = sql_skeleton
    preprocessed_data["natsql"] = natsql
    preprocessed_data["norm_natsql"] = norm_natsql
    preprocessed_data["natsql_skeleton"] = natsql_skeleton
    preprocessed_data["db_schema"] = []
    preprocessed_data["pk"] = db_schemas["pk"]
    preprocessed_data["fk"] = db_schemas["fk"]
    preprocessed_data["table_labels"] = []
    preprocessed_data["column_labels"] = []

    # add database information (including table name, column name, ..., table_labels, and column labels)
    for table in db_schemas["schema_items"]:
        if opt.use_contents:
            db_contents = get_db_contents(
                question,
                table["table_name_original"],
                table["column_names_original"],
                db_id,
                opt.db_path
                )
            preprocessed_data["db_schema"].append({
                "table_name_original": table["table_name_original"],
                "table_name": table["table_name"],
                "column_names": table["column_names"],
                "column_names_original": table["column_names_original"],
                "column_types": table["column_types"],
                "db_contents": db_contents
            })
        else:
            preprocessed_data["db_schema"].append({
                "table_name_original": table["table_name_original"],
                "table_name": table["table_name"],
                "column_names": table["column_names"],
                "column_names_original": table["column_names_original"],
                "column_types": table["column_types"],
            })
        # extract table and column classification labels
        if table["table_name_original"] in natsql_tokens:  # for used tables
            preprocessed_data["table_labels"].append(1)
            column_labels = []
            for column_name_original in table["column_names_original"]:
                if table[
                    "table_name_original"] + "." + column_name_original in natsql_used_columns:  # for used columns
                    column_labels.append(1)
                else:
                    column_labels.append(0)
            preprocessed_data["column_labels"].append(column_labels)
        else:
            preprocessed_data["table_labels"].append(0)
            preprocessed_data["column_labels"].append([0 for _ in range(len(table["column_names_original"]))])

    return [preprocessed_data]


def lista_contains_listb(lista, listb):
    for b in listb:
        if b not in lista:
            return 0

    return 1


def prepare_input_and_output(opt, ranked_data, table_label, column_label):
    question = ranked_data["question"]

    schema_sequence = ""

    for table_id in range(len(ranked_data["db_schema"])):
        table_name_original = ranked_data["db_schema"][table_id]["table_name_original"]
        print(ranked_data["db_schema"][table_id])
        # add table name
        if opt.linking_mode == "interaction":
            if table_label[table_id] == 1:
                schema_sequence += " | " + table_name_original + " : "
        else:
            schema_sequence += " | " + table_name_original + " : "

        column_info_list = []
        for column_id in range(len(ranked_data["db_schema"][table_id]["column_names_original"])):
            # extract column name
            column_name_original = ranked_data["db_schema"][table_id]["column_names_original"][column_id]
            if opt.linking_mode == "interaction":
                # print(column_label)
                # print(table_id)
                # print(column_id)
                if column_label[table_id][column_id] == 1:
                    # use database contents if opt.use_contents = True
                    if opt.use_contents:
                        db_contents = ranked_data["db_schema"][table_id]["db_contents"][column_id]
                        if len(db_contents) != 0:
                            column_contents = " , ".join(db_contents)
                            column_info = table_name_original + "." + column_name_original + " ( " + column_contents + " ) "
                            print(column_info)
                        else:
                            column_info = table_name_original + "." + column_name_original
                    else:
                        column_info = table_name_original + "." + column_name_original

                    column_info_list.append(column_info)
            else:
                # use database contents if opt.use_contents = True
                if opt.use_contents:
                    db_contents = ranked_data["db_schema"][table_id]["db_contents"][column_id]
                    if len(db_contents) != 0:
                        column_contents = " , ".join(db_contents)
                        column_info = table_name_original + "." + column_name_original + " ( " + column_contents + " ) "
                        print(column_info)
                    else:
                        column_info = table_name_original + "." + column_name_original
                else:
                    column_info = table_name_original + "." + column_name_original

                column_info_list.append(column_info)

        if opt.linking_mode == "interaction":
            if table_label[table_id] == 1:
                column_info_list.append(table_name_original + ".*")

                # add column names
                schema_sequence += " , ".join(column_info_list)
        else:
            column_info_list.append(table_name_original + ".*")

            # add column names
            schema_sequence += " , ".join(column_info_list)

    if opt.add_fk_info:
        for fk in ranked_data["fk"]:
            schema_sequence += " | " + fk["source_table_name_original"] + "." + fk["source_column_name_original"] + \
                               " = " + fk["target_table_name_original"] + "." + fk["target_column_name_original"]

    # remove additional spaces in the schema sequence
    while "  " in schema_sequence:
        schema_sequence = schema_sequence.replace("  ", " ")

    # input_sequence = question + schema sequence
    input_sequence = question + schema_sequence

    if opt.output_skeleton:

        output_sequence = ranked_data["natsql_skeleton"] + " | " + ranked_data["norm_natsql"]
    else:
        output_sequence = ranked_data["norm_natsql"]
    print(input_sequence)
    return input_sequence, output_sequence


def text2sql_data_generator(data, opt, table_label, column_label):
    table_coverage_state_list, column_coverage_state_list = [], []
    ranked_data = dict()
    data = data[0]
    ranked_data["question"] = data["question"]
    ranked_data["sql"] = data["sql"]
    ranked_data["norm_sql"] = data["norm_sql"]
    ranked_data["sql_skeleton"] = data["sql_skeleton"]
    ranked_data["natsql"] = data["natsql"]
    ranked_data["norm_natsql"] = data["norm_natsql"]
    ranked_data["natsql_skeleton"] = data["natsql_skeleton"]
    ranked_data["db_id"] = data["db_id"]
    ranked_data["db_schema"] = []

    table_pred_probs = list(map(lambda x: round(x, 4), data["table_pred_probs"]))
    # find ids of tables that have top-k probability
    if opt.linking_mode == "None" or opt.linking_mode == "interaction":
        topk_table_ids = np.argsort(-np.array(table_pred_probs), kind="stable").tolist()
    else:
        topk_table_ids = np.argsort(-np.array(table_pred_probs), kind="stable")[:opt.topk_table_num].tolist()

    # if the mode == eval, we record some information for calculating the coverage
    if opt.mode == "eval":
        used_table_ids = [idx for idx, label in enumerate(data["table_labels"]) if label == 1]
        table_coverage_state_list.append(lista_contains_listb(topk_table_ids, used_table_ids))

        for idx in range(len(data["db_schema"])):
            used_column_ids = [idx for idx, label in enumerate(data["column_labels"][idx]) if label == 1]
            if len(used_column_ids) == 0:
                continue
            column_pred_probs = list(map(lambda x: round(x, 2), data["column_pred_probs"][idx]))
            if opt.linking_mode == "None" or opt.linking_mode == "interaction":
                topk_column_ids = np.argsort(-np.array(column_pred_probs), kind="stable").tolist()
            else:
                topk_column_ids = np.argsort(-np.array(column_pred_probs), kind="stable")[
                              :opt.topk_column_num].tolist()

            column_coverage_state_list.append(lista_contains_listb(topk_column_ids, used_column_ids))

    # record top-k1 tables and top-k2 columns for each table
    for table_id in topk_table_ids:
        new_table_info = dict()
        new_table_info["table_name_original"] = data["db_schema"][table_id]["table_name_original"]
        column_pred_probs = list(map(lambda x: round(x, 2), data["column_pred_probs"][table_id]))
        if opt.linking_mode == "None" or opt.linking_mode == "interaction":
            topk_column_ids = np.argsort(-np.array(column_pred_probs), kind="stable").tolist()
        else:
            topk_column_ids = np.argsort(-np.array(column_pred_probs), kind="stable")[:opt.topk_column_num].tolist()
        # print(topk_column_ids)
        # print(data["db_schema"][table_id]["column_names_original"])
        new_table_info["column_names_original"] = [
            data["db_schema"][table_id]["column_names_original"][column_id] for column_id in topk_column_ids]
        if opt.use_contents:
            new_table_info["db_contents"] = [data["db_schema"][table_id]["db_contents"][column_id] for column_id in
                                             topk_column_ids]

        ranked_data["db_schema"].append(new_table_info)

    # record foreign keys among selected tables
    table_names_original = [table["table_name_original"] for table in data["db_schema"]]
    needed_fks = []
    for fk in data["fk"]:
        source_table_id = table_names_original.index(fk["source_table_name_original"])
        target_table_id = table_names_original.index(fk["target_table_name_original"])
        if source_table_id in topk_table_ids and target_table_id in topk_table_ids:
            needed_fks.append(fk)
    ranked_data["fk"] = needed_fks

    input_sequence, output_sequence = prepare_input_and_output(opt, ranked_data, table_label, column_label)

    # record table_name_original.column_name_original for subsequent correction function during inference
    tc_original = []
    for table in ranked_data["db_schema"]:
        for column_name_original in table["column_names_original"] + ["*"]:
            tc_original.append(table["table_name_original"] + "." + column_name_original)
    print(output_sequence)
    output_dataset = {
            "db_id": data["db_id"],
            "input_sequence": input_sequence,
            "output_sequence": output_sequence,
            "tc_original": tc_original
        }

    return output_dataset

def start_demo():
    #preprocessing
    opt = parse_init()
    tables = json.load(open(opt.table_path, 'r'))
    tables = list2dict(tables)
    db_list = list(tables.keys())

    tablesnat = json.load(open(opt.tables_for_natsql, 'r'))
    table_dict_nat = dict()
    for t in tablesnat:
        table_dict_nat[t["db_id"]] = t
    #model init
    #classifier--------------------------------
    set_seed(opt.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    tokenizer_class = RobertaTokenizerFast
    # load tokenizer
    tokenizer_classifier = tokenizer_class.from_pretrained(
        opt.classifier_name_or_path,
        add_prefix_space=True
    )
    # initialize model
    model_classifier = MyClassifier(
        model_name_or_path=opt.classifier_name_or_path,
        vocab_size=len(tokenizer_classifier),
        mode=opt.mode
    )
    # load fine-tuned params
    model_classifier.load_state_dict(
        torch.load(opt.classifier_name_or_path + "/dense_classifier.pt", map_location=torch.device('cpu')), strict=False)
    if torch.cuda.is_available():
        model_classifier = model_classifier.cuda()
    model_classifier.eval()

    # t5--------------------------------
    # initialize tokenizer
    model_name_or_path = opt.model_name_or_path_nat if opt.use_natsql else opt.model_name_or_path
    print(model_name_or_path)
    tokenizer = T5TokenizerFast.from_pretrained(
        model_name_or_path,
        add_prefix_space=True
    )
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    model_class = T5ForConditionalGeneration
    # initialize model
    model = model_class.from_pretrained(model_name_or_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    return opt, db_list, tables, table_dict_nat, tokenizer_classifier, model_classifier, tokenizer, model

def inference_linking(nl:str,
                      db_id:str,
                      selected_schemas: list,
                      opt:argparse,
                      tables,
                      tokenizer_classifier,
                      model_classifier,
                      inter_label=None,
                    ):
    entry = preprocessing(opt, nl, db_id, tables)
    entry, selected_schemas, table_label, column_label = schema_item_classifier(entry, opt, selected_schemas, tokenizer_classifier, model_classifier, inter_label)
    entry = text2sql_data_generator(entry, opt, table_label, column_label)
    entry["selected_schemas"] = selected_schemas
    return entry


def normalization(sql):
    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])

        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                if in_quotation:
                    in_quotation = False
                else:
                    in_quotation = True

        return out_s

    # remove ";"
    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation
    def double2single(s):
        return s.replace("\"", "'")

    def add_asc(s):
        pattern = re.compile(
            r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")

        return s

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]

        tables_aliases = new_tables_aliases
        for k, v in tables_aliases.items():
            s = s.replace("as " + k + " ", "")
            s = s.replace(k, v)

        return s

    processing_func = lambda x: remove_table_alias(add_asc(lower(white_space_fix(double2single(remove_semicolon(x))))))

    return processing_func(sql)

if __name__ == '__main__':
    opt, db_list, tables, table_dict_nat, tokenizer_classifier, model_classifier, tokenizer_t5, model_t5 = start_demo()
    with open('data/spider/train_spider.json', 'r') as f:
        dataset = json.load(f)
    instruct = "Convert the following question to an SQL query using the following database schema."
    process_dataset = []
    sql_type = "natsql"
    if opt.linking_mode == "interaction":
        with open("data/output/interation_label.json", 'r') as f:
            inter_labels = json.load(f)
    for i, data in enumerate(dataset):
        inter_label = inter_labels[i] if opt.linking_mode == "interaction" else {}
        process_data = {}
        sql = data["query"].strip()
        norm_sql = normalization(sql).strip()
        nl = data["question"]
        db_id = data["db_id"]
        entry = inference_linking(nl, db_id, [], opt, tables, tokenizer_classifier, model_classifier, inter_label=inter_label)
        input_sequence = entry["input_sequence"]
        question = input_sequence.split("|")[0]
        schema = "|".join(input_sequence.split("|")[1:])
        input = "database schema:\n\n" + " " + db_id +" |" + schema + "\n\n\nquestion:\n\n" + question
        process_data["instruct"] = instruct
        process_data["input"] = input
        process_data["output"] = norm_sql
        process_dataset.append(process_data)
    with open('target/dataset4structlm/use_contents/train_original.json', 'w') as file:
        json.dump(process_dataset, file, indent=4)
