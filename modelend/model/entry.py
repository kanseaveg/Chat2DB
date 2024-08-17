import json
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from modelend.model.schema_item_classifier import *
from modelend.model.t5_inference import *
import numpy as np
from modelend.utils.args import parse_init
import argparse
import time
from transformers import RobertaTokenizerFast
from modelend.model.classifier_model import MyClassifier
from transformers.trainer_utils import set_seed
from transformers import T5TokenizerFast, T5ForConditionalGeneration, MT5ForConditionalGeneration, AutoTokenizer, AutoModel
from tokenizers import AddedToken
from modelend.utils.common_utils import list2dict, get_db_schemas
from modelend.utils.bridge_content_encoder import get_database_matches
from modelend.target.third_party.spider.evaluation import main as spider_evaluate
from modelend.target.third_party.test_suite.evaluation import evaluate as ts_evaluate
from modelend.model.get_db_schema import get_db_list, get_db_schema
from sentence_transformers import util as stb_util
from torch.utils.data import DataLoader
from datetime import datetime

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

    data = {"question":nl,"db_id":db_id}
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
    # print(table_label)
    # print(column_label)
    for table_id in range(len(ranked_data["db_schema"])):
        table_name_original = ranked_data["db_schema"][table_id]["table_name_original"]
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
                            # print(column_info)
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
                        # print(column_info)
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

    output_dataset = {
            "db_id": data["db_id"],
            "input_sequence": input_sequence,
            "output_sequence": output_sequence,
            "tc_original": tc_original
        }


    return output_dataset


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


def inference_sql(entry: list,
              opt: argparse,
              tokenizer,
              model,
              table_nat,
              ):

    sql = text2sql(entry, opt, tokenizer, model, table_nat)

    return sql


def start_demo():

    #preprocessing
    opt = parse_init()
    tables = json.load(open(opt.table_path, 'r'))
    tables = list2dict(tables)
    db_list = list(tables.keys())

    # tablesnat = json.load(open(opt.tables_for_natsql, 'r'))
    # table_dict_nat = dict()
    # for t in tablesnat:
        # table_dict_nat[t["db_id"]] = t
    table_dict_nat = None
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


def update_demo(selected_ckpt_path):
    opt = parse_init()
    # t5--------------------------------
    # initialize tokenizer
    print(selected_ckpt_path)
    tokenizer = T5TokenizerFast.from_pretrained(
        selected_ckpt_path,
        add_prefix_space=True
    )
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    model_class = T5ForConditionalGeneration
    # initialize model
    model = model_class.from_pretrained(selected_ckpt_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return tokenizer, model


async def adaptive_retraining(cur_state):
    settings = cur_state.settings
    PUBLIC_SQLITE_DATABASE_PATH = settings.PUBLIC_SQLITE_DATABASE_PATH
    PRIVATE_SQLITE_DATABASE_PATH = settings.PRIVATE_SQLITE_DATABASE_PATH
    ADAPTIVE_RETRAINING_THRESHOLD = settings.ADAPTIVE_RETRAINING_THRESHOLD
    ADAPTIVE_RETRAINING_BASE_CHECKPOINTS = settings.ADAPTIVE_RETRAINING_BASE_CHECKPOINTS
    ADAPTIVE_RETRAINING_BATCH_SIZE = settings.ADAPTIVE_RETRAINING_BATCH_SIZE
    ADAPTIVE_RETRAINING_EPOCHS = settings.ADAPTIVE_RETRAINING_EPOCHS
    ADAPTIVE_RETRAINING_SAVE_STEPS = settings.ADAPTIVE_RETRAINING_SAVE_STEPS
    
    stb_model = cur_state.stb_model
    stb_tokenizer = cur_state.stb_tokenizer
    
    
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # get db list 
    old_dbs = get_db_list(settings, PUBLIC_SQLITE_DATABASE_PATH)
    new_dbs = get_db_list(settings, PRIVATE_SQLITE_DATABASE_PATH)
    
    # get db schemas by each db id
    old_db_schemas = [get_db_schema(settings, db, PUBLIC_SQLITE_DATABASE_PATH) for db in old_dbs]
    new_db_schemas = [get_db_schema(settings, db, PRIVATE_SQLITE_DATABASE_PATH) for db in new_dbs]
    
    old_db_schemas_str_list = []
    for db_schema in old_db_schemas:
        cur_db_schemas = []
        for cur_db_schema in db_schema:
            cur_db_schemas.append(str(cur_db_schema))
        old_db_schemas_str_list.append(" ".join(cur_db_schemas))
        
    old_flatten_db_schemas_str_list = []
    for db_schema in old_db_schemas:
        cur_db_schemas = []
        for cur_db_schema in db_schema:
            cur_db_schemas.append(str(cur_db_schema))
        old_flatten_db_schemas_str_list.append(" | ".join(cur_db_schemas))
    
    new_db_schemas_str_list = []
    for db_schema in new_db_schemas:
        cur_db_schemas = []
        for cur_db_schema in db_schema:
            cur_db_schemas.append(str(cur_db_schema))
        new_db_schemas_str_list.append(" ".join(cur_db_schemas))
    
    old_db_schema_embeddings = []
    new_db_schema_embeddings = []
    logging.info("starting to process the database schema")
    with torch.no_grad():
        for index, old_db_schema in enumerate(old_db_schemas_str_list):
            old_tokenizer_output = stb_tokenizer(old_db_schema, padding=True, truncation=True, return_tensors="pt")
            old_output = stb_model(**old_tokenizer_output)
            old_emb = mean_pooling(old_output, old_tokenizer_output['attention_mask'])
            old_db_schema_embeddings.append(old_emb)
            logging.info(f"processing {index} old db schema")
        for index, new_db_schema in enumerate(new_db_schemas_str_list):
            new_tokenizer_output = stb_tokenizer(new_db_schema, padding=True, truncation=True, return_tensors="pt")
            new_output = stb_model(**new_tokenizer_output)
            new_emb = mean_pooling(new_output, new_tokenizer_output['attention_mask'])
            new_db_schema_embeddings.append(new_emb)
            logging.info(f"processing {index} new db schema")
            
    # calculate the similarity between old and new db schemas
    filter_index = []
    for index, old_emb in enumerate(old_db_schema_embeddings):
        for new_emb in new_db_schema_embeddings:
            res = torch.max(stb_util.pytorch_cos_sim(old_emb, new_emb))
            if res > ADAPTIVE_RETRAINING_THRESHOLD:
                filter_index.append(index)
                
    # extract useful training dataset
    filter_index = list(set(filter_index))
    filter_dbs = [old_dbs[index] for index in filter_index]
    filter_schemas = [old_flatten_db_schemas_str_list[index] for index in filter_index]

    # perform retraining!
    logging.info(f"filter_dbs: {filter_dbs}")
    
    dbs_to_schemas_map = {}
    for db, schema in zip(filter_dbs, filter_schemas):
        dbs_to_schemas_map[db] = schema
    
    with open("data/spider/train_spider.json", 'r') as f:
        train_spider = json.load(f)
    with open("data/spider/train_others.json", 'r') as f:
        train_others = json.load(f)
    train_dataset = train_spider + train_others
    filter_train = []
    for data in train_dataset:  
        # filter the most relevant training corpus
        if data["db_id"] in filter_dbs: 
            filter_train.append(data)
    filter_dataset = []
    for data in filter_train:
        db_id = data["db_id"]
        input_seq = data["question"] + " | " + db_id + " | " + dbs_to_schemas_map[db_id]
        output_seq  = data["query"]
        filter_dataset.append({"input_sequence": input_seq, "output_sequence": output_seq})
    
    retrain_dataloader = DataLoader(
        filter_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda x: x,
        drop_last=False
    )
    
    
    for ckpts in ADAPTIVE_RETRAINING_BASE_CHECKPOINTS:
        # retrain the model | make sure the ckpts folder has `config.json`
        if "t5" in ckpts:
            tokenizer = T5TokenizerFast.from_pretrained(ckpts)
            model = T5ForConditionalGeneration.from_pretrained(ckpts)
        else:
            tokenizer = AutoTokenizer.from_pretrained(ckpts)
            model = AutoModel.from_pretrained(ckpts)
        model.train()
        train_step = 0
        
        SAVE_PATH = "Retraining" + ckpts + " " + str(datetime.now().strftime('%Y%m%d%H%M')) 
        for epoch in range(ADAPTIVE_RETRAINING_EPOCHS):
            logging.info(f"This is epoch {epoch+1}.")
            for batch in retrain_dataloader:
                train_step += 1
                logging.info(f"Curreently on training step {train_step} of {ckpts}.")
                batch_inputs = [data["input_sequence"] for data in batch]
                batch_outputs = [data["output_sequence"] for data in batch]
                
                tokenized_inputs = tokenizer(
                    batch_inputs, 
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                
                with tokenizer.as_target_tokenizer():
                    tokenized_outputs = tokenizer(
                        batch_outputs, 
                        padding = "max_length", 
                        return_tensors = 'pt',
                        max_length = 256,
                        truncation = True
                    )
                
                encoder_input_ids = tokenized_inputs["input_ids"]
                encoder_input_attention_mask = tokenized_inputs["attention_mask"]

                decoder_labels = tokenized_outputs["input_ids"]
                decoder_labels[decoder_labels == tokenizer.pad_token_id] = -100
                decoder_attention_mask = tokenized_outputs["attention_mask"]

                if torch.cuda.is_available():
                    encoder_input_ids = encoder_input_ids.cuda()
                    encoder_input_attention_mask = encoder_input_attention_mask.cuda()
                    decoder_labels = decoder_labels.cuda()
                    decoder_attention_mask = decoder_attention_mask.cuda()
                
                model_outputs = model(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    labels = decoder_labels,
                    decoder_attention_mask = decoder_attention_mask,
                    return_dict = True
                )
                
                loss = model_outputs["loss"]
                loss.backward()
                
                if train_step % ADAPTIVE_RETRAINING_SAVE_STEPS == 0 and epoch >= 1:
                    logging.info(f"At {train_step} training step, save a checkpoint.")
                    os.makedirs(SAVE_PATH, exist_ok = True)
                    model.save_pretrained(save_directory = SAVE_PATH + "/checkpoint-{}".format(train_step))
                    text2sql_tokenizer.save_pretrained(save_directory = SAVE_PATH + "/checkpoint-{}".format(train_step))
        logging.info(f"Finish retraining the model {ckpts}.")
    
    return None
    


if __name__ == "__main__":
    mode = "evaluate"        # mode = predict or evaluate or runtime
    if mode == "predict":
        results = []
        linking_time = []
        model_time = []
        opt, db_list, tables, table_dict_nat, tokenizer_classifier, model_classifier, tokenizer_t5, model_t5 = start_demo()
        with open('data/spider/test_data/dev.json', 'r') as f:
            dev = json.load(f)
        dataset = dev
        if opt.linking_mode == "interaction":
            with open("data/output/interation_label_test.json", 'r') as f:
                inter_labels = json.load(f)
        for i, data in enumerate(tqdm(dataset)):
            nl = data["question"]
            db = data["db_id"]
            inter_label = inter_labels[i] if opt.linking_mode == "interaction" else {}
            start_time = time.time()
            entry = inference_linking(nl, db, [], opt, tables, tokenizer_classifier, model_classifier, inter_label)
            end_time = time.time()
            linking_time.append(end_time - start_time)

            start_time = time.time()
            sql = inference_sql([entry], opt, tokenizer_t5, model_t5, table_dict_nat)
            end_time = time.time()
            model_time.append(end_time - start_time)

            results.append(sql.replace("\n", "") + "\t" + db)
        # save predicted SQL results
        with open(opt.predict_result, 'w') as file:
            for item in results:
                file.write(str(item) + '\n')

        # # save inference time
        # with open('target/output_result/test/linking_time_use_contents_interaction.json', 'w') as file:
        #     json.dump(linking_time, file, indent=4)
        # with open('target/output_result/test/model_time_use_contents_interaction.json', 'w') as file:
        #     json.dump(model_time, file, indent=4)


    elif mode == "evaluate":
        def evaluation(prediction_args):
            # TODO: EM/EX
            if prediction_args["etype"] in ['all', 'exact']:
                kmaps = spider_evaluate(
                    prediction_args["gold_path"],
                    prediction_args["predict_path"],
                    prediction_args["db_path"],
                    prediction_args["raw_tables_path"],
                    'match',
                    prediction_args["result_path"]
                )
            if prediction_args["etype"] in ['all', 'exec']:
                ts_evaluate(
                    prediction_args["gold_path"],
                    prediction_args["predict_path"],
                    prediction_args["db_path"],
                    etype='all',
                    kmaps=kmaps,
                    plug_value=False,
                    keep_distinct=False,
                    progress_bar_for_each_datapoint=False
                )
        def inference_args(parse):
            """
               Arguments pertaining to execution.
               """
            args = {
                "etype": "all",
                "db_path": "data/spider/test_database",
                "gold_path": "data/spider/test_data/dev_gold.sql",
                "predict_path": parse,
                "raw_tables_path": "data/spider/test_data/tables.json",
                "result_path": "target/output_result/test/retrain_result.txt"
            }
            return args
        # calculate EM and EX
        args = inference_args("target/output_result/test/filter_0.5/checkpoint-4288.txt")
        evaluation(args)

    else:
        with open("data/spider/train_spider.json", 'r') as f:
            train = json.load(f)
        with open("data/spider/dev.json", 'r') as f:
            dev = json.load(f)
        dataset = train + dev
        with open('target/output_result/linking_time_use_contents_total.json', 'r') as f:
            linking = json.load(f)
        with open('target/output_result/model_time_use_contents_total.json', 'r') as f:
            model = json.load(f)
        small, medium, large, extra = [], [], [], []
        small_level = [0, 0.5]
        medium_level = [0.5, 1]
        large_level = [1, 5]
        for i, item in enumerate(tqdm(dataset)):
            db = item["db_id"]
            size = os.path.getsize(f"data/spider/database/{db}/{db}.sqlite")
            db_size = size/(1024 * 1024)
            # 分组平均
            if db_size > small_level[0] and db_size <=small_level[1]:
                small.append(linking[i]+ model[i])
            elif db_size > medium_level[0] and db_size <= medium_level[1]:
                medium.append(linking[i]+ model[i])
            elif db_size > large_level[0] and db_size <= large_level[1]:
                large.append(linking[i]+ model[i])
            elif db_size > large_level[1]:
                extra.append(linking[i]+ model[i])
            else:
                assert False, f"db_size = {db_size}"
        print(f"small : {small}")
        print(f"medium : {medium}")
        print(f"large : {large}")
        print(f"extra : {extra}")
        colors = ['#8E8BFE', '#E88482', '#FEA3A2', '#6F6F6F']
        data = [small, medium, large, extra]
        # total = [l + m for l, m in zip(linking, model)]
        # average = sum(total) / len(total)
        # std = np.std(total)
        # print(f"ART: {average} ± {std}")

        import matplotlib.pyplot as plt

        bp = plt.boxplot(data, showfliers=False,patch_artist=True,)
        # 为每组数据设置颜色
        for i, box in enumerate(bp['boxes']):
            box.set(facecolor=colors[i])

        # 添加标题和标签
        # plt.title('Boxplot Example')
        plt.xlabel('Database Size')
        plt.ylabel('Running Time (s)')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        # 自定义 x 轴刻度标签
        plt.xticks([1, 2, 3, 4], ['small', 'medium', 'large', 'extra'])

        plt.savefig('target/output_result/time_size.png',dpi=600)

